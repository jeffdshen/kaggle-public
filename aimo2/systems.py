import json
import time
from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from dataclasses import asdict, dataclass, field
from typing import TextIO

from vllm import RequestOutput, SamplingParams

from .postprocess import convert_texts_to_int, extract_boxed_texts, max_weighted_vote


@dataclass(frozen=True)
class RequestInput:
    conversation: list[dict] = field(default_factory=list)
    sampling_params: SamplingParams = field(default_factory=SamplingParams)

    def add_request_output(self, request_output: RequestOutput) -> "RequestInput":
        message = request_output.outputs[0].text
        conversation = self.conversation + [{"role": "assistant", "content": message}]
        return RequestInput(
            conversation=conversation,
            sampling_params=self.sampling_params,
        )

    def add_user_message(self, message: str) -> "RequestInput":
        conversation = self.conversation + [{"role": "user", "content": message}]
        return RequestInput(
            conversation=conversation,
            sampling_params=self.sampling_params,
        )

    def add_system_message(self, message: str) -> "RequestInput":
        conversation = self.conversation + [{"role": "system", "content": message}]
        return RequestInput(
            conversation=conversation,
            sampling_params=self.sampling_params,
        )

    def with_seed(self, seed: int) -> "RequestInput":
        sampling_params = self.sampling_params.clone()
        sampling_params.seed = seed
        return RequestInput(
            conversation=self.conversation,
            sampling_params=sampling_params,
        )


class System(ABC):
    """A system that interacts with a model."""

    @abstractmethod
    def predict(
        self, id_: str, question: str
    ) -> Generator[list[RequestInput], list[RequestOutput], int]:
        """Predict the answer to a question.

        The generator yields RequestInputs which are then sent to the model.
        The generator receives RequestOutputs from the model.
        The generator returns the final answer.
        """
        pass


def boxed_extract(request_output: RequestOutput) -> list[int]:
    boxed = extract_boxed_texts(request_output.outputs[0].text)
    answer = convert_texts_to_int(boxed)
    answer = [a % 1000 for a in answer]
    return answer


EXTRACTORS = {
    "boxed": boxed_extract,
}


@dataclass(frozen=True)
class SystemParams:
    name: str
    """The name of the prompt."""

    sampling_params: SamplingParams
    """The sampling parameters."""

    message: str | None = None
    """The message of the prompt."""

    question_format: str | None = None
    """The format of the question. Passes in the question as {question}."""

    extract: str = "boxed"
    """The extraction function. If None, use boxed."""

    def make_input(self, question: str) -> RequestInput:
        """Convert the system params to a request input."""
        request_input = RequestInput(
            sampling_params=self.sampling_params,
        )
        if self.message is not None:
            request_input = request_input.add_system_message(self.message)
        if self.question_format is not None:
            question = self.question_format.format(question=question)
        return request_input.add_user_message(question)

    def extract_answers(self, request_output: RequestOutput) -> list[int]:
        """Extract the answers from the request output."""
        return EXTRACTORS[self.extract](request_output)


@dataclass
class RequestLogRecord:
    id: str
    """The id of the question."""

    system_name: str
    """The name of the system."""

    input_text: str
    """The input text to the LLM."""

    output_text: str
    """The output text of the question."""

    answers: list[int]
    """The extracted answers of the question."""

    correct_answer: int | None
    """The correct answer of the question."""

    token_ids: Sequence[int]
    """The token ids of the output text."""

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class WeightedEnsemble(System):
    system_params: list[SystemParams]
    question_log: TextIO | None = None
    time_per_question: float | None = None
    full_timeout: float | None = None
    start_time: float = field(default=time.time())
    correct_answers: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.num_questions = 0

    def predict(
        self, id_: str, question: str
    ) -> Generator[list[RequestInput], list[RequestOutput], int]:
        time_elapsed = time.time() - self.start_time
        self.num_questions += 1
        if self.full_timeout is not None and time_elapsed > self.full_timeout:
            return 0

        request_inputs = [params.make_input(question) for params in self.system_params]
        request_outputs: list[RequestOutput] = yield request_inputs
        assert len(request_outputs) == len(request_inputs)
        answers = [
            params.extract_answers(request_output)
            for params, request_output in zip(self.system_params, request_outputs)
        ]

        if self.question_log:
            for i, request_output in enumerate(request_outputs):
                self.question_log.write(
                    RequestLogRecord(
                        id=id_,
                        system_name=self.system_params[i].name,
                        input_text=request_output.prompt or "",
                        output_text=request_output.outputs[0].text,
                        answers=answers[i],
                        correct_answer=self.correct_answers.get(id_),
                        token_ids=request_output.outputs[0].token_ids,
                    ).to_json()
                )
                self.question_log.write("\n")
        return max_weighted_vote(answers)
