from collections.abc import Generator
from dataclasses import asdict, dataclass, field
import json
import time
from typing import TextIO, Union

import pandas as pd
import polars as pl
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from vllm import LLM, RequestOutput, SamplingParams

from .postprocess import convert_texts_to_int, extract_boxed_texts, max_weighted_vote

HFTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


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


def get_prompts(requests: list[RequestInput], llm: LLM) -> list[str]:
    tokenizer = llm.get_tokenizer()
    assert isinstance(tokenizer, HFTokenizer)
    prompts = [
        tokenizer.apply_chat_template(
            request.conversation, tokenize=False, add_generation_prompt=True
        )
        for request in requests
    ]
    assert all(isinstance(p, str) for p in prompts)
    prompts = [p for p in prompts if isinstance(p, str)]
    return prompts


@dataclass
class SystemParams:
    name: str
    """The name of the system prompt."""

    message: str | None
    """The message of the system prompt."""

    sampling_params: SamplingParams
    """The sampling parameters."""

    def to_request_input(self) -> RequestInput:
        request_input = RequestInput(
            sampling_params=self.sampling_params,
        )
        if self.message is None:
            return request_input
        return request_input.add_system_message(self.message)


SAMPLING_PARAMS = {
    "min_p": SamplingParams(
        min_p=0.01,
        skip_special_tokens=True,
        max_tokens=32768,
        seed=42,
    ),
    "uniform": SamplingParams(
        temperature=1.0,
        skip_special_tokens=True,
        max_tokens=32768,
        seed=42,
    ),
    "greedy": SamplingParams(
        temperature=0.0,
        skip_special_tokens=True,
        max_tokens=32768,
        seed=42,
    ),
    "greedy_short": SamplingParams(
        temperature=0.0,
        skip_special_tokens=True,
        max_tokens=8192,
        seed=42,
    ),
}

SYSTEM_PARAMS_LIST = [
    SystemParams(
        name="v1",
        message="Please use chained reasoning to put the answer in \\boxed{}.",
        sampling_params=SAMPLING_PARAMS["min_p"],
    ),
    SystemParams(
        name="v2",
        message="Please reflect and verify while reasoning and put the answer in \\boxed{}.",
        sampling_params=SAMPLING_PARAMS["min_p"],
    ),
    SystemParams(
        name="v3",
        message="Solve the following problem using concise and clear reasoning by placing the answer in \\boxed{}.",
        sampling_params=SAMPLING_PARAMS["min_p"],
    ),
    SystemParams(
        name="v4",
        message="You are a helpful and reflective maths assistant, please reason step by step to put the answer in \\boxed{}.",
        sampling_params=SAMPLING_PARAMS["min_p"],
    ),
    SystemParams(
        name="v5",
        message="You are the smartest maths expert in the world, please spike this question and put the answer in \\boxed{}.",
        sampling_params=SAMPLING_PARAMS["min_p"],
    ),
    SystemParams(
        name="default_v1", message=None, sampling_params=SAMPLING_PARAMS["greedy_short"]
    ),
    SystemParams(
        name="default_v2", message=None, sampling_params=SAMPLING_PARAMS["greedy"]
    ),
    SystemParams(
        name="default_v3",
        message="You are a helpful and harmless assistant. "
        "You are Qwen developed by Alibaba. "
        "You should think step-by-step. "
        "If uncertain, answer \\boxed{N/A}.",
        sampling_params=SAMPLING_PARAMS["greedy_short"],
    ),
    SystemParams(
        name="think_v1",
        message="Think step-by-step. Put the answer in \\boxed{}.",
        sampling_params=SAMPLING_PARAMS["greedy_short"],
    ),
    SystemParams(
        name="think_v2",
        message="Think step-by-step. Put the answer in \\boxed{}. "
        "If uncertain, answer \\boxed{N/A}.",
        sampling_params=SAMPLING_PARAMS["greedy_short"],
    ),
]


SYSTEM_PARAMS = {
    system_params.name: system_params for system_params in SYSTEM_PARAMS_LIST
}


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

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class MetaLLM:
    llm: LLM
    system_params: list[SystemParams]
    question_log: TextIO | None = None
    time_per_question: float | None = None
    full_timeout: float | None = None
    start_time: float = field(default=time.time())
    correct_answers: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.num_questions = 0

    def _batch_generate(
        self, batched_requests: list[list[RequestInput]]
    ) -> list[list[RequestOutput]]:
        request_inputs = [
            request for requests in batched_requests for request in requests
        ]
        prompts = get_prompts(request_inputs, self.llm)
        sampling_params = [
            request_input.sampling_params for request_input in request_inputs
        ]
        request_outputs = self.llm.generate(prompts, sampling_params=sampling_params)

        if len(request_inputs) != len(request_outputs):
            raise ValueError(f"{len(request_inputs)=} != {len(request_outputs)=}")

        output_iter = iter(request_outputs)
        batched_outputs: list[list[RequestOutput]] = []
        for requests in batched_requests:
            batched_outputs.append([next(output_iter) for _ in requests])
        return batched_outputs

    def _predict(
        self, id_: str, question: str
    ) -> Generator[list[RequestInput], list[RequestOutput], int]:
        time_elapsed = time.time() - self.start_time
        self.num_questions += 1
        if self.full_timeout is not None and time_elapsed > self.full_timeout:
            return 0

        request_inputs = [
            system_params.to_request_input().add_user_message(question)
            for system_params in self.system_params
        ]
        request_outputs: list[RequestOutput] = yield request_inputs
        assert len(request_outputs) == len(request_inputs)
        boxed = [
            extract_boxed_texts(request_output.outputs[0].text)
            for request_output in request_outputs
        ]
        answers = [convert_texts_to_int(text) for text in boxed]
        answers = [[a % 1000 for a in answer] for answer in answers]

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
                    ).to_json()
                )
                self.question_log.write("\n")
        return max_weighted_vote(answers)

    def _predict_runner(self, ids: list[str], questions: list[str]) -> list[int]:
        predict_gens = [
            self._predict(id_, question) for id_, question in zip(ids, questions)
        ]
        answers: list[int] = [0] * len(ids)
        total_answers = 0
        request_inputs: list[list[RequestInput]] = []
        for i, predict_gen in enumerate(predict_gens):
            try:
                request_inputs.append(next(predict_gen))
            except StopIteration as e:
                assert isinstance(e.value, int)
                answers[i] = e.value
                total_answers += 1
                request_inputs.append([])

        while total_answers < len(ids):
            request_outputs = self._batch_generate(request_inputs)

            request_inputs = []
            for i, (request_output, predict_gen) in enumerate(
                zip(request_outputs, predict_gens)
            ):
                if not request_output:
                    request_inputs.append([])
                    continue

                try:
                    request_inputs.append(predict_gen.send(request_output))
                except StopIteration as e:
                    assert isinstance(e.value, int)
                    answers[i] = e.value
                    total_answers += 1
                    request_inputs.append([])

        return answers

    def predict(self, id_: pl.Series, question: pl.Series) -> pl.DataFrame:
        answers = self._predict_runner(id_.to_list(), question.to_list())
        return pl.DataFrame({"id": id_, "answer": answers})
