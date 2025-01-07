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

    message: str
    """The message of the system prompt."""

    sampling_params: SamplingParams
    """The sampling parameters."""

    def to_request_input(self) -> RequestInput:
        request_input = RequestInput(
            sampling_params=self.sampling_params,
        )
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
    time_per_question: float = 0.0
    full_timeout: float = 0.0
    correct_answers: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.start_time = time.time()
        self.num_questions = 0

    def predict_question(self, id_: str, question: str) -> int:
        time_elapsed = time.time() - self.start_time
        self.num_questions += 1
        if time_elapsed > self.full_timeout:
            return 0

        request_inputs = [
            system_params.to_request_input().add_user_message(question)
            for system_params in self.system_params
        ]
        prompts = get_prompts(request_inputs, self.llm)
        sampling_params = [
            request_input.sampling_params for request_input in request_inputs
        ]
        request_outputs = self.llm.generate(prompts, sampling_params=sampling_params)
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
                        input_text=prompts[i],
                        output_text=request_output.outputs[0].text,
                        answers=answers[i],
                        correct_answer=self.correct_answers.get(id_),
                    ).to_json()
                )
                self.question_log.write("\n")
        return max_weighted_vote(answers)

    def predict(
        self, id_: pl.DataFrame, question: pl.DataFrame
    ) -> pl.DataFrame | pd.DataFrame:
        id_item = id_.item(0)
        question_item = question.item(0)
        answer = self.predict_question(id_item, question_item)
        return pl.DataFrame({"id": id_item, "answer": answer})
