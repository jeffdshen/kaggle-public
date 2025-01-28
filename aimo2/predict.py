from dataclasses import dataclass
from typing import Union

import polars as pl
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from vllm import LLM, RequestOutput

from .systems import RequestInput, System


HFTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


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
class MetaLLM:
    """LLM hooked up to a system."""

    llm: LLM
    system: System

    def _batch_generate(
        self, batched_requests: list[list[RequestInput]]
    ) -> list[list[RequestOutput]]:
        """Generate outputs for a batch of requests."""
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

    def _predict_runner(self, ids: list[str], questions: list[str]) -> list[int]:
        """Run the prediction loop."""
        predict_gens = [
            self.system.predict(id_, question) for id_, question in zip(ids, questions)
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
