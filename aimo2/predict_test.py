from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest
from transformers import AutoTokenizer, ByT5Tokenizer
from vllm import LLM, CompletionOutput, RequestOutput, SamplingParams

from .predict import MetaLLM, RequestInput, get_prompts
from .systems import SystemParams, WeightedEnsemble


# Simplified chat template derived from https://huggingface.co/Qwen/QwQ-32B-Preview.
MOCK_CHAT_TEMPLATE = """{%- if messages[0]['role'] == 'system' %}
    {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
{%- else %}
    {{- '<|im_start|>system\nThink step-by-step.<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""


class MockTokenizer(ByT5Tokenizer):
    def __init__(self):
        super().__init__(chat_template=MOCK_CHAT_TEMPLATE)


@pytest.mark.parametrize(
    "tokenizer_factory",
    [
        pytest.param(lambda: MockTokenizer()),
        pytest.param(
            lambda: AutoTokenizer.from_pretrained("Qwen/QwQ-32B-Preview"),
            marks=pytest.mark.network,
        ),
    ],
)
def test_get_prompts(tokenizer_factory):
    llm = MagicMock()
    llm.get_tokenizer.return_value = tokenizer_factory()
    request_input = RequestInput()
    request_input = request_input.add_system_message("You are a helpful assistant.")
    request_input = request_input.add_user_message("Hello!")

    prompts = get_prompts([request_input], llm)

    expected_prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
        "\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
    )
    assert prompts == [expected_prompt]


class MockLLM(LLM):
    def __init__(self, responses: list[tuple[list[str], str]] = [], *args, **kwargs):
        """
        responses: A list of tuples of (keywords, response).
        The response is returned if all keywords are in the prompt.
        """
        super().__init__(*args, **kwargs)
        self.responses = responses
        self.request_id = 0

    def get_tokenizer(self):
        return MockTokenizer()

    def generate(  # type: ignore
        self,
        prompts: list[str] | None | Any = None,
        sampling_params: Any = None,
        use_tqdm: bool = True,
        lora_request: Any = None,
        prompt_adapter_request: Any = None,
        guided_options_request: Any = None,
        priority: Any = None,
    ) -> list[RequestOutput]:
        assert isinstance(prompts, list)
        outputs = []
        for prompt in prompts:
            assert isinstance(prompt, str)
            request_id = self.request_id
            answer = ""
            for keywords, response in self.responses:
                if all(keyword in prompt for keyword in keywords):
                    answer = response
                    break

            outputs.append(
                RequestOutput(
                    request_id=str(request_id),
                    prompt=prompt,
                    prompt_token_ids=[0],
                    prompt_logprobs=None,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text=answer,
                            token_ids=[0],
                            cumulative_logprob=1.0,
                            logprobs=None,
                            finish_reason="stop",
                            stop_reason=None,
                        )
                    ],
                    finished=True,
                )
            )
            self.request_id += 1
        return outputs


def test_meta_llm():
    llm = MockLLM(
        responses=[
            (["100 + 23?", "Box it!"], "\\boxed{123}"),
            (["1000 + 456?", "Box it!"], "\\boxed{1456}"),
            (["1000 + 456?", "Program it!"], "```python\n1000 + 456\n```"),
            (["100 + 23?", "Program it!"], "```python\n100 + 23\n```"),
        ]
    )

    greedy = SamplingParams(temperature=0.0, seed=42)
    system = WeightedEnsemble(
        system_params=[
            SystemParams(name="v0.0", message="Box it!", sampling_params=greedy),
            SystemParams(name="v0.1", message="Program it!", sampling_params=greedy),
        ],
        correct_answers={
            "a": 123,
            "b": 456,
        },
    )
    meta_llm = MetaLLM(llm=llm, system=system)

    result = meta_llm.predict(
        id_=pl.Series(["a", "b"]),
        question=pl.Series(["What is 100 + 23?", "What is 1000 + 456?"]),
    )
    assert result.to_dicts() == [{"id": "a", "answer": 123}, {"id": "b", "answer": 456}]
