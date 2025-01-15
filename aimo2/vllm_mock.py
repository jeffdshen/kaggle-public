"""Mock VLLM classes for testing.

We do this because it is difficult to install VLLM in some environments.
"""

import copy
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union


class LLM:
    def __init__(self, *args, **kwargs):
        pass


@dataclass
class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index: The index of the output in the request.
        text: The generated output text.
        token_ids: The token IDs of the generated output text.
        cumulative_logprob: The cumulative log probability of the generated
            output text.
        logprobs: The log probabilities of the top probability words at each
            position if the logprobs are requested.
        finish_reason: The reason why the sequence is finished.
        stop_reason: The stop string or token id that caused the completion
            to stop, None if the completion finished for some other reason
            including encountering the EOS token.
        lora_request: The LoRA request that was used to generate the output.
    """

    index: int
    text: str
    token_ids: Sequence[int]
    cumulative_logprob: Optional[float]
    # logprobs: Optional[SampleLogprobs]
    logprobs: None = None
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None
    # lora_request: Optional[LoRARequest] = None
    lora_request: None = None


@dataclass
class RequestOutput:
    request_id: str
    prompt: Optional[str]
    prompt_token_ids: Optional[List[int]]
    # prompt_logprobs: Optional[PromptLogprobs]
    prompt_logprobs: None
    outputs: List[CompletionOutput]
    finished: bool
    # metrics: Optional[RequestMetrics] = None
    # lora_request: Optional[LoRARequest] = None
    encoder_prompt: Optional[str] = None
    encoder_prompt_token_ids: Optional[List[int]] = None
    num_cached_tokens: Optional[int] = None


@dataclass
class SamplingParams:
    n: int = 1
    best_of: Optional[int] = None
    _real_n: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    bad_words: Optional[List[str]] = None
    ignore_eos: bool = False
    max_tokens: Optional[int] = 16
    min_tokens: int = 0
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    # NOTE: This parameter is only exposed at the engine level for now.
    # It is not exposed in the OpenAI API server, as the OpenAI API does
    # not support returning only a list of token IDs.
    detokenize: bool = True
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    # Optional[List[LogitsProcessor]] type. We use Any here because
    # Optional[List[LogitsProcessor]] type is not supported by msgspec.
    # logits_processors: Optional[Any] = None
    include_stop_str_in_output: bool = False
    # truncate_prompt_tokens: Optional[Annotated[int, msgspec.Meta(ge=1)]] = None
    # output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE

    # The below fields are not supposed to be used as an input.
    # They are set in post_init.
    output_text_buffer_length: int = 0
    # _all_stop_token_ids: Set[int] = msgspec.field(default_factory=set)

    # Fields used to construct logits processors
    # guided_decoding: Optional[GuidedDecodingParams] = None
    # logit_bias: Optional[Dict[int, float]] = None
    allowed_token_ids: Optional[List[int]] = None

    def clone(self):
        return copy.deepcopy(self)
