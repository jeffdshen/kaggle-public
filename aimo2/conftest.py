import sys

from vllm_mock import CompletionOutput, LLM, RequestOutput, SamplingParams

module = type(sys)("vllm")
module.LLM = LLM
module.CompletionOutput = CompletionOutput
module.RequestOutput = RequestOutput
module.SamplingParams = SamplingParams
sys.modules["vllm"] = module
