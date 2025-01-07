import os

from vllm import LLM


def setup_environment():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_qwq(model: str) -> LLM:
    llm = LLM(
        model,
        max_model_len=32768,
        trust_remote_code=True,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.96,
    )
    return llm


MODELS = {
    "qwen-72b-instruct-awq": "/kaggle/input/qwen2.5/transformers/72b-instruct-awq/1",
    "qwq-32b-preview": "/kaggle/input/qwq-32b-preview/transformers/default/1",
    "qwq-32b-preview-awq": "/kaggle/input/m/shelterw/qwen2.5/transformers/qwq-32b-preview-awq/1",
}
