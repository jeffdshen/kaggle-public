"""Config for AIMO2. This is setup for my specific configuration."""

import os
from dataclasses import dataclass
from functools import cached_property

import polars as pl
from vllm import SamplingParams

from .systems import SystemParams


def get_environment() -> str:
    if os.getenv("KAGGLE_KERNEL_RUN_TYPE") != None:
        return "kaggle"
    elif os.getenv("COLAB_RELEASE_TAG") != None:
        return "colab"
    else:
        return "local"


ENV = get_environment()


@dataclass
class ValidationData:
    reference_df: pl.DataFrame
    aime_df: pl.DataFrame

    @cached_property
    def df(self) -> pl.DataFrame:
        aime_df = self.aime_df.select(
            [
                pl.concat_str(
                    [pl.lit("AIME_"), pl.col("id").cast(pl.Utf8).str.zfill(3)]
                ).alias("id"),
                pl.col("problem"),
                pl.col("answer").cast(pl.Int64).alias("answer"),
            ]
        )
        reference_df = self.reference_df.select(
            pl.concat_str([pl.lit("REF_"), pl.col("id")]).alias("id"),
            pl.col("problem"),
            pl.col("answer"),
        )

        df = pl.concat([aime_df, reference_df], how="vertical")
        return df

    @cached_property
    def correct_answers(self) -> dict[str, int]:
        return dict(self.df.select("id", "answer").iter_rows())


def get_reference_df():
    """Returns the reference dataset from local file."""
    if ENV == "kaggle":
        return pl.read_csv(
            "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv"
        )
    elif ENV == "local":
        parent = os.path.dirname(__file__)
        return pl.read_csv(
            f"{parent}/ai-mathematical-olympiad-progress-prize-2/reference.csv"
        )
    elif ENV == "colab":
        raise NotImplementedError("Not yet implemented")
    else:
        raise ValueError("Unknown environment")


def get_aime_df():
    """Returns the AIME dataset from NuminaMath."""
    if ENV == "kaggle":
        return pl.read_parquet(
            "/kaggle/input/ai-mo-aimo-validation-aime/data/train-00000-of-00001.parquet"
        )
    elif ENV in ["local", "colab"]:
        return pl.read_parquet(
            "hf://datasets/AI-MO/aimo-validation-aime/data/train-00000-of-00001.parquet"
        )
    else:
        raise ValueError("Unknown environment")


def get_validation_data():
    return ValidationData(get_reference_df(), get_aime_df())


def load_llm(model: str):
    from vllm import LLM

    llm = LLM(
        model,
        max_model_len=32768,
        max_num_seqs=32,
        trust_remote_code=True,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
    )
    return llm


MODELS = {
    "qwen-72b-instruct-awq": "/kaggle/input/qwen2.5/transformers/72b-instruct-awq/1",
    "qwq-32b-preview": "/kaggle/input/qwq-32b-preview/transformers/default/1",
    "qwq-32b-preview-awq": "/kaggle/input/m/shelterw/qwen2.5/transformers/qwq-32b-preview-awq/1",
    "deepseek-r1-distill-qwen-32b": "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-32b/1",
    "deepseek-r1-distill-qwen-32b-awq": "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-32b-awq/1",
}


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
    "r1": SamplingParams(
        temperature=0.6,
        top_p=0.95,
        skip_special_tokens=True,
        max_tokens=32768,
        seed=42,
    ),
    "r1_short": SamplingParams(
        temperature=0.6,
        top_p=0.95,
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
    SystemParams(
        name="r1_v1a",
        message="Please reason step by step, and put your final answer within \\boxed{}.",
        sampling_params=SAMPLING_PARAMS["greedy"],
        question_format="{question}\n\nWhat is the answer modulo 1000?",
    ),
    SystemParams(
        name="r1_v1b",
        message="Please reason step by step, and put your final answer within \\boxed{}.",
        sampling_params=SAMPLING_PARAMS["r1"],
        question_format="{question}\n\nWhat is the answer modulo 1000?",
    ),
    SystemParams(
        name="r1_v1c",
        message="Please reason step by step, and put your final answer within \\boxed{}.",
        sampling_params=SAMPLING_PARAMS["greedy_short"],
        question_format="{question}\n\nWhat is the answer modulo 1000?",
    ),
    SystemParams(
        name="r1_v2a",
        message="",
        sampling_params=SAMPLING_PARAMS["greedy"],
        question_format="{question}\n\n"
        "What is the answer modulo 1000? "
        "Please reason step by step, and put your final answer within \\boxed{{}}.",
    ),
    SystemParams(
        name="r1_v2b",
        message="",
        sampling_params=SAMPLING_PARAMS["r1"],
        question_format="{question}\n\n"
        "What is the answer modulo 1000? "
        "Please reason step by step, and put your final answer within \\boxed{{}}.",
    ),
    SystemParams(
        name="r1_v2c",
        message="",
        sampling_params=SAMPLING_PARAMS["greedy_short"],
        question_format="{question}\n\n"
        "What is the answer modulo 1000? "
        "Please reason step by step, and put your final answer within \\boxed{{}}.",
    ),
]


SYSTEM_PARAMS = {
    system_params.name: system_params for system_params in SYSTEM_PARAMS_LIST
}
