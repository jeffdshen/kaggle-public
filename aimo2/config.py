"""Config for AIMO2. This is setup for my specific configuration."""

from functools import cached_property
import os
from dataclasses import dataclass

import polars as pl


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
    """Returns the AIME dataset. Requires internet access."""
    return pl.read_parquet(
        "hf://datasets/AI-MO/aimo-validation-aime/data/train-00000-of-00001.parquet"
    )


def get_validation_data():
    return ValidationData(get_reference_df(), get_aime_df())
