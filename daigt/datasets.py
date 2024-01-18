import copy
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import datasets as hf_datasets
except ImportError:
    pass


def df_filter(df1, df2, col1, col2):
    return df1[~df1[col1].isin(df2[col2])].reset_index(drop=True)


def get_dfs(path_map):
    path_map = {k: Path(v) for k,v in path_map.items()}
    dfs = {
        "drcat": pd.read_csv(path_map["drcat"] / "train_v2_drcat_02.csv").rename(columns={"label": "generated"}),
        "drcat_v3": pd.read_csv(path_map["drcat_v3"] / "train_v3_drcat_02.csv").rename(columns={"label": "generated"}),
        "train": pd.read_csv(path_map["train"] / "train_essays.csv"),
        "test": pd.read_csv(path_map["test"] / "test_essays.csv"),
    }
    return dfs


def get_block_dataset(df, text_column, tokenizer, max_len, seed):
    if max_len is None:
        max_len = tokenizer.model_max_length
    dataset = hf_datasets.Dataset.from_pandas(df)
    dataset = dataset.shuffle(seed)

    def tokenize(examples):
        tokenized = tokenizer(
            examples[text_column], add_special_tokens=False, return_attention_mask=False
        )
        for tokens in tokenized.input_ids:
            tokens.append(tokenizer.sep_token_id)
        return tokenized

    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing examples...",
    )

    def blockify(examples):
        all = []
        for sub in examples["input_ids"]:
            all.extend(sub)
        sub_max = max_len - 2
        block_starts = range(0, len(all) - sub_max + 1, sub_max)
        blocks = [
            tokenizer.build_inputs_with_special_tokens(all[i : i + sub_max])
            for i in block_starts
        ]
        examples = {"input_ids": blocks}
        return examples

    dataset = dataset.map(
        blockify,
        batched=True,
        desc="Chunking examples...",
    )
    return dataset


def overlap(a, b, c, d):
    return a < d and c < b
