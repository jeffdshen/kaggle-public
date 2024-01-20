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
    path_map = {k: Path(v) for k, v in path_map.items()}
    dfs = {
        "drcat": pd.read_csv(path_map["drcat"] / "train_v2_drcat_02.csv").rename(
            columns={"label": "generated"}
        ),
        "drcat_v3": pd.read_csv(path_map["drcat_v3"] / "train_v3_drcat_02.csv").rename(
            columns={"label": "generated"}
        ),
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


class DaigtDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, stride, pad_to_multiple_of):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride
        self.pad_to_multiple_of = pad_to_multiple_of

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, "text"]
        label = self.df.loc[idx, "generated"]

        return text, label

    def _get_inputs(self, texts):
        inputs = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            max_length=self.max_len,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        return inputs

    def get_collate_fn(self):
        def collate_fn(examples):
            texts, labels = [list(a) for a in zip(*examples)]
            inputs = self._get_inputs(texts)
            to_sample = inputs.overflow_to_sample_mapping
            targets = torch.tensor(
                labels, dtype=to_sample.dtype, device=to_sample.device
            )

            targets = targets.gather(0, to_sample)
            return len(examples), inputs, targets, labels

        return collate_fn
