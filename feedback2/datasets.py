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


def get_texts_df(dir_path):
    paths = [x for x in dir_path.iterdir() if x.is_file() and x.suffix == ".txt"]
    paths.sort(key=lambda path: path.stem)
    texts = []
    for path in paths:
        with open(path) as f:
            texts.append((path.stem, f.read().rstrip()))
    return pd.DataFrame(texts, columns=["id", "text"])


def get_dfs(path, path2):
    path = Path(path)
    path2 = Path(path2)
    dfs = {
        "feedback2": {
            "train": {
                "texts": get_texts_df(path2 / "train"),
                "df": pd.read_csv(path2 / "train.csv"),
            },
            "test": {
                "texts": get_texts_df(path2 / "test"),
                "df": pd.read_csv(path2 / "test.csv"),
            },
        },
        "feedback": {
            "train": {
                "texts": get_texts_df(path / "train"),
                "df": pd.read_csv(path / "train.csv").rename(
                    columns={"id": "essay_id"}
                ),
            },
        },
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


DISCOURSE_TYPES = [
    "Lead",
    "Position",
    "Evidence",
    "Claim",
    "Concluding Statement",
    "Counterclaim",
    "Rebuttal",
]

MAX_DISCOURSE_TYPES = len(DISCOURSE_TYPES)
DISCOURSE_TYPE_TO_ID = {t: i for i, t in enumerate(DISCOURSE_TYPES)}

LABELS = ["Ineffective", "Adequate", "Effective"]

MAX_LABELS = len(LABELS)
LABEL_TO_ID = {t: i for i, t in enumerate(LABELS)}


def get_targets(labels, overflow_to_sample):
    labels = [LABEL_TO_ID[label] for label in labels]
    labels = overflow_to_sample.new_tensor(labels)
    return labels[overflow_to_sample]


class Feedback2Dataset(Dataset):
    def __init__(
        self,
        texts,
        df,
        tokenizer,
        max_len,
        return_overflowing_tokens,
        stride,
        pad_to_multiple_of,
    ):
        self.texts = texts.set_index("id")
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.return_overflowing_tokens = return_overflowing_tokens
        self.stride = stride
        self.pad_to_multiple_of = pad_to_multiple_of

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text_id, discourse_text, discourse_type, label = self.df.loc[
            idx,
            ["essay_id", "discourse_text", "discourse_type", "discourse_effectiveness"],
        ]

        text = self.texts.loc[text_id, "text"]
        return text, discourse_text, discourse_type, label

    def get_collate_fn(self):
        def collate_fn(examples):
            texts, discourse_texts, discourse_types, labels = [
                list(a) for a in zip(*examples)
            ]
            discourses = [
                "{}\n{}".format(d_type, d_text)
                for d_type, d_text in zip(discourse_types, discourse_texts)
            ]
            inputs = self.tokenizer(
                discourses,
                texts,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                return_overflowing_tokens=self.return_overflowing_tokens,
                return_offsets_mapping=False,
                max_length=self.max_len,
                stride=self.stride,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            if not self.return_overflowing_tokens:
                x = inputs.input_ids
                inputs["overflow_to_sample_mapping"] = torch.arange(
                    x.size(0), dtype=torch.long, device=x.device
                )

            targets = get_targets(labels, inputs.overflow_to_sample_mapping)
            return len(examples), inputs, targets, labels

        return collate_fn


def score(preds_batch, labels_batch):
    scores = []
    preds = np.array(preds_batch)
    preds = preds / np.sum(preds, axis=-1, keepdims=True)
    preds = np.clip(preds, 1e-15, 1 - 1e-15)
    labels = np.array([[LABEL_TO_ID[label]] for label in labels_batch])
    scores = np.take_along_axis(preds, labels, axis=-1)
    scores = -np.log(scores)
    return np.mean(scores)
