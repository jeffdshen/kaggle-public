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


def df_filter(df1, df2, col1, col2):
    return df1[~df1[col1].isin(df2[col2])].reset_index(drop=True)


def get_dfs(path, path2, path3):
    path = Path(path) if path is not None else None
    path2 = Path(path2) if path2 is not None else None
    path3 = Path(path3) if path3 is not None else None

    dfs = {
        "feedback3_train": {
            "texts": pd.read_csv(path3 / "train.csv").rename(
                columns={"text_id": "id", "full_text": "text"}
            )
        },
        "feedback3_test": {
            "texts": pd.read_csv(path3 / "test.csv")
            .rename(columns={"text_id": "id", "full_text": "text"})
            .assign(**{label: 5 for label in LABEL_TYPES})
        },
    }

    if path2 is None:
        return dfs
    dfs.update(
        {
            "feedback2_train": {
                "texts": get_texts_df(path2 / "train"),
                "df": pd.read_csv(path2 / "train.csv"),
            },
            "feedback2_test": {
                "texts": get_texts_df(path2 / "test"),
                "df": pd.read_csv(path2 / "test.csv").assign(
                    discourse_effectiveness="Ineffective"
                ),
            },
        }
    )

    if path is None:
        return dfs

    dfs["feedback_train"] = {
        "texts": get_texts_df(path / "train"),
        "df": pd.read_csv(path / "train.csv")
        .rename(columns={"id": "essay_id"})
        .assign(discourse_effectiveness="Ineffective"),
    }

    texts2 = dfs["feedback2_train"]["texts"]
    texts1 = dfs["feedback_train"]["texts"]
    df1 = dfs["feedback_train"]["df"]
    dfs["feedback_train"]["texts"] = df_filter(texts1, texts2, "id", "id")
    dfs["feedback_train"]["df"] = df_filter(df1, texts2, "essay_id", "id")
    return dfs


def get_label_dfs(path):
    path = Path(path)
    label_dfs = {}
    for f in path.glob("*.csv"):
        label_dfs[f.stem] = pd.read_csv(f)
    return label_dfs


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


def reencode(text):
    try:
        return text.encode("latin1").decode("cp1252")
    except:
        return text


def normalize(df, col):
    df = df.copy(deep=True)
    df[col] = df[col].map(reencode)
    df[col] = df[col].str.replace("\xa0", " ", regex=False)
    df[col] = df[col].str.strip()
    return df


LABEL_TYPES = [
    "cohesion",
    "syntax",
    "vocabulary",
    "phraseology",
    "grammar",
    "conventions",
]


def extract_offsets(prompt, left, right):
    sections = []
    offsets = []
    start = 0
    offset = 0
    while start < len(prompt):
        end = prompt.find(left, start)
        if end == -1:
            sections.append(prompt[start:])
            break

        sections.append(prompt[start:end])
        offset += end - start
        start = end + len(left)

        end = prompt.find(right, start)
        sections.append(prompt[start:end])
        offsets.append((offset, offset + end - start))
        offset += end - start
        start = end + len(right)

    return "".join(sections), offsets


def overlap(a, b, c, d):
    return a < d and c < b


def get_target_mask(inputs, offsets_batch):
    target_masks = []
    idxs = defaultdict(int)
    for tokens, sample in zip(
        inputs.offset_mapping.tolist(), inputs.overflow_to_sample_mapping.tolist()
    ):
        target_mask = []
        for token in tokens:
            idx = idxs[sample]
            offsets = offsets_batch[sample]
            if idx >= len(offsets):
                target_mask.append(0)
                continue

            offset = offsets[idx]
            if not overlap(token[0], token[1], offset[0], offset[1]):
                target_mask.append(0)
                continue

            target_mask.append(1)
            idxs[sample] += 1

        target_masks.append(target_mask)
    target_masks = inputs.overflow_to_sample_mapping.new_tensor(target_masks)
    return target_masks


LABELS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

MAX_LABELS = len(LABELS)
LABEL_TO_ID = {t: i for i, t in enumerate(LABELS)}


def get_flat_targets(labels, overflow_to_sample, target_type):
    if target_type == "logit":
        labels = [LABEL_TO_ID[label] for label in labels]
        labels = overflow_to_sample.new_tensor(labels)
    elif target_type == "soft":
        labels = np.array(labels)
        labels = overflow_to_sample.new_tensor(labels, dtype=torch.float)
    elif target_type == "linear":
        labels = np.array(labels)
        labels = overflow_to_sample.new_tensor(labels, dtype=torch.float)
    else:
        raise ValueError("Unrecognized target_type: {}".format(target_type))
    return labels[overflow_to_sample]


def get_hard_labels(labels):
    return np.array(LABELS)[np.argmax(labels, axis=-1)].tolist()


def get_targets(labels, target_mask, target_type):
    device = target_mask.device
    target_overflow = torch.arange(len(labels), dtype=torch.long, device=device)
    flat_targets = get_flat_targets(labels, target_overflow, target_type)
    if target_type == "logit":
        targets = torch.zeros(*target_mask.size(), dtype=torch.long, device=device)
    elif target_type == "soft":
        targets = torch.zeros(
            *target_mask.size(), MAX_LABELS, dtype=torch.float, device=device
        )
    elif target_type == "linear":
        targets = torch.zeros(*target_mask.size(), dtype=torch.float, device=device)
    else:
        raise ValueError("Unrecognized target_type: {}".format(target_type))
    targets[target_mask.bool()] = flat_targets
    return targets


# TODO Add label dfs
class Feedback3Dataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        max_len,
        pad_to_multiple_of,
        normalize_text,
        target_type,
        prompt,
        left_prompt_marker,
        right_prompt_marker,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_to_multiple_of = pad_to_multiple_of
        self.target_type = target_type
        self.prompt, self.offsets = extract_offsets(
            prompt, left_prompt_marker, right_prompt_marker
        )

        if normalize_text:
            self.df = normalize(self.df, "text")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, "text"]
        label = list(self.df.loc[idx, LABEL_TYPES])

        return self.prompt + text, self.offsets, label

    def _get_inputs(self, texts):
        inputs = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            max_length=self.max_len,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        x = inputs.input_ids
        inputs["overflow_to_sample_mapping"] = torch.arange(
            x.size(0), dtype=torch.long, device=x.device
        )
        return inputs

    def get_collate_fn(self):
        def collate_fn(examples):
            texts, offsets, labels = [list(a) for a in zip(*examples)]
            labels = np.concatenate(labels).tolist()
            inputs = self._get_inputs(texts)
            target_mask = get_target_mask(inputs, offsets)
            targets = get_targets(labels, target_mask, self.target_type)
            inputs["target_mask"] = target_mask

            if self.target_type == "soft":
                labels = get_hard_labels(labels)
            return len(examples), inputs, targets, labels

        return collate_fn


def score_raw(preds_batch, labels_batch):
    preds = np.array(preds_batch)
    labels = np.array(labels_batch)
    mse = (preds - labels) ** 2
    mse = np.reshape(mse, (-1, len(LABEL_TYPES)))
    return mse


def score(preds_batch, labels_batch):
    return np.mean(score_raw(preds_batch, labels_batch), axis=0)
