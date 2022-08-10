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


def get_dfs(path, path2):
    path = Path(path) if path is not None else None
    path2 = Path(path2)
    dfs = {
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
DISCOURSE_TYPE_CODE = [
    "lead",
    "position",
    "evidence",
    "claim",
    "conclusion",
    "counterclaim",
    "rebuttal",
]
START_LEFT = "<"
START_RIGHT = ">"
END_LEFT = "</"
END_RIGHT = ">"


LABELS = ["Ineffective", "Adequate", "Effective"]

MAX_LABELS = len(LABELS)
LABEL_TO_ID = {t: i for i, t in enumerate(LABELS)}


def get_targets(labels, overflow_to_sample, soft):
    if not soft:
        labels = [LABEL_TO_ID[label] for label in labels]
        labels = overflow_to_sample.new_tensor(labels)
    else:
        labels = np.array(labels)
        labels = overflow_to_sample.new_tensor(labels, dtype=torch.float)
    return labels[overflow_to_sample]


def get_hard_labels(labels):
    return np.array(LABELS)[np.argmax(labels, axis=-1)].tolist()


def get_targets_combined(labels, target_mask, soft):
    device = target_mask.device
    target_overflow = torch.arange(len(labels), dtype=torch.long, device=device)
    flat_targets = get_targets(labels, target_overflow, soft)
    if soft:
        targets = torch.zeros(
            *target_mask.size(), MAX_LABELS, dtype=torch.float, device=device
        )
    else:
        targets = torch.zeros(*target_mask.size(), dtype=torch.long, device=device)
    targets[target_mask.bool()] = flat_targets
    return targets


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


class Feedback2Dataset(Dataset):
    def __init__(
        self,
        texts,
        df,
        tokenizer,
        max_len,
        truncation,
        return_overflowing_tokens,
        stride,
        pad_to_multiple_of,
        normalize_text,
        label_df=None,
        siamese=False,
    ):
        self.texts = texts.set_index("id")
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.truncation = truncation
        self.return_overflowing_tokens = return_overflowing_tokens
        self.stride = stride
        self.pad_to_multiple_of = pad_to_multiple_of
        self.siamese = siamese

        self.label_df = None if label_df is None else label_df.set_index("discourse_id")
        if normalize_text:
            self.texts = normalize(self.texts, "text")
            self.df = normalize(self.df, "discourse_text")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        discourse_id, text_id, discourse_text, discourse_type, label = self.df.loc[
            idx,
            [
                "discourse_id",
                "essay_id",
                "discourse_text",
                "discourse_type",
                "discourse_effectiveness",
            ],
        ]

        text = self.texts.loc[text_id, "text"]
        if self.label_df is not None:
            label = self.label_df.loc[discourse_id, LABELS].to_numpy()
        return text, discourse_text, discourse_type, label

    def _get_non_siamese_inputs(self, discourses, texts):
        try:
            inputs = self.tokenizer(
                discourses,
                texts,
                add_special_tokens=True,
                padding=True,
                truncation=self.truncation,
                return_overflowing_tokens=self.return_overflowing_tokens,
                return_offsets_mapping=False,
                max_length=self.max_len,
                stride=self.stride,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        except:
            inputs = self.tokenizer(
                discourses,
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
        return inputs

    def _get_siamese_inputs(self, discourses, texts):
        if self.return_overflowing_tokens:
            raise RuntimeError("siamese with overflowing tokens not allowed")
        combined = [text for pair in zip(discourses, texts) for text in pair]
        inputs = self.tokenizer(
            combined,
            add_special_tokens=True,
            padding=True,
            truncation=self.truncation,
            return_overflowing_tokens=self.return_overflowing_tokens,
            return_offsets_mapping=False,
            max_length=self.max_len,
            stride=self.stride,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        x = inputs.input_ids
        # HACK: we *do not* repeat_interleave, because this is only used
        # after the model is applied, in which case z.dim(0) = x.dim(0) // 2
        inputs["overflow_to_sample_mapping"] = torch.arange(
            len(discourses), dtype=torch.long, device=x.device
        )
        return inputs

    def get_collate_fn(self):
        def collate_fn(examples):
            texts, discourse_texts, discourse_types, labels = [
                list(a) for a in zip(*examples)
            ]
            discourses = [
                "{}\n{}".format(d_type, d_text)
                for d_type, d_text in zip(discourse_types, discourse_texts)
            ]

            if self.siamese:
                inputs = self._get_siamese_inputs(discourses, texts)
            else:
                inputs = self._get_non_siamese_inputs(discourses, texts)

            soft = self.label_df is not None
            targets = get_targets(labels, inputs.overflow_to_sample_mapping, soft)
            if soft:
                labels = get_hard_labels(labels)
            return len(examples), inputs, targets, labels

        return collate_fn


def exact_find(text, sub):
    sub = sub.strip()
    start = text.find(sub)
    if start == -1:
        return -1, -1
    end = start + len(sub)
    return start, end


def find_split(sections, d_text, d_id, find):
    found = False
    next_sections = []
    for section, key in sections:
        if found or key is not None:
            next_sections.append((section, key))
            continue
        start, end = find(section, d_text)
        if start == -1:
            next_sections.append((section, key))
            continue

        found = True
        if start > 0:
            next_sections.append((section[:start], key))
        next_sections.append((section[start:end], d_id))
        if end < len(section):
            next_sections.append((section[end:], key))

    return next_sections, found


def make_merged_text(text, d_ids, d_texts, d_types, d_labels):
    sections = [(text, None)]
    d_items = list(zip(d_ids, d_texts, d_types, d_labels))
    d_map = {x[0]: x for x in d_items}
    d_items.sort(key=lambda x: len(x[1]), reverse=True)

    err = False
    for d_id, d_text, _, _ in d_items:
        sections, found = find_split(sections, d_text, d_id, exact_find)
        if not found:
            sections.append((d_text, d_id))
            err = True

    merged = []
    offsets = []
    d_ids = []
    d_labels = []
    for section, key in sections:
        if key is None:
            merged += section
            continue

        d_id, _, d_type, d_label = d_map[key]

        merged += START_LEFT
        d_code = DISCOURSE_TYPE_CODE[DISCOURSE_TYPE_TO_ID[d_type]]
        offsets.append((len(merged), len(merged) + len(d_code)))
        merged += d_code + START_RIGHT
        merged += section
        merged += END_LEFT + d_code + END_RIGHT

        d_ids.append(d_id)
        d_labels.append(d_label)

    offsets, d_ids, d_labels = zip(*sorted(zip(offsets, d_ids, d_labels)))
    return "".join(merged), offsets, d_ids, d_labels


def make_merged_df(texts, df, label_df=None):
    idx_map = {k: v for v, k in enumerate(df["discourse_id"])}

    df = df.groupby(["essay_id"]).agg(list)
    label_df = None if label_df is None else label_df.set_index("discourse_id")
    records = []
    for i in range(len(texts)):
        essay_id, text = texts.loc[i, ["id", "text"]]
        d_ids, d_texts, d_types, d_labels = df.loc[
            essay_id,
            [
                "discourse_id",
                "discourse_text",
                "discourse_type",
                "discourse_effectiveness",
            ],
        ]
        if label_df is not None:
            d_labels = label_df.loc[d_ids, LABELS].to_numpy().tolist()
        merged, offsets, ids, labels = make_merged_text(
            text, d_ids, d_texts, d_types, d_labels
        )
        idxs = tuple(idx_map[idx] for idx in ids)
        records.append(
            {
                "essay_id": essay_id,
                "text": merged,
                "offsets": offsets,
                "discourse_ids": ids,
                "idxs": idxs,
                "labels": labels,
            }
        )
    merged_df = pd.DataFrame.from_records(records)
    merged_df.sort_values(by="text", key=lambda col: col.str.len(), inplace=True)
    return merged_df


class Feedback2MultiDataset(Dataset):
    def __init__(
        self,
        texts,
        df,
        tokenizer,
        max_len,
        pad_to_multiple_of,
        normalize_text,
        label_df=None,
    ):
        if normalize_text:
            texts = normalize(texts, "text")
            df = normalize(df, "discourse_text")

        self.label_df = label_df
        self.df = make_merged_df(texts, df, label_df)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_to_multiple_of = pad_to_multiple_of

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text, offsets, discourse_ids, idxs, labels = self.df.loc[
            idx,
            [
                "text",
                "offsets",
                "discourse_ids",
                "idxs",
                "labels",
            ],
        ]

        return text, offsets, discourse_ids, idxs, labels

    def get_collate_fn(self):
        def collate_fn(examples):
            texts, offsets, discourse_ids, idxs, labels = [
                list(a) for a in zip(*examples)
            ]

            inputs = self.tokenizer(
                texts,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                max_length=self.max_len,
                stride=0,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            labels = np.concatenate(labels).tolist()

            soft = self.label_df is not None
            target_mask = get_target_mask(inputs, offsets)
            targets = get_targets_combined(labels, target_mask, soft)

            if soft:
                labels = get_hard_labels(labels)

            inputs["target_mask"] = target_mask
            inputs["idxs"] = inputs.overflow_to_sample_mapping.new_tensor(
                [a for idx in idxs for a in idx]
            )
            return target_mask.sum().item(), inputs, targets, labels

        return collate_fn


def score_raw(preds_batch, labels_batch):
    scores = []
    preds = np.array(preds_batch)
    preds = preds / np.sum(preds, axis=-1, keepdims=True)
    preds = np.clip(preds, 1e-15, 1 - 1e-15)
    labels = np.array([[LABEL_TO_ID[label]] for label in labels_batch])
    scores = np.take_along_axis(preds, labels, axis=-1)
    scores = -np.log(scores)
    return scores


def score(preds_batch, labels_batch):
    return np.mean(score_raw(preds_batch, labels_batch))
