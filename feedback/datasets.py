from copy import deepcopy
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import Dataset

import datasets as hf_datasets


def get_texts_df(dir_path):
    paths = [x for x in dir_path.iterdir() if x.is_file() and x.suffix == ".txt"]
    texts = []
    for path in paths:
        with open(path) as f:
            texts.append((path.stem, f.read().rstrip()))
    return pd.DataFrame(texts, columns=["id", "text"])


def get_dfs(path):
    path = Path(path)
    test_texts = get_texts_df(path / "test")
    train_df = pd.read_csv(path / "train.csv")
    train_texts = get_texts_df(path / "train")
    return train_texts, train_df, test_texts


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


def split_offsets(line):
    words = line.split()
    offsets = []
    offset = 0
    for word in words:
        begin_offset = line.index(word, offset)
        offset = offset + len(word)
        offsets.append((word, begin_offset, offset))
    return offsets


discourse_types = [
    "Lead",
    "Position",
    "Evidence",
    "Claim",
    "Concluding Statement",
    "Counterclaim",
    "Rebuttal",
]


def get_labels(types):
    labels = {"Other": 0}
    for i, t in enumerate(types):
        labels[t + "0"] = i * 2 + 1
    for i, t in enumerate(types):
        labels[t + "1"] = i * 2 + 2
    return labels


labels = get_labels(discourse_types)


def get_answer_dict(df):
    answers = defaultdict(list)
    for row in df.itertuples():
        words = row.predictionstring.split()
        words = [int(word) for word in words]
        answers[row.id].append((words, row.discourse_type))

    for answer in answers.values():
        answer.sort()
    return answers


def get_clean_answers(answers):
    answers = deepcopy.copy(answers)
    for _, answer in answers.items():
        prev_words = [-1]
        for words, _ in answer:
            if prev_words[-1] >= words[0]:
                prev_words.pop()
            prev_words = words
    return answers


def check_answer_dict(answers):
    for answer in answers.values():
        prev = -1
        for words, _ in answer:
            for word_id in words:
                if prev >= word_id:
                    return False
                prev = word_id
    return True


def get_word_dict(texts):
    offsets = {}
    for row in texts.itertuples():
        offsets[row.id] = split_offsets(row.text)
    return offsets


def overlap(a, b, c, d):
    return a < d and c < b


def intersect_ranges(ranges, items):
    # Given sorted ranges (non-overlapping) and sorted items (non-overlapping)
    # Collect items that fall into these ranges and return the indices.
    groups = []
    index = 0
    for r, s in ranges:
        group = []
        while index < len(items) and items[index][0] < s:
            if r < items[index][1]:
                group.append(index)
            index += 1
        groups.append(group)
    return groups


def get_target(token_offsets, answers, word_offsets, overflow_to_sample):
    answer_ranges = [
        [(word_offset[words[0]][1], word_offset[words[-1]][2]) for words, _ in answer]
        for answer, word_offset in zip(answers, word_offsets)
    ]
    target = torch.zeros(token_offsets.size()[:-1], dtype=torch.long)
    for i, token_offset in enumerate(token_offsets.tolist()):
        j = overflow_to_sample[i].item()
        answer_tokens = intersect_ranges(answer_ranges[j], token_offset)
        for k, answer_token in enumerate(answer_tokens):
            label = answers[j][k][1]
            label0 = labels[label + "0"]
            label1 = labels[label + "1"]
            target[i, answer_token[0:1]] = label0
            target[i, answer_token[1:]] = label1
    return target

