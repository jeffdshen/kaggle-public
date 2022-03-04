import copy
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
        offset = begin_offset + len(word)
        offsets.append((begin_offset, offset))
    return offsets


labels = [
    "Lead",
    "Position",
    "Evidence",
    "Claim",
    "Concluding Statement",
    "Counterclaim",
    "Rebuttal",
]


def get_labels_to_id(labels):
    labels_to_id = {"Other": 0}
    for i, t in enumerate(labels):
        labels_to_id[t + "0"] = i * 2 + 1
        labels_to_id[t + "1"] = i * 2 + 2
    return labels_to_id


labels_to_id = get_labels_to_id(labels)


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
    answers = copy.deepcopy(answers)
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
        [(word_offset[words[0]][0], word_offset[words[-1]][1]) for words, _ in answer]
        for answer, word_offset in zip(answers, word_offsets)
    ]
    answer_seen = set()
    target = torch.zeros(token_offsets.size()[:-1], dtype=torch.long)
    for i, token_offset in enumerate(token_offsets.tolist()):
        j = overflow_to_sample[i].item()
        answer_tokens = intersect_ranges(answer_ranges[j], token_offset)
        for k, answer_token in enumerate(answer_tokens):
            label = answers[j][k][1]
            label1 = labels_to_id[label + "1"]
            label0 = labels_to_id[label + "0"] if (j, k) not in answer_seen else label1
            target[i, answer_token[0:1]] = label0
            target[i, answer_token[1:]] = label1

            if len(answer_token) > 0:
                answer_seen.add((j, k))
    return target


class FeedbackDataset(Dataset):
    def __init__(self, texts, df, tokenizer, max_len):
        self.texts = texts
        self.answers = get_answer_dict(df)
        self.answers = get_clean_answers(self.answers)
        self.words = get_word_dict(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.loc[idx, "text"]
        text_id = self.texts.loc[idx, "id"]
        answer = self.answers[text_id]
        words = self.words[text_id]
        return text, answer, words

    def get_collate_fn(self):
        def collate_fn(examples):
            text, answer, words = [list(a) for a in zip(*examples)]
            inputs = self.tokenizer(
                text,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            target = get_target(
                inputs.offset_mapping,
                answer,
                words,
                inputs.overflow_to_sample_mapping,
            )
            return len(examples), inputs, target, words, answer

        return collate_fn

def get_matches(preds, golds):
    pred_sets = [set(pred) for pred in preds]
    gold_sets = [set(gold) for gold in golds]
    
    seen = set()
    matches = []
    for i, pred_set in enumerate(pred_sets):
        for j, gold_set in enumerate(gold_sets):
            if j in seen:
                continue
            intersection = len(pred_set.intersection(gold_set))
            if intersection <= 0.5 * len(gold_set):
                continue
            if intersection <= 0.5 * len(pred_set):
                continue
            seen.add(j)
            matches.append((i, j))
            break
    return matches

def score(preds_batch, words_batch, answers_batch):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for preds, words, answers in zip(preds_batch, words_batch, answers_batch):
        pred_ranges = [pred[0] for pred in preds]
        pred_labels = [pred[1] for pred in preds]
        answer_words = [answer[0] for answer in answers]
        answer_labels = [answer[1] for answer in answers]

        pred_words = intersect_ranges(pred_ranges, words)
        matches = get_matches(pred_words, answer_words)
        for l in pred_labels:
            fp[l] += 1
        for l in answer_labels:
            fn[l] += 1
        for i, j in matches:
            l = pred_labels[i]
            if l != answer_labels[j]:
                continue
            tp[l] += 1
            fp[l] -= 1
            fn[l] -= 1

    return {l: (tp[l], fp[l], fn[l]) for l in labels}
    