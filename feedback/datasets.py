import copy
from pathlib import Path
from collections import defaultdict

import pandas as pd
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


def get_label_ids(labels):
    label_to_id = {(None, 0): 0}
    id_to_label = {0: (None, 0)}
    for i, t in enumerate(labels):
        label_to_id[(t, True)] = i * 2 + 1
        label_to_id[(t, False)] = i * 2 + 2
        id_to_label[i * 2 + 1] = (t, True)
        id_to_label[i * 2 + 2] = (t, False)
    return label_to_id, id_to_label


label_to_id, id_to_label = get_label_ids(labels)
max_labels = len(label_to_id)


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
        cleaned_answer = []
        for words, label in answer:
            if prev_words[-1] >= words[0]:
                if len(prev_words) == 1:
                    if len(words) == 1:
                        continue
                    words.pop(0)
                else:
                    prev_words.pop()
            cleaned_answer.append((words, label))
            prev_words = words
        answer.clear()
        answer.extend(cleaned_answer)
    return answers


def check_answer_dict(answers):
    for answer in answers.values():
        prev = -1
        for words, _ in answer:
            if len(words) == 0:
                return False
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
            label1 = label_to_id[(label, False)]
            label0 = label_to_id[(label, True)] if (j, k) not in answer_seen else label1
            target[i, answer_token[0:1]] = label0
            target[i, answer_token[1:]] = label1

            if len(answer_token) > 0:
                answer_seen.add((j, k))
    return target


class FeedbackDataset(Dataset):
    def __init__(self, texts, df, tokenizer, max_len, stride):
        self.texts = texts
        self.answers = get_answer_dict(df)
        self.answers = get_clean_answers(self.answers)
        self.words = get_word_dict(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride

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
                stride=self.stride,
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


def _score_single(tp, fp, fn, pred_words, pred_labels, answer_words, answer_labels):
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


def score(preds_batch, words_batch, answers_batch):
    return score_words(pred_to_words(preds_batch, words_batch), answers_batch)

def score_words(preds_batch, answers_batch):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for preds, answers in zip(preds_batch, answers_batch):
        pred_words, pred_labels = zip(*preds) if preds else ([], [])
        answer_words, answer_labels = zip(*answers)
        _score_single(tp, fp, fn, pred_words, pred_labels, answer_words, answer_labels)

    return {l: (tp[l], fp[l], fn[l]) for l in labels}


def pred_to_words(preds_batch, words_batch):
    pred_words_batch = []
    for preds, words in zip(preds_batch, words_batch):
        if not preds:
            pred_words_batch.append([])
            continue

        pred_ranges, pred_labels = zip(*preds)
        pred_words = intersect_ranges(pred_ranges, words)
        pred_words = [(a, b) for a, b in list(zip(pred_words, pred_labels)) if a]
        pred_words_batch.append(pred_words)

    return pred_words_batch
