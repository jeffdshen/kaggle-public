import re
from collections import Counter, defaultdict
import json
import heapq

import numpy as np
import pandas as pd
from tqdm import tqdm

from spellchecker import SpellChecker


def collapse(text: str, chars="\naeoitnhrs"):
    table = str.maketrans("", "", chars)
    return text.translate(table)


def ngrams(text: str, n=3):
    return Counter([text[i : i + n] for i in range(len(text) - n)])


def to_freq(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def normalize(x):
    total = np.sqrt(sum([v**2 for v in x.values()]))
    return {k: v / total for k, v in x.items()}


def similarity(a, b):
    # Assumes a and b are already normalized
    dot = 0.0
    for k, v in a.items():
        if k in b:
            dot += v * b[k]
    return dot


def l2_error(a, b):
    # a, b should be unnormalized
    error = 0
    for k in a.keys() | b.keys():
        error += (a.get(k, 0) - b.get(k, 0)) ** 2
    return error


def l1_error(a, b):
    # a, b should be unnormalized
    error = 0
    for k in a.keys() | b.keys():
        error += abs(a.get(k, 0) - b.get(k, 0))
    return error


def text_similarity(a, b, n=3):
    a = ngrams(a, n)
    b = ngrams(b, n)
    return similarity(normalize(a), normalize(b))


def build_ngrams(df, to_ngrams):
    df_ngrams = []
    for i, row in df.iterrows():
        df_ngrams[i] = normalize(to_ngrams(row["text"]))
    return df_ngrams


def find_matches(
    df, source_df, to_ngrams, threshold=0.95, relative_threshold=0.05, k=3
):
    source_ngrams = []
    for i, row in source_df.iterrows():
        source_ngrams.append((i, normalize(to_ngrams(row["text"]))))

    matches = {}
    errors = {}
    for i, row in tqdm(df.iterrows()):
        x = normalize(to_ngrams(row["text"]))
        heap = []
        for j, y in source_ngrams:
            s = similarity(x, y)
            heapq.heappush(heap, (s, j, y))
            if len(heap) > k:
                heapq.heappop(heap)

        heap.sort(reverse=True)
        if heap[0][0] >= threshold:
            i_matches = [j for s, j, y in heap if s >= threshold]
        else:
            i_matches = [j for s, j, y in heap if s >= heap[0][0] - relative_threshold]

        if len(i_matches) == 1:
            matches[i] = i_matches[0]
        else:
            errors[i] = i_matches

    return matches, errors


def filter_characters(
    s,
    allowed="\n !\"%&',.0123456789:?ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz",
):
    s = s.strip()
    s = "".join([c for c in s if c in allowed])
    return s


def replace_common(
    s,
    replacements=[("''", '"'), (" .", "."), ("/", "")],
):
    for a, b in replacements:
        s = s.replace(a, b)
    return s


def drop_lines(s: str, keep=".!?\",-'", keep_len=50):
    lines = s.splitlines()
    lines = [
        line for line in lines if any(c in line for c in keep) or len(line) >= keep_len
    ]
    return "\n".join(lines)


def concat_lines(s: str, stop=".!?"):
    lines = s.splitlines()
    # assumes happens after character filtering
    lines = [" ".join(line.split()) for line in lines]
    lines = [
        line if any(line.endswith(c) for c in stop) else line + " " for line in lines
    ]
    t = "\n\n".join(lines)
    t = t.replace(" \n\n", " ")
    return t


def strip_ending(s: str, ending="."):
    last = max(s.rfind(c) for c in ending)
    return s[: last + 1]


def strip_colons(s: str, colon_re=re.compile("^\w+:", flags=re.MULTILINE)):
    return colon_re.sub("", s)


def strip_beginning_colon(s: str, colon_re=re.compile("^.*:\n", flags=re.MULTILINE)):
    return colon_re.sub("", s)


def drop_bad_lines(s: str, bad_re=re.compile("^[\w ]*:$", flags=re.MULTILINE)):
    s = bad_re.sub("", s)
    return s


def drop_small_lines(s: str, bad_re=re.compile("^[\w: ]*$"), keep_len=58):
    lines = s.splitlines()
    lines = [
        line for line in lines if bad_re.match(line) is None or len(line) >= keep_len
    ]
    return "\n".join(lines)


def standardize_lines(s: str):
    lines = s.splitlines()
    lines = [line for line in lines if len(line) > 0]
    return "\n".join(lines)


def postprocess(s):
    # s = replace_common(s)
    # s = drop_bad_lines(s)
    # s = drop_small_lines(s)
    # s = strip_colons(s)
    # s = strip_beginning_colon(s)
    s = standardize_lines(s)
    s = filter_characters(s)
    s = concat_lines(s)
    s = strip_ending(s)
    return s


def preprocess(s):
    s = replace_common(s)
    s = drop_bad_lines(s)
    s = drop_small_lines(s)
    s = strip_colons(s)
    s = strip_beginning_colon(s)
    return s


def find_substitution(s, t, n=1, threshold=0.004, max_error=0.002):
    s_freq = to_freq(ngrams(s, n))
    t_freq = to_freq(ngrams(t, n))

    diffs = []
    for c, p in s_freq.items():
        q = t_freq.get(c, 0.0)
        diff = q - p
        diffs.append((diff, c))
    diffs.sort()
    a_diff, a = diffs[0]
    b_diff, b = diffs[-1]
    if a_diff > -threshold or b_diff < threshold:
        return None, None
    if abs(a_diff + b_diff) > max_error:
        return None, None
    if t_freq.get(a, 0.0) > 1.5 / len(t):
        return None, None
    return a, b


def find_post_added_ngram(s, t, n=3, k=3, threshold=0.005, max_error=0.0006):
    s_counts = [ngrams(s, n - 1 + i) for i in range(k)]
    t_counts = [ngrams(t, n + i) for i in range(k)]

    diffs = []
    for token in t_counts[0]:
        good = True
        for i in range(k):
            p = t_counts[i].get(token + token[-1] * i, 0)
            q = s_counts[i].get(token[:-1] + token[-1] * i, 0)
            diff = abs(p - q) / len(t)
            if diff > max_error:
                good = False
                break
        if not good:
            continue
        diffs.append((t_counts[0].get(token, 0) / len(t), token))

    if len(diffs) == 0:
        return None, None

    max_p, token = max(diffs)
    if max_p < threshold:
        return None, None

    return token[:-1], token


def find_pre_added_ngram(s, t, n=3, k=3, threshold=0.005, max_error=0.0006):
    s_counts = [ngrams(s, n - 1 + i) for i in range(k)]
    t_counts = [ngrams(t, n + i) for i in range(k)]

    diffs = []
    for token in t_counts[0]:
        good = True
        for i in range(k):
            p = t_counts[i].get(token[0] * i + token, 0)
            q = s_counts[i].get(token[0] * i + token[1:], 0)
            diff = abs(p - q) / len(t)
            if diff > max_error:
                good = False
                break
        if not good:
            continue
        diffs.append((t_counts[0].get(token, 0) / len(t), token))
    if len(diffs) == 0:
        return None, None

    max_p, token = max(diffs)
    if max_p < threshold:
        return None, None

    return token[1:], token


def remove_short_lines(s, max_len=10):
    s = "\n".join(
        [line for line in s.split("\n") if len(line) >= max_len or len(line) == 0]
    )
    s = s.replace("\n\n\n", "\n\n")
    s = s.strip()
    return s


DETECTOR_MAP = {
    "sub1": lambda s, t: find_substitution(s, t),
    "sub2": lambda s, t: find_substitution(s, t, n=2),
    "pre2": lambda s, t: find_pre_added_ngram(s, t, n=2),
    "add2": lambda s, t: find_post_added_ngram(s, t, n=2),
    "add3": lambda s, t: find_post_added_ngram(s, t),
}


def detect_replacements(s, t, named_detectors):
    replacements = []
    for name, detector in named_detectors:
        a, b = detector(s, t)
        if a is not None:
            s = s.replace(a, b)
            replacements.append((name, a, b))
    return s, replacements


def uniquify(words):
    # preserves order
    seen = set()
    words = [w for w in words if w not in seen and not seen.add(w)]
    return words


class WordFinder:
    def __init__(
        self, chars="'/0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz"
    ):
        self.chars = set(chars)

    def findall(self, text):
        words = []
        buffer = []
        for c in text:
            if c in self.chars:
                buffer.append(c)
            else:
                words.append("".join(buffer))
                buffer.clear()
        if len(buffer) > 0:
            words.append("".join(buffer))

        return words


def find_spell_check_candidates(
    source_df: pd.DataFrame,
    df: pd.DataFrame,
    df_to_source_index: dict[int, int],
    score_fn,
    ngram_fn,
    preprocess_fn,
    postprocess_fn,
    words_re: re.Pattern,
    spell: SpellChecker,
    memoize_candidates,
):
    replace_candidates = {}
    for i, row in tqdm(df.iterrows()):
        text = row["text"]
        x = ngram_fn(text)
        j = df_to_source_index[i]

        source = preprocess_fn(source_df.loc[j, "text"])
        words = words_re.findall(source)
        words = uniquify(words)

        s = score_fn(x, ngram_fn(postprocess_fn(source)))

        unknown = spell.unknown(words)
        replace = []
        for word in words:
            if word.lower() not in unknown:
                continue
            if word.lower() not in memoize_candidates:
                memoize_candidates[word.lower()] = spell.candidates(word.lower())
            candidates = memoize_candidates[word.lower()]

            best_candidate = []
            for next_word in candidates or []:
                next_text = source.replace(word, next_word)
                t = score_fn(x, ngram_fn(postprocess_fn(next_text)))
                if t > s:
                    best_candidate.append((t, word, next_word))
            if len(best_candidate) > 0:
                best_candidate = max(best_candidate)
                replace.append(tuple(best_candidate[1:]))

        replace_candidates[i] = replace
    return replace_candidates


def apply_replacements(source, replace, postprocess_fn):
    y = source
    for word, next_word in replace:
        y = y.replace(word, next_word)
    y = postprocess_fn(y)
    return y


def prune_spell_check_candidates(
    source_df: pd.DataFrame,
    df: pd.DataFrame,
    df_to_source_index: dict[int, int],
    score_fn,
    ngram_fn,
    preprocess_fn,
    postprocess_fn,
    replace_candidates,
):
    replacements = {}
    results = {}
    scores = {}
    for i, row in tqdm(df.iterrows()):
        text = row["text"]
        x = ngram_fn(text)
        j = df_to_source_index[i]

        source = preprocess_fn(source_df.loc[j, "text"])

        replace = replace_candidates[i]
        y = apply_replacements(source, replace, postprocess_fn)
        score = score_fn(x, ngram_fn(y))

        if text != y:
            removes = []
            for k in range(len(replace)):
                next_replace = replace[:k] + replace[k + 1 :]
                z = apply_replacements(source, next_replace, postprocess_fn)
                next_score = score_fn(x, ngram_fn(z))
                removes.append(next_score > score)
            if any(removes):
                replace = [rep for remove, rep in zip(removes, replace) if not remove]
                y = apply_replacements(source, replace, postprocess_fn)
                score = score_fn(x, ngram_fn(y))
        
        # remove no-ops
        y = source
        removes = []
        for word, next_word in replace:
            next_y = y.replace(word, next_word)
            removes.append(next_y == y)
            y = next_y
        if any(removes):
            replace = [rep for remove, rep in zip(removes, replace) if not remove]
        y = postprocess_fn(y)

        replacements[i] = replace
        results[i] = y
        scores[i] = score

    return replacements, results, scores


def save_memoize_candidates(x, file):
    y = {k: list(v) if v is not None else None for k, v in x.items()}
    with open(file, "w") as fp:
        json.dump(y, fp)


def load_memoize_candidates(file):
    with open(file) as fp:
        y = json.load(fp)
    x = {k: set(v) if v is not None else None for k, v in y.items()}
    return x
