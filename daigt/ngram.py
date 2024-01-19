import re
from collections import Counter, defaultdict
import json
import heapq
from random import Random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

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
    texts,
    score_fn,
    ngram_fn,
    preprocess_fn,
    postprocess_fn,
    words_re: re.Pattern,
    spell: SpellChecker,
    memoize_candidates,
):
    replace_candidates = {}
    for i, (text, source) in tqdm(texts.items()):
        x = ngram_fn(text)

        source = preprocess_fn(source)
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
    texts,
    score_fn,
    ngram_fn,
    preprocess_fn,
    postprocess_fn,
    replace_candidates,
):
    replacements = {}
    results = {}
    scores = {}
    for i, (text, source) in tqdm(texts.items()):
        x = ngram_fn(text)

        source = preprocess_fn(source)

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


def get_word_freqs(sources, preprocess_fn, words_re):
    word_freqs = Counter()
    for source in tqdm(sources):
        source = preprocess_fn(source)
        word_freqs.update([s.lower() for s in words_re.findall(source)])

    return word_freqs


def get_word_freqs_set(sources, preprocess_fn, words_re):
    word_freqs = Counter()
    for source in tqdm(sources):
        source = preprocess_fn(source)
        word_freqs.update(set(s.lower() for s in words_re.findall(source)))

    return word_freqs


def prefix_match(a, b):
    a = a.lower()
    b = b.lower()
    c = min(len(a), len(b))
    return a[:c] == b[:c]


@dataclass
class CandData:
    distance: int
    freq: float

    def __str__(self):
        return str((self.distance, round(self.freq, 5)))


@dataclass
class WordData:
    word: str
    next_word: Optional[str]
    maybe: bool
    first_char_match: bool
    last_char_match: bool
    last_char: str
    freq: float
    min_freq: float
    max_freq: float
    distance: int
    index: int
    index_frac: float
    char_start: int
    char_start_frac: float
    cands: Dict[str, CandData]

    def __str__(self):
        contents = [
            str(x)
            for x in (
                self.word,
                self.next_word,
                1 if self.maybe else 0,
                (self.first_char_match, self.last_char_match, self.last_char),
                (round(self.freq, 5), round(self.min_freq, 5), round(self.max_freq, 5)),
                self.distance,
                (self.index, round(self.index_frac, 5)),
                (self.char_start, round(self.char_start_frac, 5)),
                ", ".join(f"({k}, {v})" for k, v in self.cands.items()),
            )
        ]
        contents = ", ".join(contents)
        return f"({contents})"


@dataclass
class EssayData:
    words: List[WordData]
    total_chars: int
    total_words: int
    cutoff_range: Tuple[int, int]
    cutoff_char_range: Tuple[int, int]
    cutoff_word_range: Tuple[int, int]
    cutoff_range_frac: Tuple[float, float]
    cutoff_char_range_frac: Tuple[float, float]
    cutoff_word_range_frac: Tuple[float, float]

    def __str__(self):
        summary = [
            str(x)
            for x in (
                self.total_chars,
                self.total_words,
                self.cutoff_range,
                self.cutoff_char_range,
                self.cutoff_word_range,
                self.cutoff_range_frac,
                self.cutoff_char_range_frac,
                self.cutoff_word_range_frac,
            )
        ]
        summary = ", ".join(summary)
        contents = "\n".join(str(x) for x in self.words)
        return f"EssayData: {summary}\n{contents}"


class CandDataGetter:
    def __init__(self, spell, memoize_candidates):
        self.spell = spell
        self.memoize_candidates = memoize_candidates

    def __call__(self, word):
        memoize_candidates = self.memoize_candidates
        spell = self.spell
        if word.lower() not in memoize_candidates:
            memoize_candidates[word.lower()] = spell.candidates(word.lower())
        cands = memoize_candidates[word.lower()]
        if cands is None:
            return None
        spell.distance = 1
        cands_1 = spell.candidates(word.lower()) or set()
        spell.distance = 2
        total_freq = sum(spell[next_word] for next_word in cands)

        cands_data = {
            w: CandData(1, spell[w] / total_freq)
            if w in cands_1
            else CandData(2, spell[w] / total_freq)
            for w in cands
        }
        return cands_data


class FreqGetter:
    def __init__(self, word_freqs, memoize_candidates):
        self.word_freqs = word_freqs
        self.memoize_candidates = memoize_candidates

    def __call__(self, word, include_word=False, prior=0):
        memoize_candidates = self.memoize_candidates
        word_freqs = self.word_freqs
        if word.lower() not in memoize_candidates:
            raise RuntimeError("No candidate: ", word.lower())
        cands = memoize_candidates[word.lower()]
        if cands is None:
            return None
        freqs = {w: word_freqs[w] if w in word_freqs else prior for w in cands}
        if include_word:
            freqs[word] = (
                word_freqs[word.lower()] if word.lower() in word_freqs else prior
            )

        total_freq = sum(freqs.values())
        if total_freq == 0:
            freqs = {k: 1 / len(freqs) for k in freqs}
        else:
            freqs = {k: v / total_freq for k, v in freqs.items()}
        return freqs


def analyze_text(
    source, text, replace, words_re, spell: SpellChecker, memoize_candidates
) -> EssayData:
    words, char_starts = zip(*((m[0], m.start(0)) for m in words_re.finditer(source)))
    total_chars = len(source)
    total_words = len(words)
    unknown = spell.unknown(words)
    seen = set()
    replace_checks = set()
    cander = CandDataGetter(spell, memoize_candidates)

    words_data = []
    for index, word in enumerate(words):
        if word in seen:
            continue
        seen.add(word)

        if word.lower() not in unknown:
            continue
        if word.isupper():
            continue
        cands_data = cander(word)
        if cands_data is None:
            continue
        next_word = replace.get(word)
        maybe = next_word is None and any(w in word for w in replace_checks)
        if replace.get(word) is not None:
            replace_checks.add(word)

        # distance will always be the same because spellchecker searches one at a time.
        distance = min(c.distance for c in cands_data.values())
        min_freq = min(c.freq for c in cands_data.values())
        max_freq = max(c.freq for c in cands_data.values())
        freq = 1.0 if next_word is None else cands_data[next_word].freq
        words_data.append(
            WordData(
                word,
                next_word,
                maybe,
                word[0].lower() == next_word[0] if next_word is not None else None,
                word[-1].lower() == next_word[-1] if next_word is not None else None,
                next_word[-1] if next_word is not None else "",
                freq,
                min_freq,
                max_freq,
                distance,
                index,
                index / total_words,
                char_starts[index],
                char_starts[index] / total_chars,
                cands_data,
            )
        )
    # range(cutoff) partitions words_data into feasible and unfeasible words
    # lower_bound = last confirmed word, upper_bound = first confirmed 1 cand miss
    # cutoff is between [cutoff_a, cutoff_b]
    a = max((i + 1 for i, w in enumerate(words_data) if w.next_word), default=0)
    is_miss = lambda w: not w.next_word and not w.maybe and len(w.cands) == 1
    b = min(
        (i for i, w in enumerate(words_data) if is_miss(w)), default=len(words_data)
    )
    b += 1
    a_word = words_data[a - 1].index + 1 if a > 0 else 0
    b_word = words_data[b - 1].index + 1 if b - 1 < len(words_data) else total_words + 1
    a_char = words_data[a - 1].char_start + len(words_data[a - 1].word) if a > 0 else 0
    b_char = (
        words_data[b - 1].char_start if b - 1 < len(words_data) else total_chars + 1
    )
    return EssayData(
        words_data,
        total_chars,
        total_words,
        (a, b),
        (a_char, b_char),
        (a_word, b_word),
        (a / (len(words_data) or 1), (b - 1) / (len(words_data) or 1)),
        (a_char / total_chars, (b_char - 1) / total_chars),
        (a_word / total_words, (b_word - 1) / total_words),
    )


def analyze_texts(
    match_texts,
    results,
    replacements,
    preprocess_fn,
    words_re,
    spell,
    memoize_candidates,
):
    all_data = {}
    skipped = set()
    for i, (text, source) in tqdm(match_texts.items()):
        if results[i] != text:
            skipped.add(i)
            continue
        source = preprocess_fn(source)
        replace = {word: next_word for word, next_word in replacements[i]}

        data = analyze_text(source, text, replace, words_re, spell, memoize_candidates)
        all_data[i] = data
    return all_data, skipped


def augment_random(
    text,
    preprocess_fn,
    postprocess_fn,
    words_re: re.Pattern,
    spell: SpellChecker,
    freq_getter,
    memoize_candidates,
    rand: Random,
    quota_prob=0.5,
    word_base_base_prob=0.4,
    word_base_scaling=1,
    proper_noun_base_prob=0.65,
    proper_noun_scaling=3,
):
    text = preprocess_fn(text)
    words = words_re.findall(text)

    unknown = spell.unknown(words)
    words = uniquify(words)
    words = [word for word in words if word.lower() in unknown and not word.isupper()]
    for word in words:
        if word.lower() not in memoize_candidates:
            memoize_candidates[word.lower()] = spell.candidates(word.lower())
    words = [word for word in words if memoize_candidates[word.lower()] is not None]

    quota = sum(rand.random() < quota_prob for _ in range(len(words)))

    replace = []
    for index, word in enumerate(words):
        if len(replace) >= quota:
            break
        cands = memoize_candidates[word.lower()]
        if len(cands) == 1:
            replace.append((word, list(cands)[0]))
            continue

        freqs = freq_getter(word, include_word=True)
        prob_check = proper_noun_base_prob if word[0].isupper() else word_base_base_prob
        prob_scaling = proper_noun_scaling if word[0].isupper() else word_base_scaling
        prob_check += prob_scaling * freqs.get(word, 0.0)
        if rand.random() < prob_check:
            continue

        next_words, weights = zip(*[(k, v) for k, v in freqs.items() if k != word])
        next_word = rand.choices(next_words, weights)[0]
        replace.append((word, next_word))

    text = apply_replacements(text, replace, postprocess_fn)
    return text, replace
