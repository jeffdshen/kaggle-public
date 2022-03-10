from collections import defaultdict
import copy

import numpy as np


def make_ranges(preds, weights):
    ranges = []
    for pred, weight in zip(preds, weights):
        for words, label in pred:
            ranges.append(((words[0], words[-1] + 1), label, weight))
    ranges.sort()
    return ranges


def find_windows(ranges, threshold):
    # Find all windows (and the largest one) satisfy the threshold
    window = []
    windows = []
    best_windows = []
    for i, ((start, end), label, weight) in enumerate(ranges):
        while window and window[-1][0] <= start:
            window.pop()
        window.append((end, label, weight))

        if i < len(ranges) - 1 and ranges[i + 1][0][0] == start:
            continue

        window.sort(reverse=True)
        sum_weight = 0.0
        for e, _, weight in window:
            sum_weight += weight
            if sum_weight > threshold:
                best_windows.append((start, e))
                windows.append(copy.deepcopy(window))
                break
    return windows, best_windows


def clamp(x, y, a, b):
    return max(x, a), min(y, b)


def find_largest_ranges(windows, best_windows, a, b, alpha, beta, answers):
    if a == b:
        return
    best_i = -1
    best_length = 0
    for i in range(a, b):
        s, e = best_windows[i]
        s, e = clamp(s, e, alpha, beta)
        if e - s > best_length:
            best_i = i
            best_length = e - s

    if best_length == 0:
        return
    best_s, best_e = best_windows[best_i]
    best_s, best_e = clamp(best_s, best_e, alpha, beta)

    right_a = best_i + 1
    left_b = best_i
    while left_b > a:
        s, e = best_windows[left_b - 1]
        if s < best_s:
            break
    while right_a < b:
        s, e = best_windows[right_a]
        if e > best_e:
            break

    window = windows[best_i]
    scores = defaultdict(float)
    for e, label, weight in window:
        if e < best_e:
            break
        scores[label] += weight
    max_label = max(scores, key=scores.get)
    answers.append((best_s, best_e, max_label))
    find_largest_ranges(windows, best_windows, a, left_b, alpha, best_s, answers)
    find_largest_ranges(windows, best_windows, right_a, b, best_e, beta, answers)


def ensemble_single(preds, weights, threshold):
    ranges = make_ranges(preds, weights)
    windows, best_windows = find_windows(ranges, threshold)
    assert len(windows) == len(best_windows)
    answers = []
    find_largest_ranges(
        windows, best_windows, 0, len(windows), 0, float("inf"), answers
    )
    answers.sort()
    return [(list(range(s, e)), label) for s, e, label in answers]


def ensemble_list(preds_batch, weights, segment_threshold):
    final_preds = []
    for preds in preds_batch:
        final_pred = ensemble_single(preds, weights, segment_threshold)
        final_preds.append(final_pred)
    return final_preds


def ensemble(preds, weights, segment_threshold):
    pred_weights = [(v, weights[k]) for k, v in preds.items()]
    preds, weights = zip(*pred_weights)
    preds = zip(*preds)
    weights = np.array(weights) / np.sum(weights)
    return ensemble_list(preds, weights, segment_threshold)


## Old code for ensembling where each model votes for adjacent connections
## rather than voting for ranges as a whole

# def ensemble_calc_scores(preds, weights):
#     scores = defaultdict(lambda: defaultdict(float))
#     total_scores = defaultdict(float)
#     max_word = 0
#     for pred, weight in zip(preds, weights):
#         for words, label in pred:
#             words.sort()
#             max_word = max(max_word, words[-1] * 2)
#             for i in range(words[0] * 2, words[-1] * 2 + 1):
#                 scores[i][label] += weight
#                 total_scores[i] += weight
#     return scores, total_scores, max_word


# def ensemble_calc_pred(scores, total_scores, max_word, segment_threshold):
#     final_pred = []
#     cur_words = []
#     cur_labels = defaultdict(float)
#     for i in range(0, max_word + 2):
#         if total_scores[i] > segment_threshold:
#             cur_words.append(i)
#             for k, v in scores[i].items():
#                 cur_labels[k] += v
#             continue

#         if not cur_words:
#             continue

#         start, end = cur_words[0] // 2, (cur_words[-1] // 2 + 1)
#         max_label = max(cur_labels, key=cur_labels.get)
#         final_pred.append((list(range(start, end)), max_label))
#         cur_words.clear()
#         cur_labels.clear()
#     return final_pred


# def ensemble(preds, weights, segment_threshold):
#     pred_weights = [(v, weights[k]) for k, v in preds.items()]
#     preds, weights = zip(*pred_weights)
#     preds = zip(*preds)
#     final_preds = []
#     weights = np.array(weights) / np.sum(weights)
#     for pred in preds:
#         scores, total_scores, max_word = ensemble_calc_scores(pred, weights)
#         final_pred = ensemble_calc_pred(scores, total_scores, max_word, segment_threshold)
#         final_preds.append(final_pred)
#     return final_preds
