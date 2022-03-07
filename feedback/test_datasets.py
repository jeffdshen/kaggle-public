import unittest

import torch
import pandas as pd

from .datasets import (
    check_answer_dict,
    get_answer_dict,
    get_clean_answers,
    get_matches,
    get_target,
    get_word_dict,
    intersect_ranges,
    pred_to_words,
    score,
    split_offsets,
)


class SplitOffsetsTestCase(unittest.TestCase):
    def test_empty(self):
        offsets = split_offsets("")
        expected = []
        self.assertEqual(offsets, expected)

    def test_short(self):
        offsets = split_offsets("Hello world!")
        expected = [(0, 5), (6, 12)]
        self.assertEqual(offsets, expected)

    def test_long(self):
        offsets = split_offsets("Hello world,\nthis is    mooooon.\n\n Hello moon!!!")
        expected = [(0, 5), (6, 12), (13, 17), (18, 20), (24, 32), (35, 40), (41, 48)]
        self.assertEqual(offsets, expected)


class DictTestCase(unittest.TestCase):
    def test_get_answers_dict(self):
        df = pd.DataFrame(
            [
                ["A", "Lead", "0 1 2"],
                ["B", "Claim", "3 4 5 6"],
                ["B", "Lead", "1 2 3"],
            ],
            columns=["id", "discourse_type", "predictionstring"],
        )
        answers = get_answer_dict(df)
        expected = {
            "A": [([0, 1, 2], "Lead")],
            "B": [([1, 2, 3], "Lead"), ([3, 4, 5, 6], "Claim")],
        }
        self.assertEqual(answers, expected)

    def test_get_clean_answers(self):
        answers = {
            "A": [([0, 1, 2], "Lead")],
            "B": [([1, 2, 3], "Lead"), ([3, 4, 5, 6], "Claim")],
        }
        answers = get_clean_answers(answers)
        expected = {
            "A": [([0, 1, 2], "Lead")],
            "B": [([1, 2], "Lead"), ([3, 4, 5, 6], "Claim")],
        }
        self.assertEqual(answers, expected)

    def test_check_answer_dict(self):
        answers = {
            "A": [([0, 1, 2], "Lead")],
            "B": [([1, 2], "Lead"), ([3, 4, 5, 6], "Claim")],
        }
        self.assertTrue(check_answer_dict(answers))
        answers = {
            "A": [([0, 1, 2], "Lead")],
            "B": [([1, 2, 3], "Lead"), ([3, 4, 5, 6], "Claim")],
        }
        self.assertFalse(check_answer_dict(answers))

    def test_get_word_dict(self):
        texts = pd.DataFrame(
            [
                ["A", "Hello world!"],
                ["B", "hello mooonnnn"],
            ],
            columns=["id", "text"],
        )
        words = get_word_dict(texts)
        expected = {
            "A": [(0, 5), (6, 12)],
            "B": [(0, 5), (6, 14)],
        }
        self.assertEqual(words, expected)


class RangesTestCase(unittest.TestCase):
    def test_ranges(self):
        ranges = [(0, 2), (2, 4), (4, 15)]
        items = [(0, 0), (0, 1), (2, 3), (3, 5), (6, 10), (12, 13)]
        groups = intersect_ranges(ranges, items)
        expected = [
            [1],
            [2, 3],
            [4, 5],
        ]
        self.assertEqual(groups, expected)


class TargetTestCase(unittest.TestCase):
    def test_get_target(self):
        offset_mapping = torch.tensor(
            [
                [
                    [0, 0],
                    [0, 1],
                    [2, 4],
                    [5, 10],
                    [11, 15],
                    [16, 17],
                    [17, 18],
                    [18, 19],
                    [20, 21],
                    [22, 23],
                    [24, 25],
                    [25, 26],
                    [27, 32],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [33, 36],
                    [37, 41],
                    [42, 43],
                    [43, 44],
                    [45, 45],
                    [46, 47],
                    [48, 49],
                    [49, 50],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0, 5],
                    [5, 6],
                    [7, 12],
                    [13, 19],
                    [20, 22],
                    [22, 23],
                    [24, 25],
                    [25, 26],
                    [27, 32],
                    [33, 36],
                    [36, 37],
                    [0, 0],
                    [0, 0],
                ],
            ],
            dtype=torch.long,
        )
        answers = [
            [([0, 1, 2, 3, 4], "Lead"), ([8, 9, 10, 11], "Position")],
            [([1, 2, 3], "Position"), ([5, 6], "Claim")],
        ]
        texts = [
            "I do agree that X.\n A A A. There are some Y.  A A.",
            "Hello. There should be. X. There are.",
        ]
        word_offsets = [split_offsets(text) for text in texts]
        overflow_to_sample_mapping = torch.tensor([0, 0, 1], dtype=torch.long)
        targets = get_target(
            offset_mapping, answers, word_offsets, overflow_to_sample_mapping
        )
        expected = torch.tensor(
            [
                [0, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 3, 0],
                [0, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 3, 4, 4, 4, 0, 0, 7, 8, 8, 0, 0],
            ],
            dtype=torch.long,
        )
        self.assertEqual(targets.tolist(), expected.tolist())


class ScoreTestCase(unittest.TestCase):
    def test_get_matches(self):
        preds = [[0, 1], [2, 3, 4, 5], [7, 8, 9], [10, 11, 12, 13, 14], [16]]
        golds = [[1, 2], [3, 4, 5, 6, 7], [8, 9, 10, 11], [13, 14, 15], [16]]
        matches = get_matches(preds, golds)
        expected = [(1, 1), (4, 4)]
        self.assertEqual(matches, expected)

    def test_score(self):
        answers = [
            [([0, 1, 2, 3, 4], "Lead"), ([8, 9, 10, 11], "Position")],
            [([1, 2, 3], "Position"), ([5, 6], "Claim")],
        ]
        texts = [
            "I do agree that X.\n A A A. There are some Y.  A A.",
            "Hello. There should be. X. There are.",
        ]
        word_offsets = [split_offsets(text) for text in texts]
        preds = [
            [
                ((0, 18), "Lead"),
                ((20, 30), "Position"),
                ((30, 45), "Claim"),
                ((45, 50), "Position"),
            ],
            [
                ((0, 15), "Position"),
                ((15, 20), "Claim"),
                ((20, 37), "Lead"),
            ],
        ]
        scores = score(preds, word_offsets, answers)
        expected = {
            "Lead": (1, 1, 0),
            "Position": (1, 2, 1),
            "Evidence": (0, 0, 0),
            "Claim": (0, 2, 1),
            "Concluding Statement": (0, 0, 0),
            "Counterclaim": (0, 0, 0),
            "Rebuttal": (0, 0, 0),
        }
        self.assertEqual(scores, expected)

    def test_to_pred_words(self):
        preds = [
            [
                ((0, 18), "Lead"),
                ((20, 30), "Position"),
                ((30, 45), "Claim"),
                ((45, 50), "Position"),
            ],
            [
                ((0, 15), "Position"),
                ((15, 20), "Claim"),
                ((20, 37), "Lead"),
            ],
        ]
        texts = [
            "I do agree that X.\n A A A. There are some Y.  A A.",
            "Hello. There should be. X. There are.",
        ]
        word_offsets = [split_offsets(text) for text in texts]
        pred_words = pred_to_words(preds, word_offsets)
        expected = [
            [
                ([0, 1, 2, 3, 4], "Lead"),
                ([5, 6, 7, 8], "Position"),
                ([9, 10, 11], "Claim"),
                ([12, 13], "Position"),
            ],
            [
                ([0, 1, 2], "Position"),
                ([3, 4, 5, 6], "Lead"),
            ],
        ]
        self.assertEqual(pred_words, expected)

if __name__ == "__main__":
    unittest.main()
