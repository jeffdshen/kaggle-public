from types import SimpleNamespace
import unittest

import torch
import pandas as pd
import numpy as np

from .datasets import (
    exact_find,
    get_hard_labels,
    get_target_mask,
    make_merged_df,
    make_merged_text,
    overlap,
    score,
    get_targets,
    find_split,
)


class ScoreTestCase(unittest.TestCase):
    def test_score(self):
        preds = [[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [5, 4, 3]]
        labels = ["Effective", "Ineffective", "Adequate"]
        loss = score(preds, labels)
        expected = 11.953510744964333
        self.assertAlmostEqual(loss, expected)


class GetTargetsTestCase(unittest.TestCase):
    def test_hard_labels(self):
        labels = ["Effective", "Ineffective", "Adequate"]
        overflow_to_sample = torch.tensor([0, 1, 1, 2, 2, 2])
        expected = torch.tensor([2, 0, 0, 1, 1, 1])
        targets = get_targets(labels, overflow_to_sample, soft=False)
        torch.testing.assert_close(targets, expected)

    def test_soft_labels(self):
        labels = torch.tensor([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5], [0.3, 0.4, 0.3]])
        overflow_to_sample = torch.tensor([0, 1, 1, 1, 2, 2])
        expected = torch.tensor(
            [
                [0.1, 0.2, 0.7],
                [0.2, 0.3, 0.5],
                [0.2, 0.3, 0.5],
                [0.2, 0.3, 0.5],
                [0.3, 0.4, 0.3],
                [0.3, 0.4, 0.3],
            ]
        )
        targets = get_targets(labels, overflow_to_sample, soft=True)
        torch.testing.assert_close(targets, expected)


class GetHardLabelsTestCase(unittest.TestCase):
    def test_get_hard_labels(self):
        labels = [[0.1, 0.2, 0.7], [0.2, 0.3, 0.5], [0.3, 0.4, 0.3]]
        expected = ["Effective", "Effective", "Adequate"]
        actual = get_hard_labels(labels)
        self.assertEqual(actual, expected)


class MergeTestCase(unittest.TestCase):
    def test_find_split(self):
        sections = [("Hello ", None), ("world.", "0"), (" The world", None)]
        d_text = "world"
        next_sections, found = find_split(sections, d_text, "1", exact_find)
        expected = [("Hello ", None), ("world.", "0"), (" The ", None), ("world", "1")]
        self.assertTrue(found)
        self.assertEqual(next_sections, expected)

    def test_make_merged_text(self):
        text = "The quick brown fox jumps over the lazy dog"
        d_ids = ["a", "b", "c"]
        d_texts = ["quick brown", "jumps", "lazy dog"]
        d_types = ["Lead", "Claim", "Rebuttal"]
        d_labels = ["Ineffective", "Effective", "Adequate"]
        merged, offsets, ids, labels = make_merged_text(
            text, d_ids, d_texts, d_types, d_labels
        )
        expected = "The <lead>quick brown</lead> fox <claim>jumps</claim> over the <rebuttal>lazy dog</rebuttal>"
        self.assertEqual(merged, expected)
        self.assertEqual(list(ids), d_ids)
        self.assertEqual(list(labels), d_labels)
        self.assertEqual(
            [merged[start:end] for start, end in offsets], ["lead", "claim", "rebuttal"]
        )

    def test_make_merged_df(self):
        texts = pd.DataFrame(
            [("x", "The quick brown fox jumps over the lazy dog")],
            columns=["id", "text"],
        )
        df = pd.DataFrame(
            {
                "essay_id": ["x", "x", "x"],
                "discourse_id": ["a", "b", "c"],
                "discourse_text": ["quick brown", "jumps", "lazy dog"],
                "discourse_type": ["Lead", "Claim", "Rebuttal"],
                "discourse_effectiveness": ["Ineffective", "Effective", "Adequate"],
            }
        )
        merged_df = make_merged_df(texts, df)
        expected = [
            {
                "essay_id": "x",
                "text": "The <lead>quick brown</lead> fox <claim>jumps</claim> over the <rebuttal>lazy dog</rebuttal>",
                "offsets": ((5, 9), (34, 39), (64, 72)),
                "discourse_ids": ("a", "b", "c"),
                "labels": ("Ineffective", "Effective", "Adequate"),
            }
        ]
        self.assertEqual(merged_df.to_dict(orient="records"), expected)

    def test_make_merged_df_labeled(self):
        texts = pd.DataFrame(
            [("x", "The quick brown fox jumps over the lazy dog")],
            columns=["id", "text"],
        )
        df = pd.DataFrame(
            {
                "essay_id": ["x", "x", "x"],
                "discourse_id": ["a", "b", "c"],
                "discourse_text": ["quick brown", "jumps", "lazy dog"],
                "discourse_type": ["Lead", "Claim", "Rebuttal"],
                "discourse_effectiveness": ["Ineffective", "Effective", "Adequate"],
            }
        )
        label_df = pd.DataFrame(
            {
                "discourse_id": ["a", "b", "c"],
                "Ineffective": [0.2, 0.4, 0.6],
                "Adequate": [0.5, 0.4, 0.3],
                "Effective": [0.3, 0.2, 0.1],
            }
        )
        merged_df = make_merged_df(texts, df, label_df)
        expected = [
            {
                "essay_id": "x",
                "text": "The <lead>quick brown</lead> fox <claim>jumps</claim> over the <rebuttal>lazy dog</rebuttal>",
                "offsets": ((5, 9), (34, 39), (64, 72)),
                "discourse_ids": ("a", "b", "c"),
                "labels": (
                    [0.2, 0.5, 0.3],
                    [0.4, 0.4, 0.2],
                    [0.6, 0.3, 0.1],
                ),
            }
        ]
        self.assertEqual(merged_df.to_dict(orient="records"), expected)


class TargetMaskTestCase(unittest.TestCase):
    def test_overlap(self):
        self.assertTrue(overlap(0, 5, 0, 5))
        self.assertTrue(overlap(0, 6, 0, 5))
        self.assertTrue(overlap(0, 5, 0, 6))
        self.assertTrue(overlap(4, 6, 0, 5))
        self.assertTrue(overlap(0, 5, 4, 6))
        self.assertTrue(overlap(2, 3, 1, 4))
        self.assertTrue(overlap(1, 4, 2, 3))
        self.assertTrue(overlap(3, 3, 0, 5))
        self.assertTrue(overlap(0, 5, 3, 3))
        self.assertFalse(overlap(0, 5, 5, 6))
        self.assertFalse(overlap(5, 6, 0, 5))
        self.assertFalse(overlap(0, 0, 0, 5))
        self.assertFalse(overlap(0, 5, 0, 0))
        self.assertFalse(overlap(0, 5, 7, 8))
        self.assertFalse(overlap(7, 8, 0, 5))

    def test_target_mask(self):
        inputs = SimpleNamespace(
            offset_mapping=torch.tensor(
                [
                    [
                        [0, 0],
                        [0, 1],
                        [1, 2],
                        [1, 2],
                        [2, 10],
                        [10, 11],
                        [11, 39],
                        [39, 48],
                    ],
                    [
                        [0, 0],
                        [0, 1],
                        [1, 5],
                        [5, 10],
                        [10, 11],
                        [11, 14],
                        [14, 30],
                        [30, 45],
                    ],
                    [
                        [0, 0],
                        [45, 46],
                        [46, 50],
                        [50, 55],
                        [55, 60],
                        [60, 63],
                        [63, 75],
                        [0, 0],
                    ],
                ],
                dtype=torch.long,
            ),
            overflow_to_sample_mapping=torch.tensor([0, 1, 1]),
            dtype=torch.long,
        )
        offsets_batch = [[(2, 11), (39, 48)], [(1, 5), (50, 60)]]
        target_mask = get_target_mask(inputs, offsets_batch)
        expected = torch.tensor(
            [
                [0, 0, 0, 0, 1, 0, 0, 1],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ], dtype=torch.long
        )
        torch.testing.assert_close(target_mask, expected)


if __name__ == "__main__":
    unittest.main()
