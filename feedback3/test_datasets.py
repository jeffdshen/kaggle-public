from types import SimpleNamespace
import unittest

import torch
import pandas as pd
import numpy as np

import transformers

from .datasets import (
    LABEL_TYPES,
    extract_offsets,
    get_flat_targets,
    get_hard_labels,
    get_targets,
    get_target_mask,
    overlap,
    score,
    Feedback3Dataset,
)


class ScoreTestCase(unittest.TestCase):
    def test_score(self):
        preds = [[1.0, 1.5, 2.0, 2.5, 3.0, 3.5], [4.0, 4.5, 5.0, 4.5, 4.0, 3.5]]
        labels = [[1.5, 2.0, 2.5, 3.0, 3.5, 4.0], [5.0, 4.5, 4.0, 3.5, 4.0, 4.5]]
        mse = score(preds, labels)
        expected = np.array([1.25, 0.25, 1.25, 1.25, 0.25, 1.25]) / 2
        np.testing.assert_almost_equal(mse, expected)


class PromptTestCase(unittest.TestCase):
    def test_get_offsets(self):
        prompt = "Hello <<abcd>> World <<efg>>\n\n<<hi>> jkl"
        left = "<<"
        right = ">>"
        prompt, offsets = extract_offsets(prompt, left, right)
        self.assertEqual(prompt, "Hello abcd World efg\n\nhi jkl")
        self.assertEqual(offsets, [(6, 10), (17, 20), (22, 24)])


class GetTargetsTestCase(unittest.TestCase):
    def test_logit_flat(self):
        labels = [5.0, 1.0, 1.5]
        overflow_to_sample = torch.tensor([0, 1, 1, 2, 2, 2])
        expected = torch.tensor([8, 0, 0, 1, 1, 1])
        targets = get_flat_targets(labels, overflow_to_sample, target_type="logit")
        torch.testing.assert_close(targets, expected)

    def test_logit(self):
        labels = [5.0, 1.0, 1.5]
        target_mask = torch.tensor([[0, 1, 1], [1, 0, 0]])
        expected = torch.tensor([[0, 8, 0], [1, 0, 0]])
        targets = get_targets(labels, target_mask, target_type="logit")
        torch.testing.assert_close(targets, expected)

    def test_soft_flat(self):
        labels = torch.tensor(
            [
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.1, 0.3, 0.5],
                [0.1, 0.3, 0.5, 0.01, 0.05, 0.01, 0.01, 0.01, 0.01],
            ]
        )
        overflow_to_sample = torch.tensor([0, 1, 1])
        expected = torch.tensor(
            [
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.1, 0.3, 0.5],
                [0.1, 0.3, 0.5, 0.01, 0.05, 0.01, 0.01, 0.01, 0.01],
                [0.1, 0.3, 0.5, 0.01, 0.05, 0.01, 0.01, 0.01, 0.01],
            ]
        )
        targets = get_flat_targets(labels, overflow_to_sample, target_type="soft")
        torch.testing.assert_close(targets, expected)

    def test_soft(self):
        labels = torch.tensor(
            [
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.1, 0.3, 0.5],
                [0.1, 0.3, 0.5, 0.01, 0.05, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.02, 0.04, 0.1, 0.3, 0.5],
            ]
        )
        target_mask = torch.tensor([[1, 0, 0], [0, 1, 1]])
        expected = torch.tensor(
            [
                [
                    [0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.1, 0.3, 0.5],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.3, 0.5, 0.01, 0.05, 0.01, 0.01, 0.01, 0.01],
                    [0.01, 0.01, 0.01, 0.01, 0.02, 0.04, 0.1, 0.3, 0.5],
                ],
            ]
        )
        targets = get_targets(labels, target_mask, target_type="soft")
        torch.testing.assert_close(targets, expected)

    def test_linear_flat(self):
        labels = [5.0, 1.0, 1.5]
        overflow_to_sample = torch.tensor([0, 1, 1, 2, 2, 2])
        expected = torch.tensor([5.0, 1.0, 1.0, 1.5, 1.5, 1.5])
        targets = get_flat_targets(labels, overflow_to_sample, target_type="linear")
        torch.testing.assert_close(targets, expected)

    def test_linear(self):
        labels = [5.0, 1.0, 1.5]
        target_mask = torch.tensor([[0, 1, 1], [1, 0, 0]])
        expected = torch.tensor([[0, 5.0, 1.0], [1.5, 0, 0]])
        targets = get_targets(labels, target_mask, target_type="linear")
        torch.testing.assert_close(targets, expected)


class GetHardLabelsTestCase(unittest.TestCase):
    def test_get_hard_labels(self):
        labels = [
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.1, 0.3, 0.5],
            [0.1, 0.3, 0.5, 0.01, 0.05, 0.01, 0.01, 0.01, 0.01],
        ]
        expected = [5.0, 2.0]
        actual = get_hard_labels(labels)
        self.assertEqual(actual, expected)


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
            ],
            dtype=torch.long,
        )
        torch.testing.assert_close(target_mask, expected)


class DatasetTestCase(unittest.TestCase):
    def test_getitem(self):
        texts = ["aa\n\n\n\n\n\n", "bb\n", "cc\n"]
        values = [1.0, 1.5, 5.0]

        df = pd.DataFrame({"text": texts, **{label: values for label in LABEL_TYPES}})
        dataset = Feedback3Dataset( df, None, 10, 8, True, "linear", "a<?>", "<", ">")
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0], ("a?aa", [(1, 2)], [1.0] * 6))
        self.assertEqual(dataset[1], ("a?bb", [(1, 2)], [1.5] * 6))
        self.assertEqual(dataset[2], ("a?cc", [(1, 2)], [5.0] * 6))


if __name__ == "__main__":
    unittest.main()
