import unittest

import torch

from .datasets import get_hard_labels, score, get_targets


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


if __name__ == "__main__":
    unittest.main()
