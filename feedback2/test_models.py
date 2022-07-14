from types import SimpleNamespace
import unittest

import torch
import numpy as np

from .models import ClassTokenHead


class SoftmaxHeadTestCase(unittest.TestCase):
    def test_get_pred(self):
        overflow_to_sample_mapping = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.long)
        z = torch.tensor(
            [
                [-1.0, 1.0, 0],
                [1.0, 0, 1.0],
                [-3.0, 0, 3.0],
                [2.0, 3.0, -1.0],
                [1.0, -3.0, -2.0],
                [100, 101, 100],
            ]
        )
        x = SimpleNamespace(
            overflow_to_sample_mapping=overflow_to_sample_mapping,
        )
        expected = [
            np.array([0.0, 0.5, 0.5]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]
        expected = np.exp(expected)
        expected = expected / np.sum(expected, axis=1, keepdims=True)
        pred = ClassTokenHead.get_pred(z, x)
        np.testing.assert_almost_equal(pred, expected)
