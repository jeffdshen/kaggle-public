from types import SimpleNamespace
import unittest

import torch
import numpy as np

from feedback3.datasets import MAX_LABELS

from .models import SoftmaxHead


class SoftmaxHeadTestCase(unittest.TestCase):
    def test_get_pred(self):
        z = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        x = SimpleNamespace()
        expected = [3, (22.0 + 5.0 * np.exp(1)) / (8 + np.exp(1))]
        pred = SoftmaxHead.get_pred(z, x)
        np.testing.assert_allclose(pred, expected, 1e-6)

    def test_forward(self):
        with torch.no_grad():
            head = SoftmaxHead(5, 20, output_dim=MAX_LABELS)
            x = torch.rand(4, 3, 5)
            mask = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.long)
            z = head(x, mask)
            self.assertEqual(list(z.size()), [8, 9])
