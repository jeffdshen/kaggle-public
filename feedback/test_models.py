from types import SimpleNamespace
import unittest

import torch
import pandas as pd

from .models import SoftmaxHead


class SoftmaxHeadTestCase(unittest.TestCase):
    def test_get_pred(self):
        offset_mapping = [
            [[0, 0], [0, 1], [2, 4], [5, 10], [11, 15], [16, 17], [17, 18], [0, 0]],
            [[0, 0], [16, 17], [17, 18], [18, 19], [20, 21], [22, 23], [0, 0], [0, 0]],
            [[0, 0], [0, 5], [5, 6], [7, 12], [13, 19], [20, 22], [0, 0], [0, 0]],
        ]
        offset_mapping = torch.tensor(offset_mapping, dtype=torch.long)
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 0],
            ],
            dtype=torch.long,
        )
        overflow_to_sample_mapping = torch.tensor([0, 0, 1], dtype=torch.long)
        zz = list(torch.split(torch.eye(15), 1))
        zz.append(zz[1] + zz[5] / 2)
        zz.append(zz[5] / 2)
        zz.append(zz[5] * 2)
        zz = torch.stack(zz).tolist()
        z0 = [0, 1, 2, 15, 2, 2, 3, 0]
        z1 = [0, 16, 17, 6, 8, 6, 1, 0]
        z2 = [0, 7, 0, 9, 0, 9, 9, 9]
        z = torch.tensor(
            [[zz[a] for a in z0], [zz[a] for a in z1], [zz[a] for a in z2]]
        )
        x = SimpleNamespace(
            attention_mask=attention_mask,
            offset_mapping=offset_mapping,
            overflow_to_sample_mapping=overflow_to_sample_mapping,
        )
        expected = [
            [[[0, 4], "Lead"], [[5, 17], "Lead"], [[17, 23], "Evidence"]],
            [
                [[0, 5], "Claim"],
                [[7, 12], "Concluding Statement"],
                [[20, 22], "Concluding Statement"],
            ],
        ]
        pred = SoftmaxHead.get_pred(z, x)
        self.assertEqual(pred, expected)

