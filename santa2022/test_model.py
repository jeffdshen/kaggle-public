import unittest

import torch
import numpy as np

from .model import one_hot2d, ArmScalarEmbedding, LocEmbedding, ObservationEmbedding


class OneHot2dTestCase(unittest.TestCase):
    def test_single(self):
        tensor = torch.tensor([1, 3], dtype=torch.long)
        expected = torch.tensor(
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        output = one_hot2d(tensor, (4, 4))
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_batched(self):
        tensor = torch.tensor([[0, 0], [1, 2], [3, 3]], dtype=torch.long)
        expected = torch.tensor(
            [
                [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
            ]
        )
        output = one_hot2d(tensor, (4, 4))
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        tensor = tensor.unsqueeze(0).expand(3, -1, -1)
        expected = expected.unsqueeze(0).expand(3, -1, -1, -1)
        output = one_hot2d(tensor, (4, 4))
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_against_loop(self):
        tensor = torch.tensor([[0, 0], [1, 2], [3, 3]])
        cases = [
            ([1, 3], (4, 4)),
            ([[0, 0], [1, 2], [3, 3]], (4, 4)),
            ([6, 1], (7, 5)),
            ([[[0, 0], [2, 1]], [[4, 1], [4, 3]]], (5, 4)),
        ]
        for data, size in cases:
            tensor = torch.tensor(data)
            output = one_hot2d(tensor, size)
            expected = torch.zeros(*tensor.size()[:-1], *size, dtype=torch.long)
            for index in np.ndindex(*tensor.size()[:-1]):
                x_index = tensor[index][0]
                y_index = tensor[index][1]
                expected[index][x_index][y_index] = 1
            torch.testing.assert_close(output, expected, rtol=0, atol=0)


class ArmEmbeddingTestCase(unittest.TestCase):
    def test_single(self):
        angle_sizes = (torch.tensor([64, 32, 16, 8, 4, 2, 1, 1]) * 8).tolist()
        arm_embed = ArmScalarEmbedding((64, 128), angle_sizes)
        output = arm_embed.forward(torch.tensor([0, 128, 64, 32, 16, 8, 4, 4]))
        expected = torch.zeros(64, 128, 8)
        expected[:, :, 1:] = 0.5
        torch.testing.assert_close(output, expected)

    def test_batch(self):
        angle_sizes = (torch.tensor([64, 32, 16, 8, 4, 2, 1, 1]) * 8).tolist()
        arm_embed = ArmScalarEmbedding((64, 128), angle_sizes)
        inputs = torch.tensor([[0, 128, 64, 32, 16, 8, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4]])
        output = arm_embed.forward(inputs)
        expected = torch.zeros(2, 64, 128, 8)
        expected[0, :, :, 1:] = 0.5
        expected[1, :, :, 0] = 1 / 128
        expected[1, :, :, 1] = 1 / 64
        expected[1, :, :, 2] = 1 / 32
        expected[1, :, :, 3] = 1 / 16
        expected[1, :, :, 4] = 1 / 8
        expected[1, :, :, 5] = 1 / 4
        expected[1, :, :, 6] = 1 / 2
        expected[1, :, :, 7] = 1 / 2
        torch.testing.assert_close(output, expected)


class LocEmbeddingTestCase(unittest.TestCase):
    def test_single(self):
        tensor = torch.tensor([1, 3])
        expected = torch.tensor(
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
        ).unsqueeze(-1)
        loc_embed = LocEmbedding((4, 4))
        output = loc_embed(tensor)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_batched_sizes(self):
        tensor = torch.tensor([[0, 0], [1, 2], [3, 3]])
        loc_embed = LocEmbedding((4, 4))
        output = loc_embed(tensor)
        self.assertEqual(output.size(), (3, 4, 4, 1))


class ObservationEmbeddingTestCase(unittest.TestCase):
    def test_single(self):
        angle_sizes = (torch.tensor([64, 32, 16, 8, 4, 2, 1, 1]) * 8).tolist()
        obs_embed = ObservationEmbedding((64, 32), angle_sizes, 24)

        colors = torch.rand(64, 32, 3)
        seen = torch.zeros(64, 32, dtype=torch.uint8)
        arm = torch.tensor([0, 128, 64, 32, 16, 8, 4, 4])
        loc = torch.tensor([0, 31])
        target = torch.tensor([63, 0])

        expected = torch.zeros(64, 32, 24)
        expected[:, :, :3] = colors
        expected[:, :, 5:12] = 0.5
        expected[0, 31, 12] = 1
        expected[63, 0, 13] = 1
        expected = expected.transpose(2, 1).transpose(1, 0)

        output = obs_embed(colors, seen, arm, loc, target)
        torch.testing.assert_close(output, expected)

    def test_batched(self):
        angle_sizes = (torch.tensor([64, 32, 16, 8, 4, 2, 1, 1]) * 8).tolist()
        obs_embed = ObservationEmbedding((64, 32), angle_sizes, 24)

        colors = torch.rand(3, 64, 32, 3)
        seen = torch.zeros(3, 64, 32, dtype=torch.uint8)
        arm = torch.tensor([[0, 128, 64, 32, 16, 8, 4, 4]] * 3)
        loc = torch.tensor([[0, 31]] * 3)
        target = torch.tensor([[63, 0]] * 3)

        expected = torch.zeros(3, 64, 32, 24)
        expected[:, :, :, :3] = colors
        expected[:, :, :, 5:12] = 0.5
        expected[:, 0, 31, 12] = 1
        expected[:, 63, 0, 13] = 1
        expected = expected.transpose(3, 2).transpose(2, 1)

        output = obs_embed(colors, seen, arm, loc, target)
        torch.testing.assert_close(output, expected)
