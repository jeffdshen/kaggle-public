import unittest

import torch
import numpy as np

from .rl_utils import multi_max, multi_argmax


class MultiMaxTestCase(unittest.TestCase):
    def test_zero_dims_left(self):
        x = torch.tensor([[0.1, 0.2], [0.4, 0.3]])
        m = multi_max(x, 2)
        expected = torch.tensor(0.4)
        torch.testing.assert_allclose(m, expected)

    def test_one_dim_left(self):
        x = torch.tensor([[[1.0, 1.1], [1.2, 1.3]], [[0.1, 0.2], [0.4, 0.3]]])
        m = multi_max(x, 2)
        expected = torch.tensor([1.3, 0.4])
        torch.testing.assert_allclose(m, expected)

    def test_two_dim_left(self):
        x = torch.tensor([[[[1.0, 1.1], [1.2, 1.3]], [[0.1, 0.2], [0.4, 0.3]]]])
        m = multi_max(x, 2)
        expected = torch.tensor([[1.3, 0.4]])
        torch.testing.assert_allclose(m, expected)


class MultiArgmaxTestCase(unittest.TestCase):
    def test_zero_dims_left(self):
        x = torch.tensor([[0.1, 0.2], [0.4, 0.3]])
        m = multi_argmax(x, 2)
        expected = np.array([1, 0])
        np.testing.assert_array_equal(m, expected)

    def test_one_dim_left(self):
        x = torch.tensor([[[1.0, 1.1], [1.2, 1.3]], [[0.1, 0.2], [0.4, 0.3]]])
        m = multi_argmax(x, 2)
        expected = np.array([[1, 1], [1, 0]])
        np.testing.assert_array_equal(m, expected)

    def test_two_dim_left(self):
        x = torch.tensor([[[[1.0, 1.1], [1.2, 1.3]], [[0.1, 0.2], [0.4, 0.3]]]])
        m = multi_argmax(x, 2)
        expected = np.array([[[1, 1], [1, 0]]])
        np.testing.assert_array_equal(m, expected)
