import unittest

import numpy as np

from .utils import to_image_array, bounds_to_size, ArmHelper, get_cost


class ArmHelperTestCase(unittest.TestCase):
    def test_rotate_single(self):
        cases = [
            ((0, 1, 4), 1),
            ((2, 1, 4), 3),
            ((3, 1, 4), 0),
            ((0, 2, 4), 3),
            ((3, 2, 4), 2),
            ((0, -1, 4), 3),
            ((3, -1, 4), 2),
            ((3, 0, 4), 3),
            ((0, 0, 4), 0),
            ((0, 2, 8), 7),
        ]
        for inputs, expected in cases:
            output = ArmHelper.rotate_single(*inputs)
            self.assertEqual(output, expected)

    def test_rotate_8(self):
        config = (32, 32, 32, 32, 31, 15, 7, 0)
        cases = [
            ((config, (0,) * 8), (32, 32, 32, 32, 31, 15, 7, 0)),
            ((config, (1,) * 8), (33, 33, 33, 33, 0, 0, 0, 1)),
            ((config, (2,) * 8), (31, 31, 31, 31, 30, 14, 6, 7)),
            ((config, (-1,) * 8), (31, 31, 31, 31, 30, 14, 6, 7)),
            ((config, (2, 1, 2, 1, 2, 1, 2, 1)), (31, 33, 31, 33, 30, 0, 6, 1)),
        ]
        arm_helper = ArmHelper((64, 32, 16, 8, 4, 2, 1, 1))

        for inputs, expected in cases:
            output = arm_helper.rotate(*inputs)
            self.assertEqual(output, expected)

    def test_rotate_4(self):
        cases = [(((31, 15, 7, 0), (1, 0, 2, 2)), (0, 15, 6, 7))]
        arm_helper = ArmHelper((4, 2, 1, 1))

        for inputs, expected in cases:
            output = arm_helper.rotate(*inputs)
            self.assertEqual(output, expected)

    def test_from_locs(self):
        arm_helper = ArmHelper((4, 2, 1, 1))
        locs = [(4, 0), (-2, 2), (-1, 0), (-1, 1)]
        self.assertEqual(arm_helper.from_locs(locs), (0, 6, 4, 3))

    def test_to_locs(self):
        arm_helper = ArmHelper((4, 2, 1, 1))
        locs = ((4, 0), (-2, 2), (-1, 0), (-1, 1))
        self.assertEqual(arm_helper.to_locs((0, 6, 4, 3)), locs)

    def test_to_loc(self):
        arm_helper = ArmHelper((4, 2, 1, 1))
        self.assertEqual(arm_helper.to_loc((0, 6, 4, 3)), (0, 3))

    def test_to_str(self):
        arm_helper = ArmHelper((64, 32, 16, 8, 4, 2, 1, 1))
        expected = "64 0;-32 0;-16 0;-8 0;-4 0;-2 0;-1 0;-1 0"
        self.assertEqual(arm_helper.to_str((0, 128, 64, 32, 16, 8, 4, 4)), expected)


class ImageMapTestCase(unittest.TestCase):
    def test_to_image_array(self):
        image_map = {
            (0, 0): (0, 0, 0),
            (-2, -2): (0, 1, 0),
            (-1, -1): (0, 0, 1),
            (1, 1): (1, 0, 0),
            (2, 2): (0, 1, 0),
            (2, -2): (1, 0, 1),
            (1, -1): (0, 1, 1),
            (-1, 1): (1, 1, 0),
            (-2, 2): (1, 0, 1),
        }

        a = np.array(
            [[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]]
        )
        b = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
            ],
        )
        c = np.array(
            [[[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]], [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]]
        )
        d = np.array(
            [[[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]], [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]
        )
        np.testing.assert_allclose(to_image_array(image_map, (-2, -1, -2, -1)), a)
        np.testing.assert_allclose(to_image_array(image_map, (0, 2, 0, 2)), b)
        np.testing.assert_allclose(to_image_array(image_map, (-2, -1, 1, 2)), c)
        np.testing.assert_allclose(to_image_array(image_map, (1, 2, -2, -1)), d)

    def test_bounds_to_size(self):
        self.assertEqual(bounds_to_size((-1, 3, 3, 5)), (5, 3))


class CostTestCase(unittest.TestCase):
    def test_get_cost(self):
        cost = get_cost(np.array([0.1, -0.1, 0.1]), np.array([1, 0, 2, 1])),
        self.assertAlmostEqual(cost, np.sqrt(3) + 0.9)