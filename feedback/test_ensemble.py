import unittest

from .ensemble import ensemble


class EnsembleTestCase(unittest.TestCase):
    def test_ensemble(self):
        preds = {
            "0": [
                [([0, 1, 2, 3, 4], "Lead"), ([5, 6, 7], "Evidence")],
                [([0, 1, 2], "Lead"), ([5, 6, 7], "Position")],
            ],
            "1": [
                [([2, 3, 4, 5], "Position"), ([6, 7, 8, 9], "Position")],
                [([3], "Evidence"), ([4, 5, 6], "Evidence")],
            ],
            "2": [
                [([1, 2, 3], "Lead"), ([4, 5, 6], "Evidence")],
                [([2, 3, 4], "Lead"), ([4], "Evidence")],
            ],
        }
        weights = {"0": 0.4, "1": 0.31, "2": 0.29}
        ensemble_preds = ensemble(preds, weights, 0.5)
        expected = [
            [([1, 2, 3], "Lead"), ([4, 5], "Position"), ([6, 7], "Evidence")],
            [([2], "Lead"), ([3], "Evidence"), ([4], "Evidence"), ([5, 6], "Position")],
        ]
        self.assertEqual(ensemble_preds, expected)


if __name__ == "__main__":
    unittest.main()
