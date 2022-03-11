import unittest

from .ensemble import ensemble


def lr(a, b):
    return list(range(a, b))


class EnsembleTestCase(unittest.TestCase):
    def test_ensemble(self):
        preds = {
            "0": [
                [(lr(0, 5), "Lead"), (lr(5, 8), "Evidence")],
                [(lr(0, 3), "Lead"), (lr(5, 8), "Position")],
            ],
            "1": [
                [(lr(2, 6), "Position"), (lr(6, 10), "Position")],
                [(lr(3, 4), "Evidence"), (lr(4, 7), "Evidence")],
            ],
            "2": [
                [(lr(1, 4), "Lead"), (lr(4, 7), "Evidence")],
                [(lr(2, 4), "Lead"), (lr(4, 5), "Evidence")],
            ],
        }
        weights = {"0": 0.4, "1": 0.31, "2": 0.29}
        ensemble_preds = ensemble(preds, weights, 0.5)
        expected = [
            [(lr(1, 4), "Lead"), (lr(4, 6), "Position"), (lr(6, 8), "Evidence")],
            [([2], "Lead"), ([3], "Evidence"), ([4], "Evidence"), ([5, 6], "Position")],
        ]
        self.assertEqual(ensemble_preds, expected)

    def test_ensemble_trim_recurse(self):
        preds = {
            "0": [
                [
                    (lr(5, 18), "Lead"),
                    (lr(18, 84), "Evidence"),
                    (lr(99, 117), "Position"),
                    (lr(117, 326), "Evidence"),
                    (lr(326, 357), "Concluding Statement"),
                ]
            ],
            "1": [
                [
                    (lr(5, 18), "Lead"),
                    (lr(18, 99), "Evidence"),
                    (lr(99, 117), "Position"),
                    (lr(216, 227), "Claim"),
                    (lr(228, 326), "Evidence"),
                    (lr(326, 357), "Concluding Statement"),
                ]
            ],
            "2": [
                [
                    (lr(5, 18), "Lead"),
                    (lr(18, 326), "Evidence"),
                    (lr(326, 357), "Concluding Statement"),
                ]
            ],
        }
        weights = {"0": 0.4, "1": 0.31, "2": 0.29}
        ensemble_preds = ensemble(preds, weights, 0.5)
        expected = [
            [
                (lr(5, 18), "Lead"),
                (lr(18, 99), "Evidence"),
                (lr(99, 117), "Position"),
                (lr(117, 326), "Evidence"),
                (lr(326, 357), "Concluding Statement"),
            ]
        ]
        self.assertEqual(ensemble_preds, expected)


if __name__ == "__main__":
    unittest.main()
