import unittest

from .datasets import score

class ScoreTestCase(unittest.TestCase):
    def test_score(self):
        preds = [[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [5, 4, 3]]
        labels = ["Effective", "Ineffective", "Adequate"]
        loss = score(preds, labels)
        expected = 11.953510744964333
        self.assertAlmostEqual(loss, expected)

if __name__ == "__main__":
    unittest.main()
