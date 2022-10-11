import unittest

import numpy as np

from .stats import MeanColRootMeter, EMAMeter, AverageMeter


class MeanColRootMeterTestcase(unittest.TestCase):
    def test_ema_meter(self):
        meter = MeanColRootMeter(EMAMeter(0.9))
        meter.add(np.array([1.0, 2.0]), 3)
        meter.add(np.array([3.0, 2.0]), 2)
        self.assertAlmostEqual(meter.avg, 1.401357864672517)

    def test_avg_meter(self):
        meter = MeanColRootMeter(AverageMeter())
        meter.add(np.array([1.0, 2.0]), 3)
        meter.add(np.array([3.0, 2.0]), 2)
        self.assertAlmostEqual(meter.avg, (np.sqrt(9 / 5.0) + np.sqrt(2)) / 2)
