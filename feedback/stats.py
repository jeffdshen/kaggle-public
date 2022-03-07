import numpy as np


class AverageMeter:
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.__init__()

    def add(self, avg, count=1):
        self.count += count
        self.sum += avg * count
        self.avg = self.sum / self.count


class EMAMeter:
    def __init__(self, weight):
        self.avg = 0
        self.sum = 0
        self.inv_count = 1.0
        self.weight = weight

    def reset(self):
        self.__init__(self.weight)

    def add(self, avg, count=1):
        batch_inv_count = self.weight**count
        self.sum = self.sum * batch_inv_count + (1 - batch_inv_count) * avg
        self.inv_count = self.inv_count * batch_inv_count
        self.avg = self.sum / (1 - self.inv_count)


class MinMeter:
    def __init__(self):
        self.min = float("inf")

    def reset(self):
        self.__init__()

    def add(self, avg, count=1):
        if avg < self.min:
            self.min = avg
            return True

        return False


class MaxMeter:
    def __init__(self):
        self.max = float("-inf")

    def reset(self):
        self.__init__()

    def add(self, avg, count=1):
        if avg > self.max:
            self.max = avg
            return True

        return False


class F1Meter:
    def __init__(self):
        self.scores = {}
        self.f1 = 0.0

    def reset(self):
        self.__init__()

    def add(self, scores, count=1):
        for k, v in scores.items():
            if k not in self.scores:
                self.scores[k] = np.array(v)
                continue

            self.scores[k] = np.add(self.scores[k], v)

        f1_score_sum = 0
        f1_score_count = 0
        for k, v in self.scores.items():
            if v[0] + v[1] + v[2] < 1:
                continue
            f1_score = v[0] / (v[0] + 0.5 * v[1] + 0.5 * v[2])
            f1_score_sum += f1_score
            f1_score_count += 1
        self.f1 = f1_score_sum / f1_score_count


class F1EMAMeter:
    def __init__(self, weight):
        self.weight = weight
        self.epoch = round(1.0 / (1.0 - weight))
        # approximately 1 / e
        self.epoch_weight = 0.36603234127322950
        self.epoch_count = 0
        self.total_inv_weight = 1.0
        self.f1_meter = F1Meter()
        self.f1 = self.f1_meter.f1

    def reset(self):
        self.__init__(self.weight)

    def add(self, scores, count=1):
        self.f1_meter.add(scores, count)
        self.epoch_count += count
        if self.epoch_count > self.epoch:
            f1 = self.f1_meter.f1
            self.f1_meter.reset()

            self.epoch_count = 0
            self.total_inv_weight *= self.epoch_weight
            self.f1 = (self.f1 * self.epoch_weight + f1 * (1 - self.epoch_weight)) / (
                1 - self.total_inv_weight
            )
