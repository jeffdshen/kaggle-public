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
