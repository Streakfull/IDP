import numpy as np


class BaseMetric():

    def calc(self, A, B):
        pass

    def calc_batch(self, pred, target):
        assert pred.shape[0] == target.shape[0], "{}, {}".format(
            pred.shape[2], pred.shape[3])

        pred = pred.detach().cpu().squeeze(dim=1).numpy()
        target = target.detach().cpu().squeeze(dim=1).numpy()
        result = np.zeros(pred.shape[0])
        for i in range(pred.shape[0]):
            result[i] = self.calc(pred[i], target[i])
        return result.mean()
