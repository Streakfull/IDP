from torch import nn
import torch
from losses.L1_loss import L1


# TODO: Add VQ losses here
class BuildLoss:
    def __init__(self, configs):
        self.configs = configs

    def get_loss(self):
        match self.configs["criterion"]:
            case "BCE":
                if (self.pos_weight == None):
                    return nn.BCEWithLogitsLoss()
                else:
                    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))

            case "CE":
                return nn.CrossEntropyLoss()

            case "MSE":
                return nn.MSELoss(reduction="mean")

            case "L1":
                return L1(reduction="mean")

        return nn.CrossEntropyLoss()
