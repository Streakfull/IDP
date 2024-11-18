from torch import nn
import torch
import torch.nn.functional as F


class L1(nn.Module):
    def __init__(self, reduction='mean'):
        super(L1, self).__init__()
        self.reduction = reduction
        # self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, prediction, target):
        loss = torch.abs(
            prediction.contiguous() - target.contiguous())

        # l

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'batch_mean':
            loss = loss.sum() / loss.flatten(1).shape[0]
            return loss
        elif self.reduction == 'none':
            return loss

        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

    # def forward(self, z):
    #     z = z.flatten(1)
    #     target = F.softmax(torch.rand_like(z), dim=1)
    #     z = F.log_softmax(z, dim=1)
    #     return self.kl_loss(z, target)
    # # input =
