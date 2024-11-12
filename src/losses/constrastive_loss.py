import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the euclidean distance and calculate the contrastive loss
        cos_distance = 1 - F.cosine_similarity(
            output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(cos_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - cos_distance, min=0.0), 2))

        return loss_contrastive
