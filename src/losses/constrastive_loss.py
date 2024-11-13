import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the euclidean distance and calculate the contrastive loss
        cos_distance = self.cos_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(cos_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - cos_distance, min=0.0), 2))

        return loss_contrastive, cos_distance

    def cos_distance(self, output1, output2):
        return 1 - F.cosine_similarity(
            output1, output2)
