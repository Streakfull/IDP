import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0, distance_type="cosine"):
        """
        Contrastive Loss with optional distance metric selection.

        Args:
            margin (float): Margin for contrastive loss.
            distance_type (str): "euclidean" or "cosine".
        """
        super().__init__()
        self.margin = margin
        self.distance_type = distance_type.lower()

    def forward(self, output1, output2, label):
        # Calculate the euclidean distance and calculate the contrastive loss
        # distance = self.cos_distance(output1, output2)
        distance = self.compute_distance(output1, output2)
        # distance = torch.cdist(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))

        return loss_contrastive, distance

    def cos_distance(self, output1, output2):
        return 1 - F.cosine_similarity(
            output1, output2)

    def compute_distance(self, output1, output2):
        """Computes either Euclidean or Cosine distance."""
        if self.distance_type == "cosine":
            return 1 - F.cosine_similarity(output1, output2, dim=1)
        elif self.distance_type == "euclidean":
            return torch.norm(output1 - output2, p=2, dim=1)
        else:
            raise ValueError(
                "distance_type must be either 'euclidean' or 'cosine'")
