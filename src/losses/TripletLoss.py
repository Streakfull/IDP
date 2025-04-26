import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLossCosine(nn.Module):
    def __init__(self, margin=0.5):
        """
        Triplet loss with margin using cosine similarity.

        Args:
            margin (float): Margin for triplet loss. Default is 0.2.
        """
        super(TripletLossCosine, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Compute the triplet loss using cosine similarity.

        Args:
            anchor (Tensor): Embeddings of the anchor images (B x D).
            positive (Tensor): Embeddings of the positive images (B x D).
            negative (Tensor): Embeddings of the negative images (B x D).

        Returns:
            loss (Tensor): The computed triplet loss.
        """
        # Compute cosine similarities
        positive_similarity = F.cosine_similarity(anchor, positive)
        negative_similarity = F.cosine_similarity(anchor, negative)

        # Convert similarity to distance (cosine distance = 1 - cosine similarity)
        positive_distance = 1 - positive_similarity
        negative_distance = 1 - negative_similarity

        # Compute the triplet loss
        loss = F.relu(positive_distance - negative_distance + self.margin)

        # Average loss across the batch
        return loss.mean(), positive_distance.detach().mean(), negative_distance.detach().mean()
