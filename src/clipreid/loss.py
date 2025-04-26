import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipLossOG(nn.Module):

    def __init__(self, device):

        super().__init__()

        self.loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.device = device

    def forward(self, query_features, gallery_features, logit_scale):

        query_features = F.normalize(query_features, dim=-1)
        gallery_features = F.normalize(gallery_features, dim=-1)

        logits_per_query = logit_scale * query_features @ gallery_features.T

        logits_per_gallery = logits_per_query.T

        labels = torch.arange(len(logits_per_query),
                              dtype=torch.long, device=self.device)

        loss = (self.loss_function(logits_per_query, labels) +
                self.loss_function(logits_per_gallery, labels))/2

        return loss


class ClipLossCont(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.margin = 1  # Margin value for negative separation
        self.lambda_margin = 1

    def forward(self, query_features, gallery_features, pids_q, pids_g, logit_scale):
        """Compute contrastive loss ensuring multiple positives per query."""

        # Normalize feature vectors
        query_features = F.normalize(query_features, dim=-1)
        gallery_features = F.normalize(gallery_features, dim=-1)

        # Compute cosine similarity logits
        logits_per_query = logit_scale * \
            query_features @ gallery_features.T  # [num_queries, num_gallery]
        logits_per_gallery = logits_per_query.T  # Symmetric loss

        # Create a binary mask for multiple positives
        labels = (pids_q[:, None] == pids_g[None, :]).float().to(
            self.device)  # Shape: [num_queries, num_gallery]

        # Avoid division by zero for soft labels (normalize over positives)
        # Sum positives for each query
        labels_sum = labels.sum(dim=1, keepdim=True)
        # Normalize to sum to 1 (soft labels)
        labels = labels / labels_sum.clamp(min=1)

        # Apply log-softmax to logits
        # Shape: [num_queries, num_gallery]
        log_probs = F.log_softmax(logits_per_query, dim=1)

        # Compute negative log-likelihood loss
        # Soft-label cross-entropy
        loss_q = (-labels * log_probs).sum(dim=1).mean()

        # Symmetric loss (gallery as queries)
        labels_g = labels.T  # Same logic for gallery-to-query
        log_probs_g = F.log_softmax(logits_per_gallery, dim=1)
        loss_g = (-labels_g * log_probs_g).sum(dim=1).mean()

        # Final loss (average both directions)
        loss_infoNCE = (loss_q + loss_g) / 2

        # === Margin Contrastive Loss ===
        cosine_sim = query_features @ gallery_features.T  # Cosine similarity matrix
        distance = 1 - cosine_sim  # Convert similarity to distance

        # Positive pairs: Minimize distance
        pos_mask = labels.bool()
        pos_loss = torch.pow(distance[pos_mask], 2).mean()

        # Negative pairs: Push distance above margin
        neg_mask = ~pos_mask
        neg_loss = torch.pow(torch.clamp(
            self.margin - distance[neg_mask], min=0.0), 2).mean()

        # Final contrastive margin loss
        margin_loss = pos_loss + neg_loss
        total_loss = loss_infoNCE + self.lambda_margin * margin_loss

        return total_loss, margin_loss, loss_infoNCE


class ClipLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.margin = 1  # Margin value for negative separation
        self.lambda_margin = 1

    def forward(self, query_features, gallery_features, pids_q, pids_g, logit_scale=1):
        """Compute contrastive loss ensuring multiple positives per query."""

        # pids_q = pids_q.flatten()
        # pids_g = pids_g.flatten()
        # Normalize feature vectors
        query_features = F.normalize(query_features, dim=-1)
        gallery_features = F.normalize(gallery_features, dim=-1)

        # Compute cosine similarity logits
        logits_per_query = logit_scale * \
            query_features @ gallery_features.T  # [num_queries, num_gallery]
        logits_per_gallery = logits_per_query.T  # Symmetric loss

        # Create a binary mask for multiple positives
        labels = (pids_q[:, None] == pids_g[None, :]).float().to(
            self.device)  # Shape: [num_queries, num_gallery]

        # Avoid division by zero for soft labels (normalize over positives)
        # Sum positives for each query
        labels_sum = labels.sum(dim=1, keepdim=True)
        # Normalize to sum to 1 (soft labels)
        labels = labels / labels_sum.clamp(min=1)

        # Apply log-softmax to logits
        # Shape: [num_queries, num_gallery]
        log_probs = F.log_softmax(logits_per_query, dim=1)

        # Compute negative log-likelihood loss
        # Soft-label cross-entropy

        loss_q = (-labels * log_probs).sum(dim=1).mean()

        # Symmetric loss (gallery as queries)
        labels_g = labels.T  # Same logic for gallery-to-query
        log_probs_g = F.log_softmax(logits_per_gallery, dim=1)
        loss_g = (-labels_g * log_probs_g).sum(dim=1).mean()

        # Final loss (average both directions)
        loss_infoNCE = (loss_q + loss_g) / 2
        return loss_infoNCE

# # Select hardest negatives (most similar wrong matches)
# negative_mask = (pids_q[:, None] != pids_g[None, :]).float().to(
#     self.device)  # 1 if different PID
# # Find most similar wrong match
# hard_negatives = (logits_per_query * negative_mask).max(dim=1)[0]

# # Add margin-based separation (ensures negatives are pushed below a threshold)
# margin = 0.1
# negative_loss = F.relu(hard_negatives - margin).mean()

# margin = 0.1
# negative_loss = F.relu(hard_negatives - margin).mean()
# loss = (loss_q + loss_g) / 2 + 0.5 * negative_loss
