import torch
import torch.nn as nn


class PoseTransformer(nn.Module):
    def __init__(self, input_dim=2, embed_dim=512, num_heads=8, output_dim=2048):
        super(PoseTransformer, self).__init__()

        # Keypoint embedding
        self.embedding = nn.Linear(input_dim, embed_dim)

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=4
        )

        # Layer Normalization instead of BatchNorm
        self.ln = nn.LayerNorm(embed_dim)

        # Attention Pooling (learnable query)
        self.attn_pool = nn.Linear(embed_dim, 1)

        # Final MLP to output 2048-D feature vector
        # self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        """
        x: (Batch, 17, 2)  # 17 keypoints, 2D coordinates
        """
        x = self.embedding(x)  # (Batch, 17, embed_dim)
        x = self.transformer(x)  # (Batch, 17, embed_dim)
        x = self.ln(x)  # Normalize features

        # 🔥 Attention-based pooling: Learn which keypoints are important
        attn_weights = torch.softmax(
            self.attn_pool(x), dim=1)  # (Batch, 17, 1)
        # Weighted sum (Batch, embed_dim)
        x = torch.sum(attn_weights * x, dim=1)
        return x

        # return self.fc(x)  # Final 2048-D feature


# # Example usage
# pose = torch.randn(1, 17, 2)  # Batch of 1 pose
# model = PoseTransformer()
# features = model(pose)
# print(features.shape)  # Output: (1, 2048)
