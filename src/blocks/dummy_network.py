import torch.nn as nn
from einops import rearrange


# A dummy network composed of a deep CNN for classification


# (1, 32, 32, 32)
class DummyNetwork(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=13),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x
