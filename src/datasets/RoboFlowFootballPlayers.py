import os
from torch.utils.data.dataset import Dataset
import numpy as np


class RoboFlowFootballPlayers(Dataset):
    def __init__(self, path="./raw_dataset/footballPlayers", split="train"):
        self.path = path
        self.split = split
        self.items = self.get_items()

    def get_items(self):
        return os.listdir(f"{self.path}/{self.split}/images")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return {
            "path": self.items[index]
        }

    def sample_images(self, n_images=5):
        random_images = np.random.choice(
            self.items, size=n_images, replace=False)
        return random_images
