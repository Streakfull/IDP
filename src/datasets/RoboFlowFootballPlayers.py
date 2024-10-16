import os
from torch.utils.data.dataset import Dataset
import numpy as np


class RoboFlowFootballPlayers(Dataset):
    def __init__(self, path="./raw_dataset/footballPlayers", split="train"):
        self.path = path
        self.split = split
        self.items = self.get_items()

    def get_items(self):
        images = os.listdir(f"{self.path}/{self.split}/images")
        ids = np.core.defchararray.replace(images, '.jpg', '')
        return ids.tolist()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return {
            "img": f"{self.path}/{self.split}/images/{self.items[index]}.jpg",
            "label": f"{self.path}/{self.split}/labels/{self.items[index]}.txt"
        }

    def sample_images(self, n_images=5):
        random_indices = np.random.choice(
            np.arange(0, len(self.items)), size=n_images, replace=False)
        data = []
        for index in random_indices:
            # Use the __getitem__ method to get the image and label for each index
            data.append(self.__getitem__(index))

        return data
