import os
from torch.utils.data.dataset import Dataset
import numpy as np
from base_dataset import BaseDataSet


class SimMatches(BaseDataSet):
    def __init__(self, path="./raw_dataset/match_pairs/random_samples_30k", split="train"):
        self.path = path
        self.split = split
        self.items = self.get_items()

    def get_items(self):
        items = os.listdir(self.path)
        return items

    def __getitem__(self, index):
        filename = self.items[index]
        pair = np.loadtxt(f"{self.path}/{filename}")
        x1f = pair[0][5:]
        x2f = pair[1][5:]
        tgt = int(filename.split("-")[-1].split(".")[0])
        return {
            "x1": x1f,
            "x2": x2f,
            "tgt": tgt
        }

    def __len__(self):
        return len(self.items)
