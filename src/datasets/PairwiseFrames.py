import os
from torch.utils.data.dataset import Dataset
import numpy as np
from datasets.base_dataset import BaseDataSet
from PIL import Image


class PairwiseFrames(BaseDataSet):
    def __init__(self, dataset_options, pairwise_options):
        super().__init__(
            dataset_options, pairwise_options)
        self.items = self.get_items()
        self.frames_path = self.local_options["raw_frames_path"]
        # self.transform = transforms.Compose([transforms.ToTensor()])

    def get_items(self):
        pairs_array = np.loadtxt(self.dataset_path)
        return pairs_array

    def __getitem__(self, index):
        f1, f2 = self.items[index]
        f1, f2 = int(f1), int(f2)
        x1_img = f"{self.frames_path}/frame_{f1}.jpg"
        x2_img = f"{self.frames_path}/frame_{f2}.jpg"
        x1_img = Image.open(x1_img).convert('RGB')
        x2_img = Image.open(x2_img).convert('RGB')
        x1_img = np.array(x1_img)
        x2_img = np.array(x2_img)

        return {
            "f1": f1,
            "f2": f2,
            "x1": x1_img,
            "x2": x2_img
        }

    def __len__(self):
        return 2500
