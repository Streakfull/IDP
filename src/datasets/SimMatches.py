import os
from torch.utils.data.dataset import Dataset
import numpy as np
from datasets.base_dataset import BaseDataSet


class SimMatches(BaseDataSet):
    def __init__(self, dataset_options, sim_matches_options):
        super().__init__(
            dataset_options, sim_matches_options)
        self.items = self.get_items()

    def get_items(self):
        items = os.listdir(self.dataset_path)
        return items

    def __getitem__(self, index):
        filename = self.items[index]
        pair = np.loadtxt(f"{self.dataset_path}/{filename}")
        x1f = pair[0][5:]
        x2f = pair[1][5:]
        tgt = int(filename.split("-")[-1].split(".")[0])
        return {
            "x1": x1f,
            "x2": x2f,
            "label": tgt
        }

    def move_batch_to_device(batch, device):
        batch["x1"] = batch['x1'].float().to(device)
        batch["x2"] = batch["x2"].float().to(device)
        batch["label"] = batch["label"].long().to(device)
