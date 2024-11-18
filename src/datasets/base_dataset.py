"""PyTorch base data set class each data set takes 2 sets of options a global one and a specific dataset config"""
import torch
from pathlib import Path


class BaseDataSet(torch.utils.data.Dataset):
    def __init__(self, global_options: dict, local_options: dict):
        self.global_options = global_options
        self.local_options = local_options
        self.is_overfit = self.global_options.get("is_overfit", False)
        self.overfit_size = self.global_options.get("overfit_size", 0)
        local_options_path = local_options.get("path", None)
        self.dataset_path = Path(
            local_options_path if local_options_path is not None else global_options["path"])
        self.items = self.get_items()

    def get_items(self):
        return []

    def __len__(self):
        if (self.is_overfit and self.overfit_size != 0 and self.overfit_size < len(self.items)):
            return self.overfit_size
        return len(self.items)

    def __getitem__(self, index):
        return None

    @staticmethod
    def move_batch_to_device(batch, device):
        raise Exception("Not Implemented method")

    @staticmethod
    def move_batch_to_device_float(batch, device):
        raise Exception("Not Implemented method")
