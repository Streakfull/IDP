from pathlib import Path

from sklearn.model_selection import train_test_split
from datasets.base_dataset import BaseDataSet
from torch.utils.data import DataLoader, Subset

from training.ModuleLoader import load_module_from_path


class DataLoaderHandler:
    def __init__(self, global_configs: dict, batch_size: int,
                 num_workers: int = 1, test_size: float = 0.2):
        global_dataset_config = global_configs["dataset"]
        dataset_field = global_dataset_config["dataset_field"]
        local_dataset_config = global_dataset_config[dataset_field]
        self.dataset_type = load_module_from_path(filepath=Path(local_dataset_config["class_filepath"]),
                                                  class_name=local_dataset_config["class"])
        self.dataset = self.dataset_type(
            global_dataset_config, local_dataset_config)

        train_ds = self.dataset
        local_dataset_config["split"] = "valid"
        valid_ds = self.dataset_type(
            global_dataset_config, local_dataset_config)

        # train_ds, valid_ds = self.train_val_dataset(self.dataset, test_size)

        self.train_dataloader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,  # Data is usually loaded in parallel by num_workers
            pin_memory=True,
            # This is an implementation detail to speed up data uploading to the GPU
            drop_last=not global_dataset_config["is_overfit"],
            # drop_last=False

        )

        self.validation_dataloader = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,  # Data is usually loaded in parallel by num_workers
            pin_memory=True,
            # This is an implementation detail to speed up data uploading to the GPU
            drop_last=not global_dataset_config["is_overfit"]
            # drop_last=False

        )

    @staticmethod
    def train_val_dataset(dataset, test_size):
        # return dataset, dataset
        if (len(dataset) == 1):
            return dataset, dataset
        train_idx, val_idx = train_test_split(
            list(range(len(dataset))), test_size=test_size)

        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        return train_ds, val_ds

    def get_dataloaders(self):
        return self.train_dataloader, self.validation_dataloader

    def get_dataset(self):
        return self.dataset
