from torchreid.data import ImageDataManager
import torchreid
from clipreid.config import imagedata_kwargs, get_default_config
from clipreid.MergedDataLoader import MergedDataLoader
from pathlib import Path

from sklearn.model_selection import train_test_split
from datasets.base_dataset import BaseDataSet
from torch.utils.data import DataLoader, Subset
from datasets.SoccerNetMatches import SoccerNetDataset
from datasets.SoccerNetTripletDataset import SoccerNetTripleLossDataset
from datasets.JerseyId import JerseyDataset


from training.ModuleLoader import load_module_from_path


class DataLoaderHandler:
    def __init__(self, global_configs: dict, batch_size: int,
                 num_workers: int = 1, test_size: float = 0.2):
        global_dataset_config = global_configs["dataset"]
        dataset_field = global_dataset_config["dataset_field"]
        local_dataset_config = global_dataset_config[dataset_field]

        train_ds = SoccerNetDataset(
            "./raw_dataset/soccernet-tracking/raw/tracking/train3.txt")
        valid_ds = SoccerNetDataset(
            "./raw_dataset/soccernet-tracking/raw/tracking/test3-fixed.txt", train=False)
        self.dataset_type = SoccerNetDataset
        self.dataset = train_ds
        sn_dataloader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,  # Data is usually loaded in parallel by num_workers
            pin_memory=True,
            # This is an implementation detail to speed up data uploading to the GPU
            drop_last=not global_dataset_config["is_overfit"],
            #  drop_last=False)1
        )

        self.val_sn = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,  # Data is usually loaded in parallel by num_workers
            pin_memory=True,
            # This is an implementation detail to speed up data uploading to the GPU
            drop_last=not global_dataset_config["is_overfit"]
            # drop_last=False

        )
        cfg_c = get_default_config()
        cfg_c = imagedata_kwargs(cfg_c)
        datamanager = torchreid.data.ImageDataManager(
            **cfg_c)
        datamanager_tl = datamanager.train_loader
        self.test_reid = datamanager.test_loader
        # self.dataset = train_ds
        # self.train_dataloader = sn_dataloader
        self.validation_dataloader = self.val_sn
        # self.train_dataloader = MergedDataLoader(
        #     datamanager_tl, sn_dataloader)
        self.train_dataloader = datamanager_tl

        # train_ds = JerseyDataset(
        #     root_dir="./raw_dataset/soccernet-jersey/raw/jersey-2023")
        # self.dataset = train_ds

        # cfg = get_default_config()
        # cfg = imagedata_kwargs(cfg)
        # datamanager = torchreid.data.ImageDataManager(
        #     **cfg)
        # self.dataset_type = JerseyDataset

        # self.train_dataloader = datamanager.train_loader
        # self.validation_dataloader = datamanager.test_loader
        # self.train_dataloader = MergedDataLoader(
        #     self.train_dataloader, sn_dataloader)

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
