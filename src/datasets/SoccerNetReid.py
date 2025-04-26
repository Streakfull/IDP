import pdb
import random
import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as T
from torch.cuda.amp import autocast
import torchinfo
from pytorchmodels.base_model import BaseModel
from termcolor import cprint
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from clipreid.timmbackbone import OpenClipModel
import torch
import torch.nn.functional as F
import numpy as np
import random
from clipreid.loss import ClipLoss
import clipreid.metrics_reid as metrics
from torchreid.utils import re_ranking
from datasets.base_dataset import BaseDataSet
from collections import Counter
from torchvision import transforms


class SoccerNetReidDataset(BaseDataSet):
    def __init__(self, dataset_options, classif_options):
        self.label_class_map = {
            "Player_team_left": 0,
            "Player_team_right": 1,
            "Main_referee": 2,
            "Side_referee": 3,
            "Goalkeeper_team_left": 4,
            "Goalkeeper_team_right": 5,
            "Staff_members": 6,
            "unknown": 7
        }
        super().__init__(
            dataset_options, classif_options)
        self.split = classif_options["split"]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.class_weights = self.calculate_class_weights()

        # self.dataset_path
        # PosixPath('raw_dataset/soccernet-reid/raw/reid')

    def get_items(self):
        self.dataset_path = self.dataset_path / self.split
        if (self.split == "train"):
            items = self.get_items_train()
            return items
        else:
            return self.get_items_eval()

    def get_items_train(self):
        items = []

        # Go through all leagues
        for league in self.dataset_path.glob("*"):
            if not league.is_dir():
                continue

            # All seasons within a league
            for season in league.glob("*"):
                if not season.is_dir():
                    continue

                # All games within a season
                for game in season.glob("*"):
                    if not game.is_dir():
                        continue

                    # All frame folders within a game
                    for frame_folder in game.glob("*"):
                        if not frame_folder.is_dir():
                            continue

                        # All image files in a frame folder
                        for img_path in frame_folder.glob("*.png"):
                            filename_parts = img_path.name.split("-")

                            if len(filename_parts) < 6:
                                continue  # skip invalid file names

                            class_name = filename_parts[4]

                            label = self.label_class_map.get(
                                class_name, self.label_class_map["Staff_members"])

                            items.append((img_path, label))

        return items

    def get_items_eval(self):
        items = []

        for split_type in ["gallery", "query"]:
            split_path = self.dataset_path / split_type

            # Walk through league/season/game/frame
            for league in split_path.glob("*"):
                if not league.is_dir():
                    continue

                for season in league.glob("*"):
                    if not season.is_dir():
                        continue

                    for game in season.glob("*"):
                        if not game.is_dir():
                            continue

                        for frame_folder in game.glob("*"):
                            if not frame_folder.is_dir():
                                continue

                            for img_path in frame_folder.glob("*.png"):
                                filename_parts = img_path.name.split("-")

                                if len(filename_parts) < 6:
                                    continue  # skip invalid file names

                                class_name = filename_parts[4]
                                label = self.label_class_map.get(
                                    class_name, self.label_class_map["Staff_members"])

                                items.append((img_path, label, split_type))

        return items

    def __getitem__(self, index):
        item = self.items[index]

        # Eval samples have 3 values, train has 2
        if self.split == "train":
            img_path, label = item
            split_type = None
        else:
            img_path, label, split_type = item

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        if split_type is not None:
            return {
                "img": img,
                "label": label,
                "path": str(img_path),
                "split_type": split_type
            }
        else:
            return {
                "img": img,
                "label": label,
                "path": str(img_path)
            }

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Moves each element of the batch (images and features) to the specified device.
        """
        # Move the images and feature tensors to the specified device (e.g., 'cuda:0' or 'cpu')
        batch["img"] = batch["img"].to(device)
        batch["label"] = batch["label"].to(device)

        return batch

    def calculate_class_weights(self):
        """
        Calculate class weights based on the number of occurrences of each class in the dataset.
        Weights are inversely proportional to the frequency of each class.
        """
        if (self.split != "train"):
            return None
        class_counts = Counter()

        # Go through the dataset and count the occurrences of each class label
        for img_path, label in self.items:
            class_counts[label] += 1

        # Calculate the inverse of the class frequency (class weight)
        total_samples = sum(class_counts.values())
        class_weights = {class_id: total_samples /
                         count for class_id, count in class_counts.items()}

        # Normalize the weights (optional step, can be commented out if not needed)
        max_weight = max(class_weights.values())
        normalized_weights = {
            class_id: weight / max_weight for class_id, weight in class_weights.items()}

        # Return the weights in a format that can be used for loss calculation
        return normalized_weights
