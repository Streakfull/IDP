import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets.base_dataset import BaseDataSet
import pdb


class TeamClusterDataset(BaseDataSet):
    """
    Dataset class for loading cropped images and their IDs from the dataset directory.
    """

    def __init__(self, dataset_options, classif_options):
        """
        Initializes the dataset.

        Args:
            dataset_path (str): Path to the dataset directory containing cropped images.
            image_size (tuple): The size to which images should be resized (default: (128, 128)).
            normalize (bool): Whether to normalize images to [-1, 1] (default: True).
        """
        super().__init__(
            dataset_options, classif_options)
        self.dataset_path = "./raw_dataset/teams_clustering"
        image_size = (128, 128)
        normalize = True
        self.image_size = image_size
        self.normalize = normalize
        self.transform = self._build_transform()
        self.data = self._load_data()
        self.player_team_mapping = {
            "1": 0,
            "2": 0,
            "3": 1,
            "4": 1,
            "5": 1,
            "6": 0,
            "7": 0,
            "8": 0,
            "9": 1,
            "10": 0,
            "11": 0,
            "12": 1,
            "13": 1,
            "14": 0,
            "15": 0,
            "16": 0,
            "17": 2,
            "18": 1,
            "19": 2,
            "37": 1,
            "33": 1,
            "51": 0,
            "50": 1
        }

    def _build_transform(self):
        """
        Builds the transformation pipeline.

        Returns:
            torchvision.transforms.Compose: A composition of image transformations.
        """
        transform_steps = [
            transforms.Resize(self.image_size),  # Resize to the specified size
            transforms.ToTensor(),              # Convert to PyTorch tensor
        ]

        if self.normalize:
            transform_steps.append(
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[
                                     0.5, 0.5, 0.5])  # Normalize to [-1, 1]
            )

        return transforms.Compose(transform_steps)

    def _load_data(self):
        """
        Loads the dataset by scanning the dataset directory.

        Returns:
            list: A list of tuples (image_path, crop_id).
        """
        data = []
        for file_name in os.listdir(self.dataset_path):
            if file_name.endswith(".jpg"):  # Check for image files
                # Extract crop ID from file name
                crop_id = int(file_name.split("_")[1])
                image_path = os.path.join(self.dataset_path, file_name)
                data.append((image_path, crop_id))
        return data

    def __len__(self):
        """
        Returns the total number of items in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        image_path, crop_id = self.data[index]
        image = Image.open(image_path).convert("RGB")

       # pdb.set_trace()
        image = self.transform(image)

        # Determine the team label based on the player_team_mapping
        crop_id_str = str(crop_id)  # Convert crop ID to string for mapping
        team_label = self.player_team_mapping.get(
            crop_id_str, 2)  # Default to 2 if not found

        return {"img": image, "crop_id": crop_id, "label": team_label, "path": image_path}

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Moves each element of the batch (images and features) to the specified device.
        """
        # Move the images and feature tensors to the specified device (e.g., 'cuda:0' or 'cpu')
        batch["img"] = batch["img"].to(device)
        batch["label"] = batch["label"].to(device)

        return batch
