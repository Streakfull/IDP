import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as T

from datasets.SoccerNetMatches import SoccerNetDataset


class SoccerNetTripleLossDataset(SoccerNetDataset):
    def __init__(self, pairs_file, train=True, crop_size=(128, 128)):
        super(SoccerNetTripleLossDataset, self).__init__(
            pairs_file, train, crop_size)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Extract metadata
        game_path = sample[0]
        x1_meta = list(map(int, sample[1:7]))  # frame, id, x, y, w, h
        x2_meta = list(map(int, sample[7:13]))  # frame, id, x, y, w, h
        # New meta for the negative pair
        x3_meta = list(map(int, sample[13:19]))

        # print(len(sample), "SHAPEE")
       # is_matching = int(sample[19])  # For positive/negative pair label

        # Load images
        x1_frame_path = os.path.join(
            game_path, "img1", f"{x1_meta[0]:06d}.jpg")
        x2_frame_path = os.path.join(
            game_path, "img1", f"{x2_meta[0]:06d}.jpg")
        # New frame path for negative
        x3_frame_path = os.path.join(
            game_path, "img1", f"{x3_meta[0]:06d}.jpg")

        x1_img = Image.open(x1_frame_path).convert("RGB")
        x2_img = Image.open(x2_frame_path).convert("RGB")
        x3_img = Image.open(x3_frame_path).convert(
            "RGB")  # Load negative image

        # Crop bounding boxes
        x1_crop = self.crop_image(x1_img, x1_meta)
        x2_crop = self.crop_image(x2_img, x2_meta)
        x3_crop = self.crop_image(x3_img, x3_meta)  # Crop for negative sample

        # Apply transformations
        x1_crop = self.transform(x1_crop)
        x2_crop = self.transform(x2_crop)
        x3_crop = self.transform(x3_crop)  # Transform for negative sample

        x1_kp = self.crop_image(x1_img, x1_meta)
        x2_kp = self.crop_image(x2_img, x2_meta)
        x3_kp = self.crop_image(x3_img, x3_meta)  # Keypoints for negative

        x1_kp = self.transform_kp(x1_kp)
        x2_kp = self.transform_kp(x2_kp)
        x3_kp = self.transform_kp(x3_kp)  # Transform for negative keypoints

        x1_img_transformed = self.transformf(x1_img)
        x2_img_transformed = self.transformf(x2_img)
        # Transform for negative original frame
        x3_img_transformed = self.transformf(x3_img)

        return {
            "x1_img": x1_crop,
            "x2_img": x2_crop,
            "x3_img": x3_crop,  # Return the negative crop
            # "label": torch.tensor(is_matching, dtype=torch.float32),
            "x1_frame": x1_img_transformed,
            "x2_frame": x2_img_transformed,
            "x3_frame": x3_img_transformed,  # Return transformed negative frame
            "x1_kp": x1_kp,
            "x2_kp": x2_kp,
            "x3_kp": x3_kp,  # Return keypoints for negative
            "meta": {
                "game_path": game_path,
                "x1_frame": x1_meta[0], "x1_id": x1_meta[1],
                "x2_frame": x2_meta[0], "x2_id": x2_meta[1],
                # Include meta for negative pair
                "x3_frame": x3_meta[0], "x3_id": x3_meta[1]
            }
        }

    @staticmethod
    def move_batch_to_device(batch, device):
        """Moves a batch to the specified device."""
        batch["x1_img"] = batch["x1_img"].to(device)
        batch["x2_img"] = batch["x2_img"].to(device)
        batch["x3_img"] = batch["x3_img"].to(device)
        # batch["x1_frame"] = batch["x1_frame"].to(device)
        # batch["x2_frame"] = batch["x2_frame"].to(device)
        batch["x1_kp"] = batch["x1_kp"].to(device)
        batch["x2_kp"] = batch["x2_kp"].to(device)
        batch["x3_kp"] = batch["x3_kp"].to(device)
        # batch["label"] = batch["label"].to(device)
        return batch
