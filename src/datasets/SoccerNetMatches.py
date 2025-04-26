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


norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[
    0.229, 0.224, 0.225])


class ResizeOrPadToSize:
    def __init__(self, target_height, target_width, fill=0):
        self.target_height = target_height
        self.target_width = target_width
        self.fill = fill  # black padding

    def __call__(self, img):
        w, h = img.size

        if h > self.target_height or w > self.target_width:
            # Resize down while maintaining aspect ratio
            img = ImageOps.contain(
                img, (self.target_width, self.target_height))
            w, h = img.size

        # Now pad if necessary
        pad_w = max(self.target_width - w, 0)
        pad_h = max(self.target_height - h, 0)

        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top

        img = ImageOps.expand(img, border=(
            left, top, right, bottom), fill=self.fill)
        return img


class SoccerNetDataset(Dataset):
    def __init__(self, pairs_file, train=True, crop_size=(224, 224)):
        """
        Args:
            pairs_file (str): Path to the text file containing sample pairs.
            train (bool): Whether to use training or testing transforms.
            crop_size (tuple): Target size for padded crops.
        """
        self.pairs_file = pairs_file
        self.kp_crop_size = (224, 224)
        self.crop_size = crop_size
        self.samples = self._load_pairs()
        self.train = train

        # Define train and test transforms
        self.train_transform = T.Compose([
            T.Resize((128, 128)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.1),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05),
                           scale=(0.9, 1.1)),
            T.GaussianBlur(kernel_size=(5, 5), sigma=(
                0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_transform_kp = T.Compose([
            T.Resize((128, 128)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.1),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05),
                           scale=(0.9, 1.1)),
            T.GaussianBlur(kernel_size=(5, 5), sigma=(
                0.1, 2.0)),
            T.ToTensor(),


        ])

        self.test_transform_kp = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

        self.test_transform = T.Compose([
            # T.Resize((224, 224)),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_transform_f = T.Compose([
            T.Resize((256, 456)),  # Reduce size while maintaining aspect ratio
            T.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),  # Crop & resize
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.1),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05),
                           scale=(0.9, 1.1)),  # Small transformations
            T.GaussianBlur(kernel_size=(5, 5), sigma=(
                0.1, 2.0)),  # Slight blurring
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_transformf = T.Compose([
            T.Resize((256, 456)),  # Keep aspect ratio but downscale
            T.CenterCrop((224, 224)),  # Ensure consistency
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.kp_transform = T.Compose([T.ToTensor()])

        self.sn_transform = T.Compose([
            # ResizeOrPadToSize(256, 128),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Select appropriate transform based on mode
        self.lg_transform = T.Compose([
            T.Resize((224, 224)),
            # T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform = self.train_transform if train else self.test_transform
        self.transformf = self.train_transform_f if train else self.test_transformf
        self.transform_kp = self.train_transform_kp if train else self.test_transform_kp

    def _load_pairs(self):
        with open(self.pairs_file, 'r') as f:
            lines = f.readlines()
        return [line.strip().split(',') for line in lines]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Extract metadata
        game_path = sample[0]
        x1_meta = list(map(int, sample[1:7]))  # frame, id, x, y, w, h
        x2_meta = list(map(int, sample[7:13]))  # frame, id, x, y, w, h
        is_matching = int(sample[13])

        # Load images
        x1_frame_path = os.path.join(
            game_path, "img1", f"{x1_meta[0]:06d}.jpg")
        x2_frame_path = os.path.join(
            game_path, "img1", f"{x2_meta[0]:06d}.jpg")

        x1_img = Image.open(x1_frame_path).convert("RGB")
        x2_img = Image.open(x2_frame_path).convert("RGB")

        # Crop bounding boxes
        x1_crop = self.crop_image(x1_img, x1_meta)
        x2_crop = self.crop_image(x2_img, x2_meta)

        x1_kp = self.crop_image(x1_img, x1_meta)
        x2_kp = self.crop_image(x2_img, x2_meta)

        # Apply transformations
        x1_crop = self.transform(x1_crop)
        x2_crop = self.transform(x2_crop)

        x1_kp = self.transform_kp(x1_kp)
        x2_kp = self.transform_kp(x2_kp)

        x1_img_transformed = self.transformf(x1_img)
        x2_img_transformed = self.transformf(x2_img)

        # x1_kp = self.kp_transform(x1_kp)
        # x2_kp = self.kp_transform(x2_kp)

        return {
            "x1_img": x1_crop,
            "x2_img": x2_crop,
            "label": torch.tensor(is_matching, dtype=torch.float32),
            "x1_frame": x1_img_transformed,  # Return transformed original frame image 1
            "x2_frame": x2_img_transformed,  # Return transformed original frame image 2
            "x1_kp": x1_kp,
            "x2_kp": x2_kp,
            # "x1k": x1_kp,
            # "x2k": x2_kp,
            "meta": {
                "game_path": game_path,
                "x1_frame": x1_meta[0], "x1_id": x1_meta[1],
                "x2_frame": x2_meta[0], "x2_id": x2_meta[1]
            }
        }

    def crop_image(self, img, meta, pad=True):
        """
        Extracts the exact bounding box from the image and then applies padding to match crop_size.

        Args:
            img (PIL.Image): Input image.
            meta (tuple): Metadata containing bounding box (x, y, w, h).

        Returns:
            PIL.Image: Cropped and padded image of size (225, 225).
        """
        x, y, w, h = meta[2], meta[3], meta[4], meta[5]

        # Extract the bounding box region
        cropped_img = img.crop((x, y, x + w, y + h))

        # Pad and resize to crop_size
        if (pad):
            return self.pad_crop(cropped_img)
        else:
            return cropped_img

    def pad_crop(self, img):
        """Pads image to the target crop size while maintaining aspect ratio."""
        if img.width == 0 or img.height == 0:
            # Return black image if invalid size
            return Image.new("RGB", self.crop_size, (0, 0, 0))
        return ImageOps.pad(img, self.crop_size, color=(0, 0, 0))

    @staticmethod
    def create_image_pairs(batch, max_pairs=1000):
        imgs, pids = batch['img'], batch['pid'].numpy()

        # Dictionary mapping pid to indices
        pid_to_indices = {}
        for i, pid in enumerate(pids):
            pid_to_indices.setdefault(pid, []).append(i)

        # Step 1: Ensure at least one positive pair per unique pid
        positive_pairs = []
        remaining_pairs = []  # Store extra pairs for later

        for pid, indices in pid_to_indices.items():
            if len(indices) > 1:
                # Randomly select the first positive pair
                # Randomly pick two different indices
                first_pair = tuple(random.sample(indices, 2))
                positive_pairs.append(first_pair)

                # Store extra pairs for later distribution
                extra_pairs = [
                    (i, j) for i in indices for j in indices if i < j and (i, j) != first_pair]
                remaining_pairs.extend(extra_pairs)

        # Step 2: Shuffle remaining pairs and add them in a round-robin fashion
        random.shuffle(remaining_pairs)

        # Fill up positive pairs without overfilling
        while len(positive_pairs) < max_pairs // 2 and remaining_pairs:
            positive_pairs.append(remaining_pairs.pop())

        # Step 3: Generate negative pairs ensuring diversity
        unique_pids = list(pid_to_indices.keys())
        negative_pairs = set()

        while len(negative_pairs) < len(positive_pairs):
            # Pick two different PIDs
            pid1, pid2 = random.sample(unique_pids, 2)
            i, j = random.choice(pid_to_indices[pid1]), random.choice(
                pid_to_indices[pid2])  # Pick random images from them
            negative_pairs.add((i, j))

        negative_pairs = list(negative_pairs)

        # Step 4: Ensure an equal number of positive and negative pairs
        num_pairs = min(len(positive_pairs), len(
            negative_pairs), max_pairs // 2)
        selected_positive = positive_pairs[:num_pairs]
        selected_negative = negative_pairs[:num_pairs]

        # Prepare final output
    # Step 5: Prepare final output and shuffle
        x1_imgs, x2_imgs, labels = [], [], []

        pairs = [(imgs[i], imgs[j], 1) for i, j in selected_positive] + \
                [(imgs[i], imgs[j], 0) for i, j in selected_negative]

        random.shuffle(pairs)  # Shuffle positives and negatives together

    # Unpack shuffled pairs
        for x1, x2, label in pairs:
            x1_imgs.append(x1)
            x2_imgs.append(x2)
            labels.append(label)
        return {
            "x1_img": torch.stack(x1_imgs),
            "x2_img": torch.stack(x2_imgs),
            "label": torch.tensor(labels, dtype=torch.float32),
        }

    # @staticmethod
    # def move_batch_to_device(batch, device):

    #     if ("pid" in batch.keys()):
    #         batch = SoccerNetDataset.create_image_pairs(
    #             batch, max_pairs=len(batch["pid"]))
    #         x1_img = batch['x1_img'].clone()
    #         x2_img = batch['x2_img'].clone()
    #         x1_img = norm(x1_img)
    #         x2_img = norm(x2_img)
    #         batch["x1_kp"] = batch["x1_img"]
    #         batch["x2_kp"] = batch["x2_img"]
    #         batch["x1_img"] = x1_img
    #         batch["x2_img"] = x2_img
    #     """Moves a batch to the specified device."""
    #     batch["x1_img"] = batch["x1_img"].to(device)
    #     batch["x2_img"] = batch["x2_img"].to(device)
    #     # batch["x1_frame"] = batch["x1_frame"].to(device)
    #     # batch["x2_frame"] = batch["x2_frame"].to(device)
    #     batch["x1_kp"] = batch["x1_kp"].to(device)
    #     batch["x2_kp"] = batch["x2_kp"].to(device)
    #     batch["label"] = batch["label"].to(device)
    #     return batch

    def move_batch_to_device(batch, device):

        if ("pid" in batch.keys()):
            imgs = batch['img']
            kp = imgs.clone()
            # imgs = norm(imgs)
            batch['img'] = imgs
            batch['kp'] = kp
            return batch
        """Moves a batch to the specified device."""
        batch["x1_img"] = batch["x1_img"].to(device)
        batch["x2_img"] = batch["x2_img"].to(device)
        # batch["x1_frame"] = batch["x1_frame"].to(device)
        # batch["x2_frame"] = batch["x2_frame"].to(device)
        batch["x1_kp"] = batch["x1_kp"].to(device)
        batch["x2_kp"] = batch["x2_kp"].to(device)
        batch["label"] = batch["label"].to(device)
        return batch

    def load_frame_crops2(self, det, frame_path):
        x_img = Image.open(frame_path).convert("RGB")
        boxes = det.boxes.xywh.cpu()
        x_crops = []
        for bb in boxes:
            z = torch.zeros(2)
            box = torch.cat((z, bb))
         #   crop = self.crop_image(x_img, box.numpy(), self.crop_size)
            crop = self.crop_image(x_img, box.numpy(), pad=False)
            crop = self.transform(crop)
            x_crops.append(crop)
        x_img = self.transformf(x_img)

        if len(boxes) == 0:
            # Return empty tensors if no detections are found
            return torch.empty(0).to("cuda:0"),  x_img.to("cuda:0")
        x_crops = torch.stack(x_crops)

        return x_crops.to("cuda:0"), x_img.to("cuda:0")

    def load_frame_crops(self, det, frame_path):
        x_img = Image.open(frame_path).convert("RGB")
        boxes = det.boxes.xywh.cpu()
        x_crops = []
        for bb in boxes:
            z = torch.zeros(2)
            box = torch.cat((z, bb))
         #   crop = self.crop_image(x_img, box.numpy(), self.crop_size)
            crop = self.crop_image(x_img, box.numpy(), pad=False)
            crop = self.transform(crop)
            x_crops.append(crop)

        if len(boxes) == 0:
            # Return empty tensors if no detections are found
            return torch.empty(0).to("cuda:0")
        x_crops = torch.stack(x_crops)

        return x_crops.to("cuda:0")

    def load_frame_crops_kp(self, det, frame_path):
        x_img = Image.open(frame_path).convert("RGB")
        boxes = det.boxes.xywh.cpu()
        x_kp = []
        x_crops = []
        for bb in boxes:
            z = torch.zeros(2)
            box = torch.cat((z, bb))
         #   crop = self.crop_image(x_img, box.numpy(), self.crop_size)
            crop = self.crop_image(x_img, box.numpy())
            kp = self.crop_image(x_img, box.numpy())
            crop = self.transform(crop)
            kp = self.transform_kp(kp)
            x_crops.append(crop)
            x_kp.append(kp)
        if len(x_crops) == 0 or len(x_kp) == 0:
            # Return empty tensors if no detections are found
            return torch.empty(0).to("cuda:0"),  torch.empty(0).to("cuda:0")
        x_crops = torch.stack(x_crops)
        x_kp = torch.stack(x_kp)
        return x_crops.to("cuda:0"), x_kp.to("cuda:0")

    def load_frame_crops_det(self, det, frame_path):
        x_img = Image.open(frame_path).convert("RGB")
        x_crops = []
        boxes = [torch.Tensor(d.get_xywh()) for d in det]
        for bb in boxes:
            z = torch.zeros(2)
            box = torch.cat((z, bb))
         #   crop = self.crop_image(x_img, box.numpy(), self.crop_size)
            crop = self.crop_image(x_img, box.numpy(), pad=False)
           # crop = self.sn_transform(crop)
            crop = self.lg_transform(crop)
            x_crops.append(crop)

        if len(boxes) == 0:
            # Return empty tensors if no detections are found
            return torch.empty(0).to("cuda:0")
        x_crops = torch.stack(x_crops)

        return x_crops.to("cuda:0")

    def load_frame_crops_det_lg(self, det, frame_path):
        x_img = Image.open(frame_path).convert("RGB")
        x_crops = []
        boxes = [torch.Tensor(d.get_xywh()) for d in det]
        for bb in boxes:
            z = torch.zeros(2)
            box = torch.cat((z, bb))
         #   crop = self.crop_image(x_img, box.numpy(), self.crop_size)
            crop = self.crop_image(x_img, box.numpy(), pad=False)
            # crop = self.sn_transform(crop)
            crop = self.lg_transform(crop)
            x_crops.append(crop)

        if len(boxes) == 0:
            # Return empty tensors if no detections are found
            return torch.empty(0).to("cuda:0")
        x_crops = torch.stack(x_crops)

        return x_crops.to("cuda:0")

    def load_frame_krops_kp_det(self, det, frame_path):
        x_img = Image.open(frame_path).convert("RGB")
        boxes = [torch.Tensor(d.get_xywh()) for d in det]
        x_kp = []
        x_crops = []
        for bb in boxes:
            z = torch.zeros(2)
            box = torch.cat((z, bb))
         #   crop = self.crop_image(x_img, box.numpy(), self.crop_size)
            crop = self.crop_image(x_img, box.numpy())
            kp = self.crop_image(x_img, box.numpy())
            crop = self.transform(crop)
            kp = self.transform_kp(kp)
            x_crops.append(crop)
            x_kp.append(kp)
        if len(x_crops) == 0 or len(x_kp) == 0:
            # Return empty tensors if no detections are found
            return torch.empty(0).to("cuda:0"),  torch.empty(0).to("cuda:0")
        x_crops = torch.stack(x_crops)
        x_kp = torch.stack(x_kp)
        return x_crops.to("cuda:0"), x_kp.to("cuda:0")
