import numpy as np
from PIL import Image
import cv2
from tabnanny import verbose
import torchvision.utils as vutils
from numpy import NaN
import torchvision.transforms.functional as F
import os
import pandas as pd
import torch.nn as nn
import torch
from pytorchmodels.base_model import BaseModel
from blocks.siamese_network import SiameseNetwork
from torch import optim
from lutils.model_utils import init_weights
from losses.constrastive_loss import ContrastiveLoss
from blocks.visual_siamese_network import VisualSiameseNetwork
from blocks.visual_siamese_features import VisualSiameseNetworkFeatures
from blocks.visual_siamese_features_kp import VisualSiameseNetworkFeaturesKP
from blocks.visual_siamese_features_kp_pose_transformer import VisualSiameseNetworkFeaturesKPTransformer
from blocks.siamese_pnet import SiameseNetworkPnet
from torch.optim.lr_scheduler import OneCycleLR
from pytorchmodels.jerseyId.legibility_classifier import LegibilityClassifier34, JerseyNumberMulticlassClassifier
from termcolor import cprint
from ultralytics import YOLO

from pytorchmodels.jerseyId.strhub.data.module import SceneTextDataModule

ALPHA = 0.5
BETA = 0.25
GAMMA = 0.25


class JerseyID(BaseModel):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        init_type = self.configs['weight_init']
        self.lc = self.load_lc()
        self.pose = self.load_pose()
        self.jerseyc = JerseyNumberMulticlassClassifier()
        self.lc_count = 0
        self.criterion = nn.CrossEntropyLoss()
       # self.parseq = self.load_parseq()

        self.optimizer = optim.Adam(params=self.parameters(), lr=configs["lr"])

        self.scheduler = OneCycleLR(
            optimizer=self.optimizer,
            max_lr=configs["lr"],
            total_steps=10*7788,
            pct_start=0.05,
            anneal_strategy='cos',
            div_factor=100,
            final_div_factor=1000
        )
        self.parseq = self.load_parseq()

    def load_parseqc(self):
        ckpt_path = "./chkpts2/jerseyId/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt"
        parseq = torch.hub.load('baudm/parseq', 'parseq',
                                pretrained=False).eval()
        checkpoint = torch.load(ckpt_path, map_location=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'))

        state_dict = checkpoint['state_dict']
        new_state_dict = {f"model.{k}" if not k.startswith(
            "model.") else k: v for k, v in state_dict.items()}

        parseq.load_state_dict(new_state_dict)
        print("Parseq checkpoint loaded successfully!")
        self.hp = checkpoint["hyper_parameters"]
        return parseq

    def load_parseq(self):
        parseq = torch.hub.load(
            'baudm/parseq', 'parseq_patch16_224', pretrained=True).eval()

        return parseq

    def load_lc(self):
        lc = LegibilityClassifier34()
        ckpt_path = "./chkpts2/jerseyId/legibility_resnet34_soccer_20240215.pth"
        ckpt = torch.load(
            ckpt_path)

        lc.load_state_dict(ckpt)
        cprint(f"{self.name()} loaded from {ckpt_path}")
        return lc

    def load_pose(self):
        model = YOLO("yolo11x-pose.pt", verbose=False)
        model.fuse()

        def train_override(self, mode=True):
            pass  # Simply do nothing

        model.train = train_override.__get__(model, YOLO)
        model.eval()
        return model

    def get_kp_tensor(self, kp):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        keypoints_list = []
        legible_mask = []

        for det in kp:  # Iterate over detections
            if det.keypoints is not None and det.keypoints.xyn.numel() > 0:
                # Extract normalized keypoints (CUDA)
                keypoints = det.keypoints.xyn.to(device)
                keypoints = keypoints.mean(dim=0, keepdim=True)
                mask = 1  # Mark as valid
            else:
                # If missing, replace with zeros
                keypoints = torch.zeros((1, 17, 2), device=device)
                mask = 0  # Mark as invalid

            keypoints_list.append(keypoints)
            legible_mask.append(mask)

        keypoints_tensor = torch.stack(keypoints_list).squeeze(1)
        legible_mask = torch.tensor(
            legible_mask, dtype=torch.bool, device=device)  # Convert to 1D tensor

        return keypoints_tensor, legible_mask

    def crop_torso(self, images, keypoints, padding=5):
        """
        Crops the torso region from images based on keypoints and resizes back to the original size.

        Args:
            images (torch.Tensor): Batch of images [bs, C, H, W] (assumed normalized 0-1 or 0-255).
            keypoints (torch.Tensor): Normalized keypoints [bs, 17, 2] (values in range [0,1]).
            padding (int): Extra padding around the torso.

        Returns:
            torch.Tensor: Images with cropped and resized torso regions.
        """
        bs, C, H, W = images.shape
        cropped_images = []

        # Torso keypoint indices (COCO format: shoulders, hips)
        # Left shoulder, Right shoulder, Left hip, Right hip
        torso_indices = [5, 6, 11, 12]

        for i in range(bs):
            # Convert to HWC format for OpenCV
            img = images[i].permute(1, 2, 0).cpu().numpy()
            # Convert to pixel coordinates
            keypoints_i = keypoints[i] * torch.tensor([W, H]).to(images.device)

            # Filter valid keypoints
            filtered_points = [keypoints_i[j].tolist()
                               for j in torso_indices if keypoints_i[j].sum() > 0]
            if not filtered_points:
                # No valid keypoints, keep original
                cropped_images.append(images[i])
                continue

            # Compute bounding box with padding
            x_min = max(0, min(p[0] for p in filtered_points) - padding)
            x_max = min(W - 1, max(p[0] for p in filtered_points) + padding)
            y_min = max(0, min(p[1] for p in filtered_points) - padding)
            y_max = min(H - 1, max(p[1] for p in filtered_points) + padding)

            # Crop the torso region
            cropped = img[int(y_min):int(y_max), int(x_min):int(x_max), :]
            cropped = torch.tensor(
                cropped, dtype=torch.float32, device=images.device)
            cropped = cropped.cpu().clamp(0, 255).to(torch.uint8)
            save_dir = "../logs/debug_legible_images2"
            save_path = os.path.join(save_dir, f"cropped_{i}.png")
            vutils.save_image(cropped, save_path)
            # Resize back to original dimensions
            # resized = cv2.resize(
            #     cropped, (W, H), interpolation=cv2.INTER_LINEAR)
            # cropped_pil = Image.fromarray((cropped * 255).astype(np.uint8))
            # t = SceneTextDataModule.get_transform([32, 128])
            # resized = t(cropped_pil)
            # cropped_images.append((cropped))
        import pdb
        pdb.set_trace()
        # return torch.stack(cropped_images).to(images.device)

    def forward(self, x, debug_dir="../logs/debug_legible_images2"):
        with torch.no_grad():
            imgs = x["image"]
            labels = x["label"]
            digit1 = x["digit1"]
            digit2 = x["digit2"]
            kp = x["kp"]
            kp_raw = x["kp"]
            pred = self.lc(imgs)
            legible_mask = (pred > 0.5).squeeze(1)
            # Extract only legible instances
            self.imgs = imgs[legible_mask]
            self.target = labels[legible_mask]
            self.dg1 = digit1[legible_mask]
            self.dg2 = digit2[legible_mask]
            self.imgp = x["imgp"][legible_mask]
            kp = kp[legible_mask]
            kp_raw = kp_raw[legible_mask]
            kp = self.pose(kp, verbose=False)
            kp, kp_mask = self.get_kp_tensor(kp)

            kp = self.crop_torso(kp_raw[kp_mask], kp)

            # self.imgs[kp_mask] = kp_crops

            os.makedirs(debug_dir, exist_ok=True)
            for i, img in enumerate(self.imgp):
                # Convert tensor to PIL image
                img_pil = F.to_pil_image(img.cpu())
                img_pil.save(os.path.join(debug_dir, f"legible_{i}.jpg"))

        self.inference_parseq()
        # self.pred = self.jerseyc(self.imgs)
       # return self.pred, self.imgs.shape[0] > 0
        # return self.pred, self.imgs.shape[0] > 0
        return None, None

    def inference_parseq(self):
        logits = self.parseq(self.imgp)
        probs_full = logits[:, :3, :11].softmax(-1)
        self.preds, probs = self.parseq.tokenizer.decode(probs_full)
        logits = logits[:, :3, :11].cpu().detach().numpy()[0].tolist()

    # def get_metrics(self):
    #     acc = self.accuracy()
    #     return {'loss': self.loss,
    #             'lc_count': self.lc_count,
    #             'acc': acc,
    #             'dg1': self.dg1_loss,
    #             'dg2': self.dg2_loss,
    #             'cls': self.cls,
    #             'tp': self.tp,
    #             'predc': self.predc
    #             }

    def get_metrics(self):
        acc = self.accuracy()
        return {'loss': torch.tensor(0),
                'lc_count': self.lc_count,
                'acc': acc,
                'dg1': torch.tensor(0),
                'dg2': torch.tensor(0),
                'cls': torch.tensor(0),
                'tp': self.tp,
                'predc': self.predc
                }

    def set_loss(self):
        pass
        # self.cls = self.criterion(self.pred[0], self.target)
        # self.dg1_loss = self.criterion(self.pred[1], self.dg1)
        # self.dg2_loss = self.criterion(self.pred[2], self.dg2)
        # self.loss = ALPHA * self.cls + BETA * self.dg1_loss + GAMMA * self.dg2_loss

    def set_zero_loss(self):
        self.loss = torch.tensor(float('nan'))
        self.dg1_loss = torch.tensor(float('nan'))
        self.dg2_loss = torch.tensor(float('nan'))
        self.cls = torch.tensor(float('nan'))

    def backward(self):
        self.set_loss()
        self.loss.backward()

    def step(self, x):
        self.train()
        self.optimizer.zero_grad()
        x, is_lc = self.forward(x)
        return
        # if not is_lc or self.pred is None:
        #     self.set_zero_loss()
        #     return
        self.backward()
        self.optimizer.step()
        self.update_lr()

    # def accuracy(self):
    #     if (self.pred is None):
    #         return 0
    #     _, predictions = torch.max(self.pred[0], 1)
    #     total_correct = predictions == self.target
    #     self.tp = total_correct.sum()
    #     self.predc = predictions.shape[0]
    #     acc = self.tp/self.predc
    #     # print(acc, "ACC")
    #     return acc

    def accuracy(self):
        predictions = torch.tensor([int(d)
                                   for d in self.preds], dtype=torch.long).to(self.target.device)
        total_correct = predictions == self.target
        self.tp = total_correct.sum()
        self.predc = predictions.shape[0]
        acc = self.tp/self.predc
        print(acc, "ACC")
        return acc

    def inference(self, x):
        self.eval()
        self.forward(x)
        return self.preds

    def name(self):
        return 'JerseyId'

    def prepare_visuals(self):
        """
        Prepares and returns a tensor containing all legible images stacked together.
        """
        if not hasattr(self, 'imgs') or self.imgs is None or len(self.imgs) == 0:
            print("No legible images to visualize")
            return None

        # Initialize a list to store the images
        legible_images = []

        # Iterate over the images and add them to the list
        for img in self.imgs:
            legible_images.append(img)

        if legible_images:
            legible_images_tensor = torch.stack(legible_images)
        else:
            print("No legible images to log")
            return None

        return legible_images_tensor
