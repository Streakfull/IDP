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
        self.optimizer = optim.Adam(params=self.parameters(), lr=configs["lr"])
        self.scheduler = OneCycleLR(
            optimizer=self.optimizer,
            max_lr=configs["lr"],
            total_steps=10*13822,
            pct_start=0.05,
            anneal_strategy='cos',
            div_factor=100,
            final_div_factor=1000
        )

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

    def forward(self, x, debug_dir="../logs/debug_legible_images2"):
        with torch.no_grad():
            imgs = x["image"]
            labels = x["label"]
            digit1 = x["digit1"]
            digit2 = x["digit2"]
            pred = self.lc(imgs)
            legible_mask = (pred > 0.5).squeeze(1)
            # Extract only legible instances
            self.imgs = imgs[legible_mask]
            self.target = labels[legible_mask]
            self.dg1 = digit1[legible_mask]
            self.dg2 = digit2[legible_mask]
            self.lc_count = len(self.imgs)  # Count of legible
            if self.lc_count == 0:
                self.pred = None
                return None, False
            # os.makedirs(debug_dir, exist_ok=True)
            # for i, img in enumerate(self.imgs):
            #     # Convert tensor to PIL image
            #     img_pil = F.to_pil_image(img.cpu())
            #     img_pil.save(os.path.join(debug_dir, f"legible_{i}.jpg"))
        self.pred = self.jerseyc(self.imgs)
       # return self.pred, self.imgs.shape[0] > 0
        return self.pred, self.imgs.shape[0] > 0, legible_mask

    def get_metrics(self):
        acc = self.accuracy()
        return {'loss': self.loss,
                'lc_count': self.lc_count,
                'acc': acc,
                'dg1': self.dg1_loss,
                'dg2': self.dg2_loss,
                'cls': self.cls,
                'tp': self.tp,
                'predc': self.predc
                }

    def set_loss(self):
        self.cls = self.criterion(self.pred[0], self.target)
        self.dg1_loss = self.criterion(self.pred[1], self.dg1)
        self.dg2_loss = self.criterion(self.pred[2], self.dg2)
        self.loss = ALPHA * self.cls + BETA * self.dg1_loss + GAMMA * self.dg2_loss

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
        x, is_lc, mask = self.forward(x)
        if not is_lc or self.pred is None:
            self.set_zero_loss()
            return
        self.backward()
        self.optimizer.step()
        self.update_lr()

    def accuracy(self):
        if (self.pred is None):
            return 0
        _, predictions = torch.max(self.pred[0], 1)
        total_correct = predictions == self.target
        self.tp = total_correct.sum()
        self.predc = predictions.shape[0]
        acc = self.tp/self.predc
        # print(acc, "ACC")
        return acc

    def inference(self, x):
        self.eval()
        out = self.forward(x)
        return self.pred, out

    def inference_unlabeled(self, x):
        self.eval()
        out = self.forward_unlabeled(x)
        return out

    def forward_unlabeled(self, x, debug_dir="../logs/debug_legible_images2"):
        with torch.no_grad():
            imgs = x
            pred = self.lc(imgs)
            legible_mask = (pred > 0.5).squeeze(1)
            # Extract only legible instances
            self.imgs = imgs[legible_mask]
            self.lc_count = len(self.imgs)  # Count of legible
            if self.lc_count == 0:
                self.pred = None
                return None, False, []
            # os.makedirs(debug_dir, exist_ok=True)
            # for i, img in enumerate(self.imgs):
            #     # Convert tensor to PIL image
            #     img_pil = F.to_pil_image(img.cpu())
            #     img_pil.save(os.path.join(debug_dir, f"legible_{i}.jpg"))
        self.pred = self.jerseyc(self.imgs)
       # return self.pred, self.imgs.shape[0] > 0
        return self.pred, self.imgs.shape[0] > 0, legible_mask

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

    def inference_unlabeled(self, x):
        with torch.no_grad():
            pred = self.lc(x)
            legible_mask = (pred > 0.5).squeeze(1)
            x = x[legible_mask]
            self.lc_count = len(x)
            # Count of legible
            if self.lc_count == 0:
                return torch.Tensor([]), torch.Tensor([])
            pred = self.jerseyc(x)
            _, pred = torch.max(pred[0], 1)
            return pred, legible_mask
