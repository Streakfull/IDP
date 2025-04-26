import torch.nn.functional as F
from clipreid.timmbackbone import OpenClipModel
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


class Siamese(BaseModel):
    def __init__(self, configs, train_config):
        super().__init__()

        self.configs = configs
        init_type = self.configs['weight_init']
        self.use_visual = self.configs['use_visual']
        self.use_combined = self.configs['use_combined']
        self.use_pnet = self.configs["use_pnet"]
        # self.network = self.build_network()
        self.clip = self.build_network()
        if (init_type != "None"):
            print("Initializing model weights with %s initialization" % init_type)
            self.init_weights()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.contrastive_loss = ContrastiveLoss(margin=0.3)
        self.optimizer = optim.Adam(params=self.parameters(), lr=configs["lr"])
        # self.scheduler = optim.lr_scheduler.StepLR(
        #     self.optimizer, step_size=configs["scheduler_step_size"], gamma=configs["scheduler_gamma"])
        self.scheduler = OneCycleLR(
            optimizer=self.optimizer,
            max_lr=configs["lr"],
            total_steps=train_config['n_epochs'] * train_config['save_every'],
            pct_start=0.05,
            anneal_strategy='cos',
            div_factor=100,
            final_div_factor=1000


        )

    def init_weights(self):
        init_type = self.configs['weight_init']
        gain = self.configs['gain']
        init_weights(self.encoder, init_type=init_type, gain=gain)
        init_weights(self.decoder, init_type=init_type, gain=gain)

    # def forward_og(self, x):
    #     self.x1, self.x2, self.pred = self.network.forward(x)

    #     return self.x1, self.x2, self.pred

    def forward(self, x):
        # self.x1, self.x2, self.pred = self.network.forward(x)
        x1 = x["x1_img"]
        x2 = x["x2_img"]
        x1k = x["x1_kp"]
        x2k = x["x2_kp"]
        self.img1 = x1.detach()
        self.img2 = x2.detach()
        self.x1, self.x2 = self.clip(x1, x2)
        self.pred = F.cosine_similarity(self.x1, self.x2, dim=-1).unsqueeze(1)
        self.target = x["label"]
        return self.x1, self.x2, self.pred

    def set_loss(self):
        # self.bce = self.bce_loss(self.pred*10,
        #                          self.target.unsqueeze(1).float())

        self.cont, self.cos_distance = self.contrastive_loss(
            self.x1, self.x2, self.target)
        # self.loss = self.bce + self.cont
        self.loss = self.cont

    def backward(self):
        self.set_loss()
        self.loss.backward()

    def step(self, x):
        self.train()
        self.optimizer.zero_grad()
        x = self.forward(x)
        self.backward()
        self.optimizer.step()
        self.update_lr()

    def get_metrics(self):
        cos_p = self.cos_distance[self.target == 1].detach()
        cos_n = self.cos_distance[self.target == 0].detach()
        # self.bce = 0
        return {'loss': self.loss.data, 'acc': self.accuracy(), "cont": self.cont, "cosp": cos_p.mean(), "cosn": cos_n.mean()}

    def accuracy(self):
        # predictions = nn.functional.sigmoid(self.pred.detach())

        # predictions[predictions >= 0.5] = 1
        # predictions[predictions < 0.5] = 0
        predictions = self.pred.detach()
        predictions[predictions >= 0.9] = 1
        predictions[predictions < 0.9] = 0
        total_correct = predictions == self.target.unsqueeze(1)
        acc = total_correct.sum()/predictions.shape[0]
        return acc

    def inference(self, x):
        self.eval()
        _, _, pred = self.forward(x)
        return pred

    # def cos_distance(self, x1, x2):
    #     self.eval()
    #     with torch.no_grad():
    #         x1, x2 = self.network.backbone(x1), self.network.backbone(x2)
    #         return self.contrastive_loss.cos_distance(x1, x2)

    def name(self):
        return 'SiameseFCNetwork'

    def prepare_visuals_og(self):

        # Compute softmax predictions and get match/non-match classifation
        pred = torch.nn.functional.sigmoid(self.pred)
        matches = pred > 0.5  # Assuming index 1 corresponds to "match"

        # Get images
        imgs_a = self.network.img1  # Tensor: (batch_size, C, H, W)
        imgs_b = self.network.img2  # Tensor: (batch_size, C, H, W)

        # Initialize lists for storing match and non-match images
        match_images = []
        non_match_images = []

        # Iterate over the batch
        for i in range(len(imgs_a)):
            img_a = imgs_a[i]
            img_b = imgs_b[i]

            # Combine img_a and img_b side by side
            # Concatenate along width (dim=2)
            combined_img = torch.cat((img_a, img_b), dim=2)

            if matches[i]:  # If it's a match
                match_images.append(combined_img)
            else:  # If it's a non-match
                non_match_images.append(combined_img)

        # Check if there are any match images to log
        if match_images:
            match_images_tensor = torch.stack(match_images)

        else:
            print("No match images to log")

        # Check if there are any non-match images to log
        if non_match_images:
            non_match_images_tensor = torch.stack(non_match_images)

        return match_images, non_match_images

    def prepare_visuals(self):

        matches = self.pred >= 0.9  # Assuming index 1 corresponds to "match"

        # Get images
        imgs_a = self.img1  # Tensor: (batch_size, C, H, W)
        imgs_b = self.img2  # Tensor: (batch_size, C, H, W)

        # Initialize lists for storing match and non-match images
        match_images = []
        non_match_images = []

        # Iterate over the batch
        for i in range(len(imgs_a)):
            img_a = imgs_a[i]
            img_b = imgs_b[i]

            # Combine img_a and img_b side by side
            # Concatenate along width (dim=2)
            combined_img = torch.cat((img_a, img_b), dim=2)

            if matches[i]:  # If it's a match
                match_images.append(combined_img)
            else:  # If it's a non-match
                non_match_images.append(combined_img)

        # Check if there are any match images to log
        if match_images:
            match_images_tensor = torch.stack(match_images)

        else:
            print("No match images to log")

        # Check if there are any non-match images to log
        if non_match_images:
            non_match_images_tensor = torch.stack(non_match_images)

        return match_images, non_match_images

    def build_network(self):
        return OpenClipModel(self.configs["name"],
                             self.configs["pretrained"],
                             True)

        # return VisualSiameseNetworkFeaturesKPTransformer()
        # return VisualSiameseNetworkFeaturesKP()
        # return VisualSiameseNetworkFeatures()
        # if self.use_combined:
        #     return VisualSiameseNetworkFeatures()
        # if self.use_pnet:
        #     return SiameseNetworkPnet()
        # if self.use_visual:
        #     return VisualSiameseNetwork()
        # return SiameseNetwork()

    def img_feature(self, x):
        x = self.clip(x)
        return x
