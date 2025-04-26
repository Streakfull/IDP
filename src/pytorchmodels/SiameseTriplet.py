import pandas as pd
import torch.nn as nn
import torch
from pytorchmodels.base_model import BaseModel
from blocks.siamese_network import SiameseNetwork
from torch import optim
from lutils.model_utils import init_weights
from losses.TripletLoss import TripletLossCosine
from blocks.visual_siamese_network import VisualSiameseNetwork
from blocks.visual_siamese_features import VisualSiameseNetworkFeatures
from blocks.visual_siamese_features_kp import VisualSiameseNetworkFeaturesKP
from blocks.visual_siamese_features_kp_pose_transformer import VisualSiameseNetworkFeaturesKPTransformer
from blocks.siamese_pnet import SiameseNetworkPnet
from torch.optim.lr_scheduler import OneCycleLR
from blocks.VisualSiameseKPTriplet import VisualSiameseKPTriplet


class TripletSiamese(BaseModel):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs
        init_type = self.configs['weight_init']
        self.use_visual = self.configs['use_visual']
        self.use_combined = self.configs['use_combined']
        self.use_pnet = self.configs["use_pnet"]
        self.network = self.build_network()
        if (init_type != "None"):
            print("Initializing model weights with %s initialization" % init_type)
            self.init_weights()
        self.triplet_loss = TripletLossCosine(
            margin=0.5)  # Triplet loss
        self.optimizer = optim.Adam(params=self.parameters(), lr=configs["lr"])
        self.scheduler = OneCycleLR(
            optimizer=self.optimizer,
            max_lr=configs["lr"],
            total_steps=12 * 7900,
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

    def forward(self, x):
        self.x1, self.x2, self.x3 = self.network.forward(
            x)  # Get anchor, positive, and negative
        # elf.target = x["label"]
        return self.x1, self.x2, self.x3

    def set_loss(self):
        # Triplet loss calculation
        self.triplet_loss_val, self.positive_distance, self.negative_distance = self.triplet_loss(
            self.x1, self.x2, self.x3)
        self.loss = self.triplet_loss_val  # Triplet loss is our main loss now

    def backward(self):
        self.set_loss()
        self.loss.backward()

    def step(self, x):
        self.train()
        self.optimizer.zero_grad()
        x = self.forward(x)
        self.backward()
        self.optimizer.step()

    def get_metrics(self):
        return {'loss': self.loss.data, 'cosp': self.positive_distance, "cosn": self.negative_distance}

    def accuracy(self):
        # Accuracy based on triplet loss might not be as relevant, so it's omitted here.
        return 0

    def inference(self, x):
        self.eval()
        _, _, = self.forward(x)
        return self.x1  # Returning anchor embedding for inference

    def name(self):
        return 'TripletSiameseNetwork'

    def prepare_visuals(self):
        """
        Prepares visualizations for the triplet network, separating the match and non-match examples.
        """
        # In triplet loss, there are no "match" vs "non-match" predictions, but you can visualize
        # the anchor, positive, and negative pairs.
        imgs_a = self.network.img1  # Tensor: (batch_size, C, H, W)
        imgs_b = self.network.img2  # Tensor: (batch_size, C, H, W)
        imgs_c = self.network.img3  # Tensor: (batch_size, C, H, W)

        # Initialize lists for storing the visualizations
        triplet_images = []

        # Iterate over the batch and combine the triplets for visualization
        for i in range(len(imgs_a)):
            img_a = imgs_a[i]
            img_b = imgs_b[i]
            img_c = imgs_c[i]

            # Concatenate the three images (anchor, positive, negative)
            # Concatenate side by side
            combined_img = torch.cat((img_a, img_b, img_c), dim=2)

            triplet_images.append(combined_img)

        if triplet_images:
            triplet_images_tensor = torch.stack(triplet_images)
        else:
            print("No triplet images to log")

        return triplet_images_tensor

    def build_network(self):
        # Return the model that computes the triplet embeddings
        return VisualSiameseKPTriplet()
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
