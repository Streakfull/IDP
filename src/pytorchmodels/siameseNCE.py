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
from clipreid.loss import ClipLoss


class SiameseNCE(BaseModel):
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
        self.clip_loss = ClipLoss("cuda:0")
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

    def get_query_gallery_pairs(self, x1_images, x2_images, labels):
        """
        Helper function to process matching and mismatching pairs of images and assign unique IDs.
        - x1_images: The first image in the pair (query).
        - x2_images: The second image in the pair (gallery).
        - labels: 1 for match, 0 for mismatch.
        """
        I_q, I_g = [], []
        ids_q, ids_g = [], []  # Initialize empty lists for query and gallery IDs
        current_id = 0  # Counter to assign IDs

        for idx in range(len(labels)):
            # Match (query and gallery should have the same ID)
            if labels[idx] == 1:
                I_q.append(x1_images[idx])
                I_g.append(x2_images[idx])
                ids_q.append(current_id)  # Same ID for match
                ids_g.append(current_id)  # Same ID for match
            # Mismatch (query and gallery should have different IDs)
            elif labels[idx] == 0:
                I_q.append(x1_images[idx])
                I_g.append(x2_images[idx])
                ids_q.append(current_id)  # Assign a unique ID to query
                current_id += 1  # Increment ID for mismatch
                # Assign a unique ID to gallery for mismatch
                ids_g.append(current_id)
                current_id += 1  # Increment ID for mismatch

        # Convert lists to tensors
        I_q = torch.stack(I_q)  # [num_queries, C, H, W]
        I_g = torch.stack(I_g)  # [num_gallery, C, H, W]
        ids_q = torch.tensor(ids_q)
        ids_g = torch.tensor(ids_g)

        return I_q, I_g, ids_q, ids_g

    def get_batch_input(self, batch):

        x1_images = batch["x1_img"]  # [B, C, H, W] (Image 1)
        x2_images = batch["x2_img"]  # [B, C, H, W] (Image 2)
        labels = batch["label"]  # [B] (1 for match, 0 for mismatch)

        # Use the helper function to get query-gallery pairs and their IDs
        I_q, I_g, ids_q, ids_g = self.get_query_gallery_pairs(
            x1_images, x2_images, labels)

        return {
            "q": I_q.to("cuda:0"),
            "g": I_g.to("cuda:0"),
            "ids_q": ids_q.to("cuda:0"),  # IDs for query images
            "ids_g": ids_g.to("cuda:0"),  # IDs for gallery images
            "x1_img": x1_images,
            "x2_img": x2_images,
            "label": labels
        }

    def forward(self, x):
        # self.x1, self.x2, self.pred = self.network.forward(x)
        x1 = x["x1_img"]
        x2 = x["x2_img"]
        self.q, self.g, self.ids_g, self.ids_q = x["q"], x["g"], x["ids_g"], x["ids_q"]
        self.x1, self.x2 = self.clip(self.q, self.g)
        self.pred = F.cosine_similarity(self.x1, self.x2, dim=-1).unsqueeze(1)
        self.target = x["label"]
        return self.x1, self.x2, self.pred

    def set_loss(self):

        self.loss = self.clip_loss(
            self.x1, self.x2, self.ids_q, self.ids_g, self.clip.model.logit_scale.exp())

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
        with torch.no_grad():
            cosp, cosn = self.compute_avg_similarity()

        return {'loss': self.loss,
                'cosp': cosp,  # cos p,
                'cosn': cosn,  # cosn,
                "acc": self.accuracy()
                # "cont": self.cont.detach(),
                # "nce": self.nce.detach()
                }

    def accuracy(self):
        """
        Computes accuracy by checking if the most similar gallery image 
        to each query is the correct match (based on the provided IDs).
        """
        with torch.no_grad():
            # Compute cosine similarities between all query-gallery pairs
            cosine_sim = F.cosine_similarity(self.x1.unsqueeze(
                1), self.x2.unsqueeze(0), dim=-1)  # Shape: (Nq, Ng)

            # Identify the index of the most similar gallery image for each query
            # (Nq,) - Index of best match in gallery
            pred_indices = cosine_sim.argmax(dim=1)

            # Check if the predicted gallery index has the correct ID
            correct_matches = self.ids_q == self.ids_g[pred_indices]  # (Nq,)

            # Compute accuracy as the fraction of correctly matched queries
            acc = correct_matches.float().mean().item()

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

        def get_metrics(self):
            with torch.no_grad():
                cosp, cosn = self.compute_avg_similarity()

            return {'loss': self.loss,
                    'cosp': cosp,  # cos p,
                    'cosn': cosn,  # cosn,
                    # "cont": self.cont.detach(),
                    # "nce": self.nce.detach()
                    }

    def compute_avg_similarity(self):
        x1, x2, ids_q, ids_g = self.x1, self.x2, self.ids_q, self.ids_g
        """
        Computes the average cosine similarity between positive and negative pairs.

        Args:
            x1 (torch.Tensor): Query embeddings of shape [N, D]
            x2 (torch.Tensor): Gallery embeddings of shape [N, D]
            ids_q (torch.Tensor): IDs corresponding to x1 (query)
            ids_g (torch.Tensor): IDs corresponding to x2 (gallery)

        Returns:
            avg_positive_sim (float): Average cosine similarity between positive pairs
            avg_negative_sim (float): Average cosine similarity between negative pairs
        """
        # Compute pairwise cosine similarity (N x N)
        cosine_sim = F.cosine_similarity(x1.unsqueeze(
            1), x2.unsqueeze(0), dim=-1)  # Shape: (N, N)

        # Create mask for positive and negative pairs
        positive_mask = ids_q.unsqueeze(1) == ids_g.unsqueeze(
            0)  # True where IDs match (positive pairs)
        negative_mask = ~positive_mask  # Inverse for negative pairs

        # Compute average similarity for positives
        pos_sims = cosine_sim[positive_mask]
        avg_positive_sim = pos_sims.mean().item(
        ) if pos_sims.numel() > 0 else 0.0  # Avoid division by zero

        # Compute average similarity for negatives
        neg_sims = cosine_sim[negative_mask]
        avg_negative_sim = neg_sims.mean().item(
        ) if neg_sims.numel() > 0 else 0.0  # Avoid division by zero

        return avg_positive_sim, avg_negative_sim

    def prepare_visuals(self):
        """
        Prepares and returns two tensors: one containing all query images and
        another containing all gallery images, stacked together.

        Returns:
            query_images_tensor (torch.Tensor or None): Stacked query images
            gallery_images_tensor (torch.Tensor or None): Stacked gallery images
        """
        if not hasattr(self, 'q') or self.q is None or len(self.q) == 0:
            print("No query images to visualize")
            query_images_tensor = None
        else:
            query_images_tensor = self.q if isinstance(
                self.q, torch.Tensor) else torch.stack(self.q)

        if not hasattr(self, 'g') or self.g is None or len(self.g) == 0:
            print("No gallery images to visualize")
            gallery_images_tensor = None
        else:
            gallery_images_tensor = self.g if isinstance(
                self.g, torch.Tensor) else torch.stack(self.g)

        return query_images_tensor, gallery_images_tensor
