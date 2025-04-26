from blocks.visual_siamese_features_kp_pose_transformer_osc import OSCSiameseNetworkFeaturesKPTransformer
import numpy as np
import random
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
from clipreid.loss import ClipLoss
import clipreid.metrics_reid as metrics
from torchreid.utils import re_ranking
import torchvision.transforms as T
import torch.nn.functional as F


class SiameseNCE(BaseModel):
    def __init__(self, configs, train_config):
        super().__init__()

        self.configs = configs
        init_type = self.configs['weight_init']
        self.use_visual = self.configs['use_visual']
        self.use_combined = self.configs['use_combined']
        self.use_pnet = self.configs["use_pnet"]
        # self.network = self.build_network()
        self.osc = self.build_network()
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

    def prepare_query_gallery(self, batch):
        """Organizes batch into query and gallery sets ensuring each query has a positive match."""

        images = batch["img"]  # Shape: [B, C, H, W]
        pids = batch["pid"]  # Shape: [B]
        kp = batch["kp"]  # Shape: [B, C, H, W] (same shape as images)

        unique_pids = torch.unique(pids)  # Unique player IDs
        I_q, I_g = [], []
        I_qkp, I_gkp = [], []
        pids_q, pids_g = [], []

        for pid in unique_pids:
            indices = (pids == pid).nonzero(as_tuple=True)[
                0]  # Find indices of this PID

            if len(indices) < 2:
                continue  # Skip if only one sample (no positive match)

            # Randomly select one query image from this PID
            query_idx = random.choice(indices.tolist())
            I_q.append(images[query_idx])
            I_qkp.append(kp[query_idx])
            pids_q.append(pid)

            # Remaining images act as positives in the gallery
            gallery_indices = indices[indices != query_idx]
            gallery_imgs = images[gallery_indices]
            gallery_kp = kp[gallery_indices]

            I_g.append(gallery_imgs)  # Store positive matches
            I_gkp.append(gallery_kp)  # Store keypoints images
            pids_g.extend([pid] * len(gallery_imgs))  # Ensure PIDs are aligned

        # Convert lists to tensors
        I_q = torch.stack(I_q)  # [num_queries, C, H, W]
        I_g = torch.cat(I_g, dim=0) if I_g else torch.tensor(
            [])  # [num_gallery, C, H, W]
        I_qkp = torch.stack(I_qkp)  # [num_queries, C, H, W]
        I_gkp = torch.cat(I_gkp, dim=0) if I_gkp else torch.tensor(
            [])  # [num_gallery, C, H, W]
        pids_q = torch.tensor(pids_q)
        pids_g = torch.tensor(pids_g)

        return I_q, I_g, I_qkp, I_gkp, pids_q, pids_g

    def get_batch_input(self, batch):

        if ('pid' in batch.keys()):
            I_q, I_g, I_qkp, I_gkp, pids_q, pids_g = self.prepare_query_gallery(
                batch)

            return {
                "q": I_q.to("cuda:0"),
                "g": I_g.to("cuda:0"),
                "ids_g": pids_g.to("cuda:0"),
                "ids_q": pids_q.to("cuda:0"),
                "camid": batch["camid"].to("cuda:0"),
                "qkp": I_qkp.to("cuda:0"),
                "gkp": I_gkp.to("cuda:0")
            }
        return batch

    def forward(self, x):
        self.q, self.g, self.ids_g, self.ids_q, self.gkp, self.qkp = x[
            "q"], x["g"], x["ids_g"], x["ids_q"], x["qkp"], x["gkp"]
       # self.x1, self.x2 = self.clip(self.q, self.g)

        self.x1, self.x2 = self.osc.forward_x1_x2(
            self.q, self.qkp, self.g, self.gkp)
        return self.x1, self.x2

    def set_loss(self):

        self.loss = self.clip_loss(
            self.x1, self.x2, self.ids_q, self.ids_g)

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
        import pdb
        pdb.set_trace()
        _, _, pred = self.forward(x)
        return pred

    # def cos_distance(self, x1, x2):
    #     self.eval()
    #     with torch.no_grad():
    #         x1, x2 = self.network.backbone(x1), self.network.backbone(x2)
    #         return self.contrastive_loss.cos_distance(x1, x2)

    def name(self):
        return 'SiameseOSCNetwork'

    def build_network(self):
        model = OSCSiameseNetworkFeaturesKPTransformer()
        return model
        # return OpenClipModel(self.configs["name"],
        #                      self.configs["pretrained"],
        #                      True)

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

    def test(
        self,
        dataloader,
        dist_metric='cosine',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        eval_metric='soccernetv3',
        ranks=[1, 5, 10, 20],
        rerank=False,
        export_ranking_results=False
    ):
        r"""Tests model on target datasets.

        .. note::

            This function has been called in ``run()``.

        .. note::

            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``extract_features()`` and ``parse_data_for_eval()`` (most of the time),
            but not a must. Please refer to the source code for more details.
        """
        self.eval()
        targets = list(dataloader.keys())
        last_rank1 = 0
        mAP = 0
        for name in targets:
            domain = 'target'
            print('##### Evaluating {} ({}) #####'.format(name, domain))
            query_loader = dataloader[name]['query']
            gallery_loader = dataloader[name]['gallery']
            rank1, mAP = self._evaluate(
                dataset_name=name,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                eval_metric=eval_metric,
                ranks=ranks,
                rerank=rerank,
                export_ranking_results=export_ranking_results
            )

            # if self.writer is not None and rank1 is not None and mAP is not None:
            #     self.writer.add_scalar(f'Test/{name}/rank1', rank1, self.epoch)
            #     self.writer.add_scalar(f'Test/{name}/mAP', mAP, self.epoch)
            if rank1 is not None:
                last_rank1 = rank1

        return last_rank1, mAP

    @torch.no_grad()
    def _evaluate(
        self,
        dataset_name='',
        query_loader=None,
        gallery_loader=None,
        dist_metric='cosine',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        eval_metric='soccernetv3',
        ranks=[1, 5, 10, 20],
        rerank=False,
        export_ranking_results=False
    ):

        def _feature_extraction(data_loader):
            f_, pids_, camids_ = [], [], []
            for batch_idx, data in enumerate(data_loader):
                imgs, pids, camids = self.parse_data_for_eval(data)
                if True:
                    imgs = imgs.cuda()

                features = self.extract_features(imgs)
                features = features.cpu().clone()
                f_.append(features)
                pids_.extend(pids)
                camids_.extend(camids)
            f_ = torch.cat(f_, 0)
            pids_ = np.asarray(pids_)
            camids_ = np.asarray(camids_)
            return f_, pids_, camids_

        print('Extracting features from query set ...')
        qf, q_pids, q_camids = _feature_extraction(query_loader)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        gf, g_pids, g_camids = _feature_extraction(gallery_loader)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        if normalize_feature:
            print('Normalzing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        print(
            'Computing distance matrix with metric={} ...'.format(
                dist_metric)
        )
        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()

        if rerank:
            print('Applying person re-ranking ...')
            distmat_qq = metrics.compute_distance_matrix(
                qf, qf, dist_metric)
            distmat_gg = metrics.compute_distance_matrix(
                gf, gf, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        if export_ranking_results:
            self.export_ranking_results_for_ext_eval(
                distmat, q_pids, q_camids, g_pids, g_camids, save_dir, dataset_name)

        if not query_loader.dataset.hidden_labels:
            print('Computing CMC and mAP ...')
            cmc, mAP = metrics.evaluate_rank(
                distmat,
                q_pids,
                g_pids,
                q_camids,
                g_camids,
                eval_metric=eval_metric
            )
            print('** Results **')
            print('mAP: {:.1%}'.format(mAP))
            print('CMC curve')
            for r in ranks:
                print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
            return cmc[0], mAP
        else:
            print("Couldn't compute CMC and mAP because of hidden identity labels.")
            return None, None

    def extract_features(self, imgs):
        with torch.no_grad():
            kp = imgs.clone()
            crops = norm(imgs)
            x1 = self.osc.forward_x1_x2(crops, kp)
            return x1

    def img_feature(self, imgs):
        return self.extract_features(imgs)

    def parse_data_for_eval(self, data):
        imgs = data['img']
        pids = data['pid']
        camids = data['camid']
        return imgs, pids, camids


norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[
    0.229, 0.224, 0.225])
