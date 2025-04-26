import torchvision.transforms as T
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


class SiameseMergedCont(BaseModel):
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
        self.contrastive_loss = ContrastiveLoss(margin=1)
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
        self.x1, self.x2 = self.clip(x)
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
        return {'loss': self.loss.data, 'acc': self.accuracy(), "cosp": cos_p.mean(), "cosn": cos_n.mean()}

    def accuracy(self):
        # predictions = nn.functional.sigmoid(self.pred.detach())

        # predictions[predictions >= 0.5] = 1
        # predictions[predictions < 0.5] = 0
        predictions = self.pred.detach()
        predictions[predictions >= 0.7] = 1
        predictions[predictions < 0.7] = 0
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

        matches = self.pred >= 0.7  # Assuming index 1 corresponds to "match"

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
        if match_images and len(match_images) > 0:
            match_images_tensor = torch.stack(match_images)

        else:
            print("No match images to log")

        # Check if there are any non-match images to log
        if non_match_images and len(non_match_images) > 0:
            non_match_images_tensor = torch.stack(non_match_images)

        if (len(match_images) == 0):
            match_images = None
        if (len(non_match_images) == 0):
            non_match_images = None
        return match_images, non_match_images

    def build_network(self):
        # return OpenClipModel(self.configs["name"],
        #                      self.configs["pretrained"],
        #                      True)

        return VisualSiameseNetworkFeaturesKPTransformer()
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
            x1 = self.clip.crop_kp(crops, kp)
            return x1

    def parse_data_for_eval(self, data):
        imgs = data['img']
        pids = data['pid']
        camids = data['camid']
        return imgs, pids, camids


norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[
    0.229, 0.224, 0.225])
