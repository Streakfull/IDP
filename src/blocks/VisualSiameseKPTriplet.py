from tabnanny import verbose
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from torchvision import models
from ultralytics import YOLO, ASSETS

from blocks.PoseTransformer import PoseTransformer


class VisualSiameseKPTriplet(nn.Module):

    def __init__(self) -> None:
        super(VisualSiameseKPTriplet, self).__init__()

        self.resnet = self.load_resnet()
        self.pose = PoseTransformer()
        self.bn_resnet = nn.BatchNorm1d(num_features=2048)
        self.yolo = self.load_pose()

        self.fuse = nn.Sequential(
            nn.Linear(in_features=4096, out_features=2048),
        )

    def forward(self, x):
        x1 = x["x1_img"]  # anchor image
        x2 = x["x2_img"]  # positive image
        x3 = x["x3_img"]  # negative image (new input)

        x1k = x["x1_kp"]
        x2k = x["x2_kp"]
        x3k = x["x3_kp"]

        self.img1 = x1.detach()
        self.img2 = x2.detach()
        self.img3 = x3.detach()

        x1 = self.img_feature(x1)
        x2 = self.img_feature(x2)
        x3 = self.img_feature(x3)

        x1k = self.backbone_kp(x1k)
        x2k = self.backbone_kp(x2k)
        x3k = self.backbone_kp(x3k)

        x1 = torch.cat((x1, x1k), dim=1)
        x2 = torch.cat((x2, x2k), dim=1)
        x3 = torch.cat((x3, x3k), dim=1)

        # Fuse the concatenated features for all three samples
        x1 = self.fuse(x1)
        x2 = self.fuse(x2)
        x3 = self.fuse(x3)
        return x1, x2, x3

    def load_resnet(self):
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
        # for param in feature_extractor.parameters():
        #     param.requires_grad = False
        resnet_features = feature_extractor.to("cuda:0")
        # resnet_features.eval()
       # resnet_features.to("cuda:0")
        return resnet_features

    def load_vit(self):
        vit = models.vit_b_16(models.ViT_B_16_Weights.DEFAULT)
        vit.to("cuda:0")
        feature_extractor = nn.Sequential(*list(vit.children())[:-1])
        conv = feature_extractor[0]
        encoder = feature_extractor[1]

        return vit, encoder

    def img_feature(self, img):
        img = self.resnet(img)
        img = img.flatten(start_dim=1)
        img = self.bn_resnet(img)
        return img

    def global_feat(self, img):
        img = self.resnetg(img)
        img = img.flatten(start_dim=1)
        # img = self.bn_resnetg(img)
        return img

    def img_feature_vit(self, x):
        vit, encoder = self.vit
        n = x.shape[0]
        x = vit._process_input(x)
        batch_class_token = vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = encoder(x)
        x = x[:, 0]
        x = self.linear_vit(x)
        x = self.bn_resnet(x)
        return x

    def img_feature_no_bn(self, img):
        img = self.resnet(img)
        img = img.flatten(start_dim=1)
        return img

    def bb_features(self, bb):
        return self.backbone_bb(bb)

    def kp_features(self, kp):
        return self.backbone_kp(kp)

    def bb_kp_features(self, bb, kp):
        bb = self.bb_features(bb)
        kp = self.kp_features(kp)
        feat = torch.cat((bb, kp), dim=1)
        feat = self.backbone_feat(feat)
        return feat

    def classif(self, x1, x2):
        x1c = x1.unsqueeze(1)
        x2c = x2.unsqueeze(1)
        fc = torch.cat((x1c, x2c), dim=1)
        last_dim = fc.shape[-1]
        conv_dim = int(np.sqrt(last_dim))
        fc = rearrange(fc, 'bs ch (w h) -> bs ch w h', w=conv_dim, h=conv_dim)
        fc = self.comb(fc)
        fc = fc.flatten(start_dim=1)
        fc = self.cls(fc)
        fc = torch.nn.functional.sigmoid(fc)
        return fc

    def load_pose(self):
        model = YOLO("yolo11x-pose.pt", verbose=False)
        model.fuse()

        def train_override(self, mode=True):
            pass  # Simply do nothing

        model.train = train_override.__get__(model, YOLO)
        model.eval()
        return model

    def backbone_kp(self, img):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            x = self.yolo(img.to(device), verbose=False)  # Run YOLO on CUDA

        keypoints_list = []
        batch_size = img.shape[0]  # Get batch size

        for det in x:  # Iterate over detections
            if det.keypoints is not None and det.keypoints.xyn.numel() > 0:
                # Extract normalized keypoints (CUDA)
                keypoints = det.keypoints.xyn.to(device)
                keypoints = keypoints.mean(dim=0, keepdim=True)

            else:
                # If missing, replace with zeros
                keypoints = torch.zeros((1, 17, 2), device=device)

            keypoints_list.append(keypoints)

        if len(keypoints_list) == 0:
            # Return (batch_size, 2048) of zeros
            return torch.zeros((batch_size, 2048), device=device)

        # Shape: (num_objects, 17, 2), all on CUDA
        keypoints_tensor = torch.stack(keypoints_list)
        keypoints_tensor = keypoints_tensor.squeeze(1)
        # Run the PoseTransformer model to extract 2048-D feature vectors
        features = self.pose(keypoints_tensor)  # Shape: (num_objects, 2048)
        return features

    def global_crop_feature(self, crops, frame):
        x1 = self.img_feature(crops)
        x1f = self.global_feat(frame.unsqueeze(0))
        x1f = x1f.repeat(x1.shape[0], 1)
        x1 = torch.cat((x1, x1f), dim=1)
        x1 = self.fuse(x1)
        return x1

    def crop_kp(self, crops, kp):
        x1 = self.img_feature(crops)
        x1k = self.backbone_kp(kp)
        x1 = torch.cat((x1, x1k), dim=1)
        x1 = self.fuse(x1)
        return x1
