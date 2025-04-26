from tabnanny import verbose
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from torchvision import models
from ultralytics import YOLO, ASSETS

from blocks.PoseTransformer import PoseTransformer


class VisualSiameseNetworkFeaturesKPTransformer(nn.Module):

    def __init__(self) -> None:
        super(VisualSiameseNetworkFeaturesKPTransformer, self).__init__()

        self.resnet = self.load_resnet()
        self.pose = PoseTransformer()
        self.bn_resnet = nn.BatchNorm1d(num_features=2048)
        self.yolo = self.load_pose()

        # self.comb = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=128,
        #               kernel_size=1, padding=0),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=128, out_channels=256,
        #               kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=256, out_channels=128,
        #               kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=128, out_channels=64,
        #               kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=1,
        #               kernel_size=3, padding=1),

        # )

        self.fuse = nn.Sequential(
            nn.Linear(in_features=4096, out_features=2048),
        )

        # self.cls = nn.Linear(2116, 1)
    # self.cls = nn.Linear(4096, 1)

    def forward(self, x):
        x1 = x["x1_img"]
        x2 = x["x2_img"]
        x1k = x["x1_kp"]
        x2k = x["x2_kp"]
        self.img1 = x1.detach()
        self.img2 = x2.detach()
        x1 = self.img_feature(x1)
        x2 = self.img_feature(x2)
        x1k = self.backbone_kp(x1k)
        x2k = self.backbone_kp(x2k)
        x1 = torch.cat((x1, x1k), dim=1)
        x2 = torch.cat((x2, x2k), dim=1)
        x1 = self.fuse(x1)
        x2 = self.fuse(x2)

        # x1 = torch.cat((x1, x1f), dim=1)
        # x2 = torch.cat((x2, x2f), dim=1)
        # x1c = x1.unsqueeze(1)
        # x2c = x2.unsqueeze(1)
        # fc = torch.cat((x1c, x2c), dim=1)
        # last_dim = fc.shape[-1]
        # # Round up to the nearest square
        # conv_dim = int(np.ceil(np.sqrt(last_dim)))

        # padded_size = conv_dim**2  # Nearest perfect square

        # # Pad the tensor if necessary
        # if last_dim < padded_size:
        #     padding = padded_size - last_dim
        #     fc = torch.nn.functional.pad(fc, (0, padding))

        # fc = rearrange(fc, 'bs ch (w h) -> bs ch w h', w=conv_dim, h=conv_dim)
        # fc = self.comb(fc)
        # fc = fc.flatten(start_dim=1)
        # fc = self.cls(fc)
        return x1, x2

    def forward_x1_x2(self, x1, x1_kp, x2=None, x2_kp=None):
        if (x2 is not None):
            images = torch.cat([x1, x2], dim=0)
            kp = torch.cat([x1_kp, x2_kp], dim=0)
            image_features = self.crop_kp(images, kp)
            f1, f2 = image_features[:len(x1), :], image_features[len(x1):, :]
            return f1, f2

        f1 = self.crop_kp(x1, x1_kp)
        return f1

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
        # Shape: [B], True if all elements in [17,2] are zero
        # zero_mask = (keypoints_tensor == 0).all(dim=(1, 2))

# Count non-zero keypoints
        # Count how many are NOT all-zero
        # num_nonzero = (~zero_mask).sum().item()

        # print(f"Number of non-zero keypoint tensors in batch: {num_nonzero}")
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
