import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from torchvision import models
from ultralytics import YOLO, ASSETS


class VisualSiameseNetworkFeatures(nn.Module):

    def __init__(self) -> None:
        super(VisualSiameseNetworkFeatures, self).__init__()

        self.resnet = self.load_resnet()
        self.resnetg = nn.Sequential(
            # Convolutional Block 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (H/2, W/2)

            # Convolutional Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (H/4, W/4)

            # Convolutional Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (H/8, W/8)

            # Convolutional Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (H/16, W/16)

            # Convolutional Block 5
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (H/32, W/32)

            # Convolutional Block 6
            nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # (2048, 1, 1)

            # Flatten to Feature Vector
            nn.Flatten(),  # Output: (2048,)
        )
       # self.pose = self.load_pose()

        # self.backbone_kp = self.load_pose()
        # self.backbone = nn.Sequential(
        #     nn.Linear(in_features=2048, out_features=4096),
        #     nn.BatchNorm1d(4096),
        # )

        # self.vit = self.load_vit()

        # self.backbone_bb = nn.Sequential(
        #     nn.Linear(in_features=384, out_features=512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(in_features=512, out_features=768),
        #     nn.BatchNorm1d(768),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(in_features=768, out_features=1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(in_features=1024, out_features=2048),
        #     nn.BatchNorm1d(2048),
        #     # nn.ReLU(inplace=True),

        #     # nn.Linear(in_features=2048, out_features=4096),
        #     # nn.BatchNorm1d(4096),
        # )

        # self.backbone_kp = nn.Sequential(
        #     nn.Linear(in_features=768, out_features=1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(in_features=1024, out_features=1300),
        #     nn.BatchNorm1d(1300),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(in_features=1300, out_features=1700),
        #     nn.BatchNorm1d(1700),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(in_features=1700, out_features=2048),
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(in_features=2048, out_features=2048),
        #     nn.BatchNorm1d(2048),
        # )
       # self.linear_vit = nn.Linear(in_features=768, out_features=2048)
        self.bn_resnet = nn.BatchNorm1d(num_features=2048)
        self.bn_resnetg = nn.BatchNorm1d(num_features=2048)
        # self.backbone_feat = nn.Linear(in_features=4096, out_features=2048)

        self.comb = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128,
                      kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1,
                      kernel_size=3, padding=1),

        )

        self.fuse = nn.Sequential(
            nn.Linear(in_features=4096, out_features=2048),
        )

        self.cls = nn.Linear(2116, 1)
    # self.cls = nn.Linear(4096, 1)

    def forward(self, x):
        x1 = x["x1_img"]
        x2 = x["x2_img"]

        x1f = x["x1_frame"]
        x2f = x["x2_frame"]

        # x1k = x["x1k"]
        # x2k = x["x2k"]
        self.img1 = x1.detach()
        self.img2 = x2.detach()
        # x1f = self.backbone_bb(x1f)
        # x2f = self.backbone_bb(x2f)

        # x1k = self.backbone_kp(x1k)
        # x2k = self.backbone_kp(x2k)

        x1 = self.img_feature(x1)
        x2 = self.img_feature(x2)

        x1f = self.global_feat(x1f)
        x2f = self.global_feat(x2f)
        # import pdb
        # pdb.set_trace()

        # x1f = torch.cat((x1f, x1k), dim=1)
        # x2f = torch.cat((x2f, x2k), dim=1)

        # x1f = self.backbone_feat(x1f)
        # x2f = self.backbone_feat(x2f)

        x1 = torch.cat((x1, x1f), dim=1)
        x2 = torch.cat((x2, x2f), dim=1)
        x1 = self.fuse(x1)
        x2 = self.fuse(x2)

        # x1 = torch.cat((x1, x1f), dim=1)
        # x2 = torch.cat((x2, x2f), dim=1)
        x1c = x1.unsqueeze(1)
        x2c = x2.unsqueeze(1)
        fc = torch.cat((x1c, x2c), dim=1)
        last_dim = fc.shape[-1]
        # Round up to the nearest square
        conv_dim = int(np.ceil(np.sqrt(last_dim)))

        padded_size = conv_dim**2  # Nearest perfect square

        # Pad the tensor if necessary
        if last_dim < padded_size:
            padding = padded_size - last_dim
            fc = torch.nn.functional.pad(fc, (0, padding))

        fc = rearrange(fc, 'bs ch (w h) -> bs ch w h', w=conv_dim, h=conv_dim)
        fc = self.comb(fc)
        fc = fc.flatten(start_dim=1)
        fc = self.cls(fc)
        return x1, x2, fc

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
        model = YOLO("yolo11n-pose.pt", verbose=False)

        def train_override(self, mode=True):
            pass  # Simply do nothing

        model.train = train_override.__get__(model, YOLO)
        return model

    def backbone_kp(self, img):
        x = self.pose(img, embed=[22])
        x = torch.stack(x)
        return x

    def global_crop_feature(self, crops, frame):
        x1 = self.img_feature(crops)
        x1f = self.global_feat(frame.unsqueeze(0))
        x1f = x1f.repeat(x1.shape[0], 1)
        x1 = torch.cat((x1, x1f), dim=1)
        x1 = self.fuse(x1)
        return x1
