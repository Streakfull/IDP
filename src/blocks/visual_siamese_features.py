import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from torchvision import models


class VisualSiameseNetworkFeatures(nn.Module):

    def __init__(self) -> None:
        super(VisualSiameseNetworkFeatures, self).__init__()
        self.resnet = self.load_resnet()

        # self.backbone = nn.Sequential(
        #     nn.Linear(in_features=2048, out_features=4096),
        #     nn.BatchNorm1d(4096),
        # )

        self.backbone_bb = nn.Sequential(
            nn.Linear(in_features=64, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=2048),
            nn.BatchNorm1d(2048),
            # nn.ReLU(inplace=True),


            # nn.Linear(in_features=2048, out_features=4096),
            # nn.BatchNorm1d(4096),
        )

        self.bn_resnet = nn.BatchNorm1d(num_features=2048)

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

        self.cls = nn.Linear(4096, 1)

    def forward(self, x):
        x1 = x["x1_img"]
        x2 = x["x2_img"]
        x1f = x["x1f"]
        x2f = x["x2f"]
        self.img1 = x1.detach()
        self.img2 = x2.detach()
        x1f = self.backbone_bb(x1f)
        x2f = self.backbone_bb(x2f)
        x1 = self.img_feature(x1)
        x2 = self.img_feature(x2)
        x1 = torch.cat((x1, x1f), dim=1)
        x2 = torch.cat((x2, x2f), dim=1)
        x1c = x1.unsqueeze(1)
        x2c = x2.unsqueeze(1)
        fc = torch.cat((x1c, x2c), dim=1)
        last_dim = fc.shape[-1]
        conv_dim = int(np.sqrt(last_dim))
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

    def img_feature(self, img):
        img = self.resnet(img)
        img = img.flatten(start_dim=1)
        img = self.bn_resnet(img)
        return img

    def img_feature_no_bn(self, img):
        img = self.resnet(img)
        img = img.flatten(start_dim=1)
        return img

    def bb_features(self, bb):
        return self.backbone_bb(bb)

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
