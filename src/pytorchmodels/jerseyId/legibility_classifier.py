import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ResNet34 based model for binary classification


class LegibilityClassifier34(nn.Module):
    def __init__(self, train=False,  finetune=False):
        super().__init__()
        self.model_ft = models.resnet34(pretrained=True)
        if finetune:
            for param in self.model_ft.parameters():
                param.requires_grad = False
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1)
        self.model_ft.fc.requires_grad = True
        self.model_ft.layer4.requires_grad = True

    def forward(self, x):
        x = self.model_ft(x)
        x = F.sigmoid(x)
        return x


class JerseyNumberClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_ft = models.resnet50(pretrained=True)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 100)

    def forward(self, input):
        return self.model_ft(input)


class JerseyNumberMulticlassClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            *list(models.resnet34(pretrained=True).children())[:-1])

        self.head1 = nn.Linear(512, 100)
        self.head2 = nn.Linear(512, 10)
        self.head3 = nn.Linear(512, 11)

    def forward(self, input):
        # get backbone features
        backbone_feats = self.backbone(input)

        backbone_feats = backbone_feats.reshape(backbone_feats.size(0), -1)

        # pass through heads
        h1 = self.head1(backbone_feats)
        h2 = self.head2(backbone_feats)
        h3 = self.head3(backbone_feats)
        return h1, h2, h3
