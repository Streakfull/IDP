from pytorchmodels.base_model import BaseModel
from blocks.team_supervised_classifier import TeamSupervisedClassifier
from losses.build_loss import BuildLoss
from torch import nn, optim
import torch


class TeamClassifier(BaseModel):

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.network = TeamSupervisedClassifier()
        self.criterion = BuildLoss(configs).get_loss()
        self.optimizer = optim.Adam(params=self.parameters(), lr=configs["lr"])
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=configs["scheduler_step_size"], gamma=configs["scheduler_gamma"])
        self.softmax = nn.Softmax(dim=1)
        self.configs["metrics"] = "None"

    def forward(self, x):
        if (not isinstance(x, dict)):
            x = self.network(x)
            return x
        self.imgs = x["img"]
        self.target = x["label"]
        x = self.network(x["img"])
        self.predictions = x
        return x

    def set_loss(self):
        self.loss = self.criterion(self.predictions, self.target)

    def backward(self):
        self.set_loss()
        self.loss.backward()

    def step(self, x):
        self.train()
        self.optimizer.zero_grad()
        x = self.forward(x)
        self.backward()
        self.optimizer.step()

    def inference(self, x):
        self.eval()
        x = self.forward(x)
        # self.backward()
        pred = self.softmax(x)
        pred = torch.max(pred, dim=1).indices
        return pred

    def get_metrics(self):
        return {'loss': self.loss.data, 'acc': self.accuracy()}

    def accuracy(self):
        predictions = nn.functional.softmax(self.predictions.detach(), dim=1)
        predictions = torch.argmax(predictions, dim=1)
        total_correct = predictions == self.target
        acc = total_correct.sum()/predictions.shape[0]
        return acc

    def prepare_visuals(self):
        """
        Prepare visuals for team classifications.

        Returns:
            team_0 (list): List of images classified as team 0.
            team_1 (list): List of images classified as team 1.
            team_2 (list): List of images classified as team 2.
        """
        # Compute softmax predictions and get match/non-match classification
        predictions = nn.functional.softmax(self.predictions.detach(), dim=1)
        predictions = torch.argmax(predictions, dim=1)

        imgs = self.imgs

        # Ensure inputs are valid
        assert len(imgs) == len(
            predictions), "Mismatch between images and predictions."

        # Group images based on predictions
        team_0 = []
        team_1 = []
        team_2 = []

        for img, pred in zip(imgs, predictions):
            if pred == 0:
                team_0.append(img)
            elif pred == 1:
                team_1.append(img)
            elif pred == 2:
                team_2.append(img)

        return team_0, team_1, team_2
