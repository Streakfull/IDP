from src.models.base_model import BaseModel
from src.blocks.dummy_network import DummyNetwork
from src.losses.build_loss import BuildLoss
from torch import nn, optim
import torch


class DummyClassifier(BaseModel):

    def __init__(self, configs):
        super().__init__()
        self.network = DummyNetwork(configs["dummy_network"])
        self.criterion = BuildLoss(configs).get_loss()
        self.optimizer = optim.Adam(params=self.parameters(), lr=configs["lr"])
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=configs["scheduler_step_size"], gamma=configs["scheduler_gamma"])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        self.target = x["label"]
        x = self.network(x["voxels"])
        self.predictions = x
        return x

    def backward(self):
        self.loss = self.criterion(self.predictions, self.target)

    def step(self, x):
        self.train()
        self.optimizer.zero_grad()
        x = self.forward(x)
        self.backward()
        self.loss.backward()
        self.optimizer.step()

    def inference(self, x):
        self.eval()
        x = self.forward(x)
        self.backward()
        pred = self.softmax(x)
        pred = torch.max(pred, dim=1).indices
        return pred

    def get_metrics(self):
        return {'loss': self.loss.data}
