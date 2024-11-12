import torch.nn as nn
import torch
from pytorchmodels.base_model import BaseModel
from blocks.siamese_network import SiameseNetwork
from torch import optim
from lutils.model_utils import init_weights
from losses.constrastive_loss import ContrastiveLoss


class Siamese(BaseModel):
    def __init__(self, configs):
        super().__init__()
        self.network = SiameseNetwork()

        self.configs = configs
        init_type = self.configs['weight_init']
        if (init_type != "None"):
            print("Initializing model weights with %s initialization" % init_type)
            self.init_weights()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.contrastive_loss = ContrastiveLoss()
        self.optimizer = optim.Adam(params=self.parameters(), lr=configs["lr"])
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=configs["scheduler_step_size"], gamma=configs["scheduler_gamma"])

    def init_weights(self):
        init_type = self.configs['weight_init']
        gain = self.configs['gain']
        init_weights(self.encoder, init_type=init_type, gain=gain)
        init_weights(self.decoder, init_type=init_type, gain=gain)

    def forward(self, x):
        self.x1, self.x2, self.pred = self.network.forward(x)
        self.target = x["label"]

    def set_loss(self):

        self.bce = self.bce_loss(self.pred,
                                 self.target.unsqueeze(1).float())

        self.cont = self.contrastive_loss(self.x1, self.x2, self.target)
        import pdb
        pdb.set_trace()
        self.loss = self.bce + self.cont

    def backward(self):
        self.set_loss()
        self.loss.backward()

    def step(self, x):
        self.train()
        self.optimizer.zero_grad()
        x = self.forward(x)
        self.backward()
        self.optimizer.step()

    def get_metrics(self):
        return {'loss': self.loss.data, 'acc': self.accuracy(), "bce": self.bce, "cont": self.cont}

    def accuracy(self):
        predictions = nn.functional.sigmoid(self.predictions.detach())
        predictions = torch.argmax(predictions, dim=1)
        total_correct = predictions == self.target
        acc = total_correct.sum()/predictions.shape[0]
        return acc

    def inference(self, x):
        self.eval()
        _, _, pred = self.forward(x)
        return pred
