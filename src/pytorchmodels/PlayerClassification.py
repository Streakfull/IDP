from pytorchmodels.base_model import BaseModel
from blocks.resnetClassif import PlayerClassifier
from torch.optim.lr_scheduler import OneCycleLR
from clipreid.timmbackbone import OpenClipModel
from torch import optim
from torch import nn
import torch


class PlayerClassification(BaseModel):
    def __init__(self, configs, train_config):
        super().__init__()
        self.configs = configs
        self.network = self.build_network()
        class_weights = {0: 0.0329696394686907, 1: 0.03308830530745939, 3: 0.41172985781990523,
                         2: 0.3257710696540733, 4: 0.43622897313582726, 5: 0.47838656387665196, 6: 1.0}
        sorted_weights = [class_weights[i] if i in class_weights else 0.0 for i in range(
            max(class_weights.keys()) + 1)]
        weights_tensor = torch.tensor(sorted_weights, dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(
            weight=weights_tensor)  # Here weights
        self.optimizer = optim.Adam(params=self.parameters(), lr=configs["lr"])
        self.scheduler = OneCycleLR(
            optimizer=self.optimizer,
            max_lr=configs["lr"],
            total_steps=train_config['n_epochs'] * train_config['save_every'],
            pct_start=0.05,
            anneal_strategy='cos',
            div_factor=100,
            final_div_factor=1000
        )

    def build_network(self):
        return PlayerClassifier(num_classes=7)

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
        self.update_lr()

    def get_metrics(self):

        return {'loss': self.loss,
                "acc": self.accuracy()

                }

    def accuracy(self):
        # Get the predicted class (index of max logit)
        _, predicted = torch.max(self.predictions, 1)

        # Compare predictions with true labels
        correct = (predicted == self.target).sum().item()

        # Compute accuracy as a percentage
        # Number of correct predictions / total samples
        accuracy = correct / self.target.size(0)
        return accuracy

    def inference(self, x):
        self.eval()
        if (not isinstance(x, dict)):
            x = self.network(x)
            return x
        x = self.forward(x["img"])
        return x

    def inference_with_cls(self, x):
        self.eval()
        x = self.network(x)
        _, predicted = torch.max(x, 1)
        return predicted

    def prepare_visuals(self):
        """
        Prepare visuals for team classifications.

        Returns:
            A stack of images for each class group (Players, Goalkeepers, Main Referee,
            Side Referee, and Staff Members).
        """
        # Compute softmax predictions and get match/non-match classification
        predictions = nn.functional.softmax(self.predictions.detach(), dim=1)
        predictions = torch.argmax(predictions, dim=1)

        imgs = self.imgs

        # Ensure inputs are valid
        assert len(imgs) == len(
            predictions), "Mismatch between images and predictions."

        # Group images based on predictions
        players = []  # Group for players (Player_team_left, Player_team_right)
        # Group for goalkeepers (Goalkeeper_team_left, Goalkeeper_team_right)
        goalkeepers = []
        main_referee = []  # Group for main referee (Main_referee)
        side_referee = []  # Group for side referee (Side_referee)
        staff_members = []  # Group for staff members (Staff_members)

        for img, pred in zip(imgs, predictions):
            if pred == 0 or pred == 1:  # Player_team_left or Player_team_right
                players.append(img)
            elif pred == 4 or pred == 5:  # Goalkeeper_team_left or Goalkeeper_team_right
                goalkeepers.append(img)
            elif pred == 2:  # Main_referee
                main_referee.append(img)
            elif pred == 3:  # Side_referee
                side_referee.append(img)
            elif pred == 6:  # Staff_members
                staff_members.append(img)

        # Return a stack of images for each group
        players_stack = torch.stack(players) if players else None
        goalkeepers_stack = torch.stack(goalkeepers) if goalkeepers else None
        main_referee_stack = torch.stack(
            main_referee) if main_referee else None
        side_referee_stack = torch.stack(
            side_referee) if side_referee else None
        staff_members_stack = torch.stack(
            staff_members) if staff_members else None

        return players_stack, goalkeepers_stack, main_referee_stack, side_referee_stack, staff_members_stack

    def name(self):
        return "crop_classifier"
