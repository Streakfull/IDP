import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks.pointnet.utils import PointNetSetAbstractionNoSample, pc_normalize_torch
from einops import rearrange


class PNet(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super().__init__()
        in_channel = 4 if normal_channel else 2
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionNoSample(
            npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstractionNoSample(
            npoint=128, radius=0.4, nsample=64, in_channel=128 + 2, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstractionNoSample(
            npoint=None, radius=None, nsample=None, in_channel=256 + 2, mlp=[256, 512, 1024], group_all=True)
        # self.fc1 = nn.Linear(1024, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.drop1 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.drop2 = nn.Dropout(0.4)
        # self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        with torch.no_grad():
            norm = pc_normalize_torch(xyz)
            pc = torch.cat((xyz, norm), axis=1)
            xyz = pc
            # xyz = rearrange(pc, "B N D -> B D N")
            # import pdb
            # pdb.set_trace()
        if self.normal_channel:
            norm = xyz[:, 2:, :]
            xyz = xyz[:, :2, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        # import pdb
        # pdb.set_trace()
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = torch.max(l3_points, 2)[0]
        # import pdb
        # pdb.set_trace()
        x = l3_points.view(B, 1024)
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.fc3(x)
        # x = F.log_softmax(x, -1)
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
