'''
MLP模型类
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP4(nn.Module):
    def __init__(self, firstBN):
        super().__init__()
        self.firstBN = firstBN
        self.bn1 = nn.BatchNorm1d(96*96)
        self.fc1 = nn.Linear(96*96, 512)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 64)
        self.relu3 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 12)
    
    def forward(self,din):
        x = din.view(-1,96*96)
        if self.firstBN:
            x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.bn3(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.bn4(x)
        dout = self.fc4(x)
        # dout = F.softmax(x)
        return dout

class MLP1(nn.Module):
    def __init__(self, firstBN):
        super().__init__()
        self.firstBN = firstBN
        self.bn1 = nn.BatchNorm1d(96*96)
        self.fc1 = nn.Linear(96*96, 12)
        self.relu1 = nn.ReLU()
    
    def forward(self,din):
        x = din.view(-1,96*96)
        if self.firstBN:
            x = self.bn1(x)
        dout = self.fc1(x)
        # dout = self.relu1(x)
        # dout = F.softmax(x)
        return dout
