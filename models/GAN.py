import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, label_count):
        """
        定义生成器
        Args:
            label_count:数据集类别数
        """
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(label_count, 128 * 24 * 24)
        self.bn1 = nn.BatchNorm1d(128 * 24 * 24)
        self.linear2 = nn.Linear(100, 128 * 24 * 24)
        self.bn2 = nn.BatchNorm1d(128 * 24 * 24)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=(4,4),stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 1, kernel_size=(4,4), stride=2, padding=1)

    def forward(self,x1, x2):
        """
        Args:
            x1: label
            x2: 噪音

        Returns:假样本
        """
        x1 = F.relu(self.linear1(x1))
        x1 = self.bn1(x1)
        x1 = x1.view(-1, 128, 24, 24)
        x2 = F.relu(self.linear2(x2))
        x2 = self.bn2(x2)
        x2 = x2.view(-1, 128, 24, 24)
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.deconv1(x))
        x = self.bn3(x)
        x = F.relu(self.deconv2(x))
        x = self.bn4(x)
        x = torch.tanh(self.deconv3(x))
        return (x + 1)/2



class Discriminator(nn.Module):
    def __init__(self, label_count):
        """
        定义判别器
        Args:
            label_count: 数据集类别数
        """
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(label_count, 1 * 96 * 96)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=(3, 3), stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2)
        self.bn = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256 * 11 * 11, 1)  # 输出概率值

    def forward(self, x1, x2):
        """
        Args:
            x1: label
            x2: 真/假样本

        Returns:概率
        """
        x1 = F.leaky_relu(self.linear(x1))
        x1 = x1.view(-1, 1, 96, 96)
        # x2 = x2.unsqueeze(1）
        x = torch.cat([x1, x2], dim=1)
        x = F.dropout2d(F.leaky_relu(self.conv1(x)))
        x = F.dropout2d(F.leaky_relu(self.conv2(x)))
        x = F.dropout2d(F.leaky_relu(self.conv3(x)))
        x = self.bn(x)
        x = x.view(-1, 256 * 11 * 11)
        x = torch.sigmoid(self.fc(x))
        return x

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, nclass=2):
        super(LSTMClassifier,self).__init__()

        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc  = nn.Linear(hidden_size, nclass)
    def forward(self, x):
        x, _ = self.LSTM(x)
        x = self.fc(x[:, -1, :])
        return x




