from torchvision import models
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from collections import OrderedDict
import numpy as np

def resnet18(num_classes):
    model_resnet = models.resnet18(pretrained=False)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    conv1_1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
    resnet = nn.Sequential(conv1_1, model_resnet)
    return resnet

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)          
        self.bn1 = nn.BatchNorm2d(planes)                       
        self.relu = nn.ReLU(inplace=True)       
        self.conv2 = conv3x3(planes, planes)                    
        self.bn2 = nn.BatchNorm2d(planes)                       
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def resnet(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock)
    return model

class ResNet(nn.Module):

    def __init__(self, block, num_classes=12):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(9216)
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)       
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,                 
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)        
        self.layer1 = block(64,64,stride=1)
        self.pool1 = nn.MaxPool2d(2,stride=2)
        self.downsample1 = nn.Sequential(                                  
                nn.Conv2d(64, 128,                  
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(128),                           
            )
        self.layer2 = block(128,128,stride=1)  
        self.pool2 = nn.MaxPool2d(2,stride=2)  
        self.downsample2 = nn.Sequential(                                  
                nn.Conv2d(128, 64,                  
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(64),                           
            )
        self.layer3 = block(64,64,stride=1)  
        self.pool3 = nn.MaxPool2d(2,stride=2)  
        self.downsample3 = nn.Sequential(                                  
                nn.Conv2d(64, 32,                  
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(32),                           
            )
        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        a,b,c = x.size()
        x = x.view(a,-1)
        x = self.bn0(x)
        x = x.view(a,b,c)
        x = x.unsqueeze(1)
        x = self.conv1_1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x - self.relu(x)
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.downsample1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.downsample2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = self.downsample3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x



class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                            kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i *
                                growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))



class DenseNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(1,1,1,1),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=12):

        super(DenseNet, self).__init__()
        self.bn0 = nn.BatchNorm1d(9216)
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            # print(str(i), num_features)

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.num_features = num_features

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
      

    def forward(self, x):
        a,b,c = x.size()
        x = x.view(a,-1)
        x = self.bn0(x)
        x = x.view(a,b,c)
        x = x.unsqueeze(1)
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=3, stride=1).view(
            features.size(0), -1)
        out = self.classifier(out)
        return out