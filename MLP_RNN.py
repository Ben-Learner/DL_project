# -*- coding: utf-8 -*-
"""
@Time ： 2022/12/19 16:31
@Auth ： 王彧
@File ：RNN.py
@IDE Spyder
@Motto：AIE(Bug Is Everywhere)
"""
import sys
sys.path.append('.\\models')
import os
from os import path as osp
import numpy as np
import math
import torch
import torch.nn as nn
from RNN import RNN
from MLP import MLP1,MLP4
import argparse
from Construct_dataset import My_dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from thop import profile
from sklearn.metrics import confusion_matrix

labels = ['FLB', 'LLB', 'LOCA', 'LOCAC', 'LR', 'MD', 'RI', 'RW', 'SGATR', 'SGBTR', 'SLBIC', 'SLBOC']
tick_marks =  np.array(range(len(labels)))+0.5
rnntype = ['LSTM', 'GRU']
mlptype = ['MLP1', 'MLP4']
model_name = 'MLP1' # GRU,LSTM,MLP

dataset_flag = 'Fake' # Real, Fake, Total
fake_flag = 'TGAN'
if dataset_flag in ['Fake', 'Total']:
    fake_name = '_' + fake_flag
else:
    fake_name = ''

firstBN = True
if firstBN:
    firstBN_name = '_FirstBN'
else:
    firstBN_name = ''

parser = argparse.ArgumentParser(description='GRU and LSTM of RNN')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')
parser.add_argument('--raw_path', type=str, default='./dataset/raw_dataset', help='真实数据集路径')
parser.add_argument('--generate_path', type=str, default='./dataset/generated_dataset/' + fake_flag, help='生成数据集路径')
parser.add_argument('--saved_models_path', type=str, default='./saved_models/RNN_models/' + model_name + '.pt', help='模型保存')
parser.add_argument('--nepoches', type=str, default=250, help='迭代次数')
parser.add_argument('--batch_size', type=str, default=32, help='Batch大小')
parser.add_argument('--nhid', type=int, default=300, help="size of hidden units per layer")
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='initial momentum')
parser.add_argument('--runtype', type=str, default=model_name, help='type of network model')
parser.add_argument('--bidirect', type=str, default=False, help='bidirectional setting')
parser.add_argument('--dropout', type=str, default=0, help='dropout')
parser.add_argument('--train_curve_name', type=str, default='train_curve_' + model_name + '_' + dataset_flag + fake_name + firstBN_name, help='save the train/valid loss curve')

args = parser.parse_args()

# 设置随机种子
torch.manual_seed(1234)

# 构建数据集
real_dataset = My_dataset(args.raw_path)  # 真实数据集
fake_dataset = My_dataset(args.generate_path, fake_dataset=True)  # 生成数据集

# 加载数据集
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset=real_dataset,
                                                                           lengths=[0.8, 0.1, 0.1])

if dataset_flag == 'Real':
    pass
elif dataset_flag == 'Fake':
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset=fake_dataset, 
                                                                 lengths=[0.8, 0.2])
elif dataset_flag == 'Total':
    train_dataset0, valid_dataset0 = torch.utils.data.random_split(dataset=fake_dataset, 
                                                                 lengths=[0.8, 0.2])
    train_dataset = train_dataset0 + train_dataset
    valid_dataset = valid_dataset0 + valid_dataset
    
train_loader, valid_loader, test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True), \
                                          DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True), \
                                          DataLoader(test_dataset, batch_size=args.batch_size)

# Use gpu or cpu to train
use_gpu = True

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")
    
if args.runtype in mlptype:
    if args.runtype == 'MLP1':
        model = MLP1(firstBN).to(device)
    elif args.runtype == 'MLP4':
        model = MLP4(firstBN).to(device)
elif args.runtype in rnntype:
    model = RNN(args.runtype, args.bidirect, args.dropout, 96, args.nhid, args.nlayers, 12, device, firstBN).to(device)

def train():
    # 定义损失函数和优化器
    lossfunc = nn.CrossEntropyLoss()
    # lossfunc = nn.NLLLoss()
    optim = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum)
    best_accuracy = 0
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(args.nepoches):
        train_loss = 0.0
        for data, target in train_loader:
            target = target.long()
            data, target = data.to(device), target.to(device)
            optim.zero_grad()
            pred = model(data)
            loss = lossfunc(pred,target)
            loss.backward()
            optim.step()
            train_loss += loss.item() * data.size(0)
        valid_loss, acc_valid = valid()
        if acc_valid >= best_accuracy:
            torch.save(model, args.saved_models_path)
        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch: {}\tTraing Loss: {:.6f}\tValid Loss: {:.6f}\tValid Accuracy: {:.2%}'.format(epoch + 1, train_loss, valid_loss, acc_valid))
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
    return train_loss_list, valid_loss_list
    
def valid():
    corr = 0.0
    total = 0.0
    lossfunc = nn.CrossEntropyLoss()
    with torch.no_grad():
        valid_loss = 0.0
        for data, target in valid_loader:
            # Loss
            target = target.long()
            data, target = data.to(device), target.to(device)
            pred = model(data)
            loss = lossfunc(pred,target)
            valid_loss += loss.item() * data.size(0)
            
            # 准确率
            pred = model(data)
            _, predicted = torch.max(pred.data, 1)
            total += target.size(0)
            corr += (predicted == target).sum().item()
        valid_loss = valid_loss / len(valid_loader.dataset)
    return valid_loss, corr / total

def test():
    model = torch.load(args.saved_models_path).to(device)
    total = 0.0
    corr  = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            y_true.extend(target.to('cpu') )
            data, target = data.to(device), target.to(device)
            pred = model(data)
            _, predicted = torch.max(pred.data, 1)
            y_pred.extend(predicted.to('cpu'))
            total += target.size(0)
            corr += (predicted == target).sum().item()
    print("Test Accuracy: {:.2%}".format(corr / total)) 
    return np.array(y_true), np.array(y_pred)

def plot_confusion_matrix(cm,title='Confusion Matrix',cmap=plt.cm.binary):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations,labels,rotation=90)
    plt.yticks(xlocations,labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':
    train_loss_list, valid_loss_list = train()
    y_true, y_pred = test()
    
    # 模型参数规模
    input = torch.randn(32,96,96).to(device)
    flops, params = profile(model, inputs=(input,))
    print('flops:', flops / 1e6, 'params:', params / 1e6)
    
    # 学习曲线
    plt.figure()
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(len(train_loss_list)), train_loss_list)
    plt.plot(range(len(valid_loss_list)), valid_loss_list)
    plt.legend(['Training Loss','Validation Loss'])
    plt.savefig(args.train_curve_name)
    # plt.show()
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
    print(cm_normalized)
    
    
    plt.figure(figsize=(12,8),dpi=120)
    ind_array =  np.arange(len(labels))
    x,y = np.meshgrid(ind_array,ind_array)
    for x_val,y_val in zip(x.flatten(),y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val,y_val,"%0.2f"%(c,),color='red',fontsize=7,va='center',ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks,minor=True)
    plt.gca().set_yticks(tick_marks,minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True,which='minor',linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plot_confusion_matrix(cm_normalized,title='Normalized confusion matrix')
    #show confusion matrix
    plt.savefig('confusion_matrix_' + model_name + '_' + dataset_flag + fake_name + firstBN_name + '.png',format='png')
    # plt.show()
