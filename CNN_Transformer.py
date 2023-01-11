# -*- coding: utf-8 -*-
"""
@Time ： 2022/12/19 16:33
@Auth ： Lu Lin
@File ：CNN_Transformer.py
@IDE ：VSCode
@Motto：AIE(Bug Is Everywhere)
"""
import torch
import argparse
from Construct_dataset import My_dataset
from torch.utils.data import Dataset, DataLoader
from models import CNN, Transformer
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from thop import profile
from sklearn.metrics import confusion_matrix
import numpy as np
torch.manual_seed(1234)


parser = argparse.ArgumentParser(description='CNN and Transformer')
parser.add_argument('--class_number', type=int, default=12, help='数据集类别数')
parser.add_argument('--raw_path', type=str, default='./dataset/raw_dataset', help='真实数据集路径')
parser.add_argument('--generate_path', type=str, default='./dataset/generated_dataset1', help='生成数据集路径')
parser.add_argument('--saved_models_path', type=str, default='./saved_models/CNN_models/myresnet_real.pth', help='模型保存')

args = parser.parse_args()

# 构建数据集
real_dataset = My_dataset(args.raw_path)  # 真实数据集
fake_dataset = My_dataset(args.generate_path, fake_dataset=True)  # 生成数据集
total_dataset = real_dataset + fake_dataset  # 总的数据集

# 加载数据集
dataset = 'fake' 
fake_name = 'TGAN'
firstBN_name = '_BN'
model_name = 'Densenet'

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset=real_dataset,
                                                                           lengths=[0.8, 0.1, 0.1])  # 三个数据集替换
if dataset == 'fake':
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset=fake_dataset, 
                                                                 lengths=[0.8, 0.2])
elif dataset == 'total':
    train_dataset0, valid_dataset0 = torch.utils.data.random_split(dataset=fake_dataset, 
                                                                 lengths=[0.8, 0.2])
    train_dataset = train_dataset0 + train_dataset
    valid_dataset = valid_dataset0 + valid_dataset
    
train_loader, valid_loader, test_loader = DataLoader(train_dataset, batch_size=32, shuffle=True), \
                                          DataLoader(valid_dataset, batch_size=32, shuffle=True), \
                                          DataLoader(test_dataset, batch_size=32)


labelname = ['FLB', 'LLB', 'LOCA', 'LOCAC', 'LR', 'MD', 'RI', 'RW', 'SGATR', 'SGBTR', 'SLBIC', 'SLBOC']
tick_marks =  np.array(range(len(labelname)))+0.5
num_epochs = 250
if model_name == 'CNN':
    model = CNN.resnet(num_classes=12).cuda()
elif model_name == 'Transformer':
    model = Transformer.Transformer(num_classes=12, nhid=96, nlayers=2, sql=96).cuda()
elif model_name == 'Densenet':
    model = CNN.DenseNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=0)

def train(model, train_loader, optimizer, scheduler, criterion):
    model.train(True)
    epoch_loss = 0
    epoch_acc = 0
    for inputs, labels in train_loader:
        inputs = inputs.cuda()   
        labels = labels.long().cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()       
        epoch_loss += loss.item() * labels.size(0)
        epoch_acc += torch.sum(predictions == labels.data)

    # scheduler.step()
    train_loss = epoch_loss / len(train_loader.dataset)
    train_acc = epoch_acc.double() / len(train_loader.dataset)
    return train_loss, train_acc.item()

def valid(model, valid_loader, criterion):
    model.train(False)
    epoch_loss = 0
    epoch_acc = 0
    for inputs, labels in valid_loader:
        inputs = inputs.cuda()
        labels = labels.long().cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)
        epoch_loss += loss.item() * labels.size(0)
        epoch_acc += torch.sum(predictions == labels.data)

    valid_loss = epoch_loss / len(valid_loader.dataset)
    valid_acc = epoch_acc.double() / len(valid_loader.dataset)
    return valid_loss, valid_acc.item()

def test(model, test_loader, criterion):
    model.train(False)
    epoch_loss = 0
    epoch_acc = 0
    y_true = []
    y_pred = []
    for inputs, labels in test_loader:
        inputs = inputs.cuda()
        y_true.extend(labels)
        labels = labels.long().cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)
        y_pred.extend(predictions.to('cpu'))
        epoch_loss += loss.item() * labels.size(0)
        epoch_acc += torch.sum(predictions == labels.data)

    test_loss = epoch_loss / len(test_loader.dataset)
    test_acc = epoch_acc.double() / len(test_loader.dataset)
    print("test: loss: {:.4f}, acc: {:.4f}".format(test_loss, test_acc))
    return np.array(y_true), np.array(y_pred)

def plot_confusion_matrix(cm,title='Confusion Matrix',cmap=plt.cm.binary):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labelname)))
    plt.xticks(xlocations,labelname,rotation=90)
    plt.yticks(xlocations,labelname)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

train_loss_curve = []
train_acc_curve = []
val_loss_curve = []
val_acc_curve = []
best_acc = 0.0
for epoch in range(num_epochs):
    # print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
    # print('*' * 100)
    train_loss, train_acc = train(model, train_loader, optimizer, scheduler, criterion)
    # print("training: loss: {:.4f}, acc: {:.4f}".format(train_loss, train_acc))
    valid_loss, valid_acc = valid(model, valid_loader, criterion)
    # print("validation: loss: {:.4f}, acc: {:.4f}".format(valid_loss, valid_acc))
    if valid_acc > best_acc:
        best_acc = valid_acc
        best_model = model
    # print('best_acc：', best_acc)
    train_loss_curve.append(train_loss)
    train_acc_curve.append(train_acc)
    val_loss_curve.append(valid_loss)
    val_acc_curve.append(valid_acc)
    
torch.save(best_model, args.saved_models_path)
y_true, y_pred = test(best_model, test_loader, criterion)

# 模型参数规模
input = torch.randn(32,96,96).cuda()
flops, params = profile(best_model, inputs=(input,))
print('flops:', flops / 1e6, 'params:', params / 1e6)

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
# print(cm_normalized)

plt.figure(1)
plt.title('LossCurve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1,num_epochs+1), train_loss_curve, label='train_loss')
plt.plot(range(1,num_epochs+1), val_loss_curve, label='val_loss')
plt.legend()
plt.savefig('./figs/loss_curve_' + model_name + firstBN_name + '_' + fake_name + dataset +'.png', format='png')


plt.figure(figsize=(12,8),dpi=120)
ind_array =  np.arange(len(labelname))
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
plt.savefig('./figs/confusion_matrix_' + model_name + firstBN_name + '_' + fake_name + dataset + '.png', format='png')