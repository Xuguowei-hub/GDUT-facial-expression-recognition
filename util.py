# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 16:04:49 2021

@author: Sonwe
"""
from torchvision.transforms import transforms
import math 
import cv2
import numpy as np
import torch 
import torch.nn as nn
import torchvision
import time
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.utils.data as data
import os
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F

from sklearn.metrics import f1_score, accuracy_score




        
cuda = 1
path_save = '/media/6t/XGW/Conv/face_exp/save/vit/half_balanced_nonaug_vit/'
if not os.path.exists(path_save):
    os.makedirs(path_save)


def acc_class(pred, target):
    
    acc_class1 = sum(pred[torch.where(target==0)].argmax(1) == target[torch.where(target==0)])
    acc_class2 = sum(pred[torch.where(target==1)].argmax(1) == target[torch.where(target==1)])
    acc_class3 = sum(pred[torch.where(target==2)].argmax(1) == target[torch.where(target==2)])
    return acc_class1, acc_class2, acc_class3
    
def Train(net, num_epoch, train_data, test_data, optimizer, loss, schduler=None):
    best_acc = 0.75
    test_acc_log = []
    test_loss_log = []
    train_acc_log = []
    train_loss_log = []
    for epoch in range(num_epoch):
        pred_list = []
        init_time = time.time()
        
        train_acc = 0
        train_loss = 0
        l_s_label = 0
        l_s_domain = 0
        l_t_domain = 0
        test_acc = 0
        test_loss = 0
        train_acc_class1 = 0
        train_acc_class2 = 0
        train_acc_class3 = 0
        net.train()
        i = 0
        for data, label in train_data:

            
            optimizer.zero_grad()
            data = data.to(device=f'cuda:{cuda}')
            label = label.to(device=f'cuda:{cuda}')
            
            pred = net(data)
            l = loss(pred, label)
            acc_class1, acc_class2, acc_class3 = acc_class(pred, label)
            
            train_loss += l
            train_acc += (pred.argmax(1)==label).sum().detach().cpu().numpy()/len(label)
            train_acc_class1 += acc_class1.detach().cpu().numpy()
            train_acc_class2 += acc_class2.detach().cpu().numpy()
            try:
                train_acc_class3 += acc_class3.detach().cpu().numpy()
            except AttributeError:
                train_acc_class3 += 0
            l.backward()
            optimizer.step()
            if schduler:
                schduler.step()
        net.eval()
        with torch.no_grad():
            for data, label in test_data:
                data = data.to(device=f'cuda:{cuda}').float()
                label = label.to(device=f'cuda:{cuda}')
                pred = net(data)
                l = loss(pred, label)
                test_acc += (pred.argmax(1)==label).sum().detach().cpu().numpy()/len(label)
                test_loss += l
        test_acc_log.append(test_acc/len(test_data))
        test_loss_log.append(test_loss.cpu().detach().numpy()/len(test_data))
        train_acc_log.append(train_acc/len(train_data))
        train_loss_log.append(train_loss.cpu().detach().numpy()/len(train_data))
        print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_data):.3f} Valid Loss: {test_loss/len(test_data):.3f} Train Acc: {train_acc/len(train_data):.3f} Valid Acc: {test_acc/len(test_data):.3f} Epoch Time: {(time.time()-init_time):.3f} Acc_1: {train_acc_class1/863:.3f} Acc_2: {train_acc_class2/1097:.3f} Acc_3: {train_acc_class3/63:.3f}')
        
        
        #if test_acc/len(test_data) > best_acc:
        #    best_acc = test_acc/len(test_data)
        new_acc = test_acc/len(test_data)
        now_time = time.time()
        torch.save(net.state_dict(), path_save + str(epoch) +':'+  f'{new_acc:.3f}.pt')
    return np.array([test_acc_log, test_loss_log, train_acc_log, train_loss_log])

class h(nn.Module):
    def __init__(self, ci, co, ks, pool='avg'):
        super().__init__()
        if pool == 'avg':
            self.con = nn.Sequential(
                nn.Conv2d(in_channels=ci, out_channels=co, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm2d(co),
                nn.ReLU(),
                nn.AvgPool2d(2, 2))
        elif pool == 'max':
            self.con = nn.Sequential(
                nn.Conv2d(ci, co, ks,padding=ks // 2),
                nn.BatchNorm2d(co),
                nn.ReLU(),
                nn.MaxPool2d(2, 2))
        else:
            self.con = nn.Sequential(
                nn.Conv2d(ci, co, ks,padding=ks // 2),
                nn.BatchNorm2d(co),
                nn.ReLU(),
                nn.Conv2d(co, co, kernel_size=2,padding=0,stride = 2),
                nn.BatchNorm2d(co),)
            
    def forward(self, x):
        return self.con(x)



class exp_net(nn.Module):
    def __init__(self, classes = 5):
        super(exp_net,self).__init__()
        self.classes = classes
        self.aavg = nn.AdaptiveAvgPool2d(1)
        self.amax = nn.AdaptiveMaxPool2d(1)
        self.h1 = h(3, 32, 3, pool=None)
        self.h2 = h(32, 64, 3, pool=None)
        self.h3 = h(64, 64, 3, pool=None)
        self.h4 = h(64, 64, 3, pool=None)
        self.h5 = h(64, 64, 3, pool=None)
        self.l = nn.Linear(64, 3)
        
        torch.manual_seed(413)
        torch.cuda.manual_seed_all(413)

    def forward(self, x):   
     
        f1 = self.h1(x)
      
        f2 = self.h2(f1)

        f3 = self.h3(f2)
     
        f4 = self.h4(f3)
    
        f5 = self.h5(f4)
    
        f5 = self.aavg(f5)
        f5 = f5.reshape(x.shape[0], -1)
     
        pred = self.l(f5)
        return pred
   


class ImgDataset(data.Dataset): # 
    def __init__(self, x,  y, folder, transform=None):
        self.x = x
        self.y = y
        self.folder = folder
        if y is not None:
            self.y = torch.LongTensor(y) # CrossEntropyLoss
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        #print(self.folder+str(self.x[index])+'.png')
        X = cv2.imread(self.folder+str(self.x[index])+'.png')
        X = cv2.resize(X,(64,64))
        if self.transform is not None:
            X = self.transform(X)
        Y = self.y[index]
        return X, Y

def GetLoader(X, y, batch_size, folder=None, transform=None, shuffle=False):
    X = X
    y = y
    
    train_set = ImgDataset(X, y, folder, transform)
    loader = DataLoader(train_set, batch_size, shuffle=shuffle, pin_memory=True) #
    return loader
            
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
    
def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
                #self.shortcut = LambdaLayer(lambda_fun(x))
                
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=3):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])
	
