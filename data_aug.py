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



aug_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomCrop(110),
    ])
        
cuda = 0
path_read = '/media/6t/XGW/Conv/face_exp/png/png/'
path_save = '/media/6t/XGW/Conv/face_exp/half_balanced_png/'

train_label = np.load('train_lable.npy')
train_data = np.load('train_data.npy')

if not os.path.exists(path_save):
    os.makedirs(path_save)

count = 6000
class_0 = 0
class_1 = 0
half_balanced_train_data = np.empty(0).astype(np.int) 
half_balanced_train_label = np.empty(0).astype(np.int) 
for i in range(len(train_label)):
    if train_label[i] == 0:
        class_0 += 1
        if (class_0 % 2)!=0:
            continue 
    if train_label[i] == 1:
        class_1 += 1
        if (class_1 % 2)!=0:
            continue 
            
    img = cv2.imread(path_read+str(train_data[i])+'.png')
    cv2.imwrite(path_save + str(train_data[i]) +'.png',img)

    half_balanced_train_data = np.hstack((half_balanced_train_data,train_data[i]))
    half_balanced_train_label = np.hstack((half_balanced_train_label,train_label[i]))
    if train_label[i] != 2:
        continue
    #img = cv2.imread(path_read+str(train_data[i])+'.png')
    img = cv2.resize(img,(120,120))
    for k in range(15):
        output = np.array(aug_transform(img))   
        cv2.imwrite(path_save + str(count) +'.png',output)
        half_balanced_train_data = np.hstack((half_balanced_train_data,count))
        half_balanced_train_label = np.hstack((half_balanced_train_label,2))
        count += 1
    
np.save('half_balanced_train_data.npy',half_balanced_train_data)   
np.save('half_balanced_train_label.npy',half_balanced_train_label)



    
      


	
