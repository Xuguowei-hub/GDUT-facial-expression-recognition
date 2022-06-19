# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 10:46:44 2021

@author: 86156
"""
import os
import numpy as np
from util import *
from torchvision.transforms import transforms
from vit_pytorch import ViT
cuda = 1
data_path = '/media/6t/XGW/Conv/face_exp/'

torch.manual_seed(413)
torch.cuda.manual_seed_all(413)

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomCrop(48),
    #transforms.Resize(64),
    #transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(
        np.array([125.3, 123.0, 113.9]) / 255.0,
        np.array([63.0, 62.1, 66.7]) / 255.0),
    ])

valid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
        np.array([125.3, 123.0, 113.9]) / 255.0,
        np.array([63.0, 62.1, 66.7]) / 255.0),
    ])


train_labels = np.load(data_path+'half_balanced_train_label.npy')
valid_labels = np.load(data_path+'test_lable.npy')
train_list = np.load(data_path+'half_balanced_train_data.npy')
valid_list = np.load(data_path+'test_data.npy')

'''
for i in range(len(train_labels)):
    train_labels[i] = int(train_labels[i])
for i in range(len(valid_labels)):
    valid_labels[i] = int(valid_labels[i])
for i in range(len(train_list)):
    train_list[i] = int(train_list[i])
for i in range(len(valid_list)):
    valid_list[i] = int(valid_list[i])
    
    
np.save('train_lable.npy',train_labels)
np.save('test_lable.npy',valid_labels)
np.save('train_data.npy',train_list)
np.save('test_data.npy',valid_list)
'''

train_loader = GetLoader(X=train_list, y=train_labels, batch_size=256, folder=data_path+'half_balanced_png/', transform=train_transform, shuffle=True)
valid_loader = GetLoader(X=valid_list, y=valid_labels, batch_size=256, folder=data_path+'png/png/', transform=valid_transform, shuffle=False)

#model = ResNet(BasicBlock, [3,3,3], num_classes = 3).to(device=f'cuda:{cuda}')
#model = exp_net().to(device=f'cuda:{cuda}')
model = ViT(
    image_size = 64,
    patch_size = 8,
    num_classes = 3,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
).to(device=f'cuda:{cuda}')
total = get_n_params(model)
print('The number of parameters: ', total)
weight_tensor = torch.from_numpy(np.array([1/1043,1/917,1/37]).astype(np.float32))
#weight_tensor = torch.FloatTensor(weight_tensor)
print(weight_tensor)
loss1 = nn.CrossEntropyLoss().to(device=f'cuda:{cuda}')
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
schduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80, 100], gamma=0.1)
print('Start Training')
Train(model, 120, train_loader, valid_loader, optimizer, loss1)
