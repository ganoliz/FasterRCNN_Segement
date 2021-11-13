# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 14:25:18 2021

@author: willy
"""
#https://www.uj5u.com/qita/27937.html
from engine import train_one_epoch,evaluate
import utils_rcnn as utils
import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
from xml.dom.minidom import parse

import torchvision
import SpineDataset as SD
import ShowBbox





root=r"C:/SpineDataset/data/f01"   #train_dataset
root1=r"C:/SpineDataset/data/f01"  #evalueate_dataset
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes=2 #background、spine

dataset=SD.SpineDataset(root,ShowBbox.get_transform(train=True))
dataset_test=SD.SpineDataset(root1,ShowBbox.get_transform(train=False))

#indices = torch.randperm(len(dataset)).tolist()
#dataset = torch.utils.data.Subset(dataset, indices[:-100])
#dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])


data_loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True)
data_loader_test=torch.utils.data.DataLoader(dataset_test,batch_size=1,shuffle=False,collate_fn=utils.collate_fn)

model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,progress=True,num_classes=num_classes,pretrained_backbone=True)

model.to(device)

params=[p for p in model.parameters() if p.requires_grad]

optimizer=torch.optim.SGD(params,lr=0.0003,momentum=0.9,weight_decay=0.0005)

lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=1,T_mult=2)

num_epochs=70
'''
for epoch in range(num_epochs):
    train_one_epoch(model,optimizer,data_loader,device,epoch,print_freq=50)
    
    lr_scheduler.step()
    
    evaluate(model,data_loader_test,device=device)
    
    print('')
    print('=====================================')
    print('')
print("That's it")

torch.save(model, r'C:/SpineDataset/data/f01/fasterrcnn_fin.pkl')
'''
model=torch.load(r'C:/SpineDataset/data/f01/fasterrcnn70.pkl')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
#img_path=os.path.join("C:/SpineDataset/data/f02/image/0021.png")
#img=Image.open(img_path).convert("RGB")
#ShowBbox.showbbox(model,img)

for idx in range(len(dataset_test)):
    img,_=dataset_test[idx]

    ShowBbox.showbbox(model,img,idx,root1)

'''
for i in range(len(dataset_test)):
    img,_=dataset_test[i]
    ShowBbox.showbbox(model,img)
'''
'''
for i,v in dataset_test:
    print('i=',i,'v=',v)
    img,_=dataset_test[0]
    ShowBbox.showbbox(model,i)
'''    
    
    