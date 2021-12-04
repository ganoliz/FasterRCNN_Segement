# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 15:06:50 2021

@author: willy
"""

import torch
import torchvision
import cv2 as cv
from engine import train_one_epoch,evaluate
import utils_rcnn as utils
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

import transforms as T

from PIL import Image

def showbbox(model,img,idx,root):
    
    if root==r"E:/Nick/Desktop/SpineDataset/data/f02":
        idx=idx+20
    elif root==r"E:/Nick/Desktop/SpineDataset/data/f03":
        idx=idx+40
    
    if idx<9:
            s='000'
    else:
            s='00'
    
    path=root+'/image/'+s+str(idx+1)+'.png'
    path_mask=root+'/label/'+s+str(idx+1)+'.png'
    img_segment=Image.open(path)
    img_mask=Image.open(path_mask)
    
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    with torch.no_grad():
        #'labels':tensor([1,1],device='cuda:0')
        prediction=model([img.to(device)])
    
    print(prediction)
    
    img=img.permute(1,2,0)
    img=(img*255).byte().data.cpu()
    img=np.array(img)
    
    i=0
    print(prediction[0]['boxes'].cpu().shape[0])
    
    
    path_segment=root+"/segment/"
    path_mask_segment=root+"/segment_mask/"
    for i in range(prediction[0]['boxes'].cpu().shape[0]):
        xmin=round(prediction[0]['boxes'][i][0].item())
        ymin=round(prediction[0]['boxes'][i][1].item())
        xmax=round(prediction[0]['boxes'][i][2].item())
        ymax=round(prediction[0]['boxes'][i][3].item())
        
        label=prediction[0]['labels'][i].item()
        score=prediction[0]['scores'][i].item()
        if label==1: # ==1
            if score>0.8:        
                cv.rectangle(img, (xmin,ymin),(xmax,ymax),(255,0,0) , thickness=2)
                cv.putText(img, "spine", (xmin,ymin),cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),thickness=2)
                segment=img_segment.crop((xmin,ymin,xmax,ymax))
                segment.save(path_segment+str(idx+1)+'_'+str(i)+".png")
                mask=img_mask.crop((xmin,ymin,xmax,ymax))
                mask.save(path_mask_segment+str(idx+1)+'_mask'+str(i)+".png")
                
    plt.figure(figsize=(20,15))
    plt.imshow(img)

def get_transform(train):
    transforms=[]
    
    transforms=[]
    
    transforms=[] 
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    
    return T.Compose(transforms)
    
    