# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:48:58 2021

@author: Nick
"""

import os 
import torch
import numpy as np

import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt

from tools.data_loading import SpineDataset
from unet_model import UNet
#from utils.utils import plot_img_and_mask

import logging

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5
                ):
    net.eval()
    img=torch.from_numpy(SpineDataset.preprocess(full_img,scale_factor,is_mask=False)) # dataset need modify
    img=img.unsqueeze(0)
    img=img.to(device=device,dtype=torch.float32)
    
    with torch.no_grad():
        output=net(img)
        
        if net.n_classes > 1:
            probs=F.softmax(output,dim=1)[0]
        
        else:
            probs=torch.sigmoid(output)[0]
            
        tf=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1],full_img.size[0])),
            transforms.ToTensor()
            ])
        
        full_mask=tf(probs.cpu()).squeeze()
        
    if net.n_classes == 1:
        return (full_mask>out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0),net.n_classes).permute(2,0,1).numpy()

def mask_to_image(mask:np.ndarray):
    
    if mask.ndim==2:
        return Image.fromarray((mask*255).astype(np.uint8))
    
    elif mask.ndim ==3:
        return Image.fromarray((np.argmax(mask,axis=0)*255/mask.shape[0]).astype(np.uint8))
    




def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()




if __name__=='__main__':
   

    
    net=UNet(n_channels=3,n_classes=1)
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file='C:/SpineDataset/Unet/tools/checkpoint_epoch40.pth'
    scale=0.5
    threshold=0.6
    logging.info(f'Loading model {file}')
    logging.info(f'Using device {device}')
    
    net.to(device=device)
    net.load_state_dict(torch.load(file,map_location=device))
    
    logging.info('Model loaded!')
    
        #for i,filename in enumerate(in_files):
    #logging.info(f'\nPrediction image {filename} ...')
    #print('i and filename',i,filename)
    
    in_files='C:/SpineDataset/data/f03/segment/59_1.png'                       
    out_files="out.jpg"          
    
    img=Image.open(in_files).convert("RGB")
    
    mask=predict_img(net=net,full_img=img,scale_factor=scale,
                         out_threshold=threshold,device=device)
        
    
    out_filename=out_files
    result=mask_to_image(mask)
    result.save('C:/SpineDataset/Unet/predict/'+out_filename)
            

    plot_img_and_mask(img,mask)
            
            
        
        
        
        
    
    
    
    
    
    
    
    
        
        
        