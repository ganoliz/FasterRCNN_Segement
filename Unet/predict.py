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

from Unet.data_loading import SpineDatasetUnet
from Unet.unet_model import UNet
#from utils.utils import plot_img_and_mask

import logging

from Unet.dice_score import dice_coeff


'''
load_path='E:/Nick/Desktop/SpineDataset/Unet/tools/checkpoint_epoch40.pth'

test_img_dir='E:/Nick/Desktop/SpineDataset/data/f01/segment/19_6.png'
pic_save_dir='E:/Nick/Desktop/SpineDataset/'
'''
def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5
                ):
    net.eval()
    img=torch.from_numpy(SpineDatasetUnet.preprocess(full_img,scale_factor,is_mask=False)) # dataset need modify
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
        
        ######
        '''
        print('probs',probs.shape)
        topredict=(full_mask>out_threshold).float()
        mask_true = F.one_hot(mask_true, net.n_classes).float()
        print('topredict',topredict.shape)
        
        device_pre=torch.device('cuda')
        
        topredict=topredict.to(device=device_pre,dtype=torch.float)
        mask_true=mask_true.to(device=device_pre,dtype=torch.float)
        
        dice_score = dice_coeff(topredict, mask_true, reduce_batch_first=False)
        #######
        '''
    if net.n_classes == 1:
        return (full_mask>out_threshold).numpy() #,dice_score
    else:
        return F.one_hot(full_mask.argmax(dim=0),net.n_classes).permute(2,0,1).numpy() #,dice_score

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


def DC_coeff(input,target):
        epsilon=1e-6
        inter=torch.dot(input.ravel(),target.ravel())
        sets_sum = torch.sum(input) + torch.sum(target)             # unite of predict and ground truth
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)

#if __name__=='__main__':
def UnetPredict(model_path,img_dir,pic_save_dir):

    
    #load_path='E:/Nick/Desktop/SpineDataset/Unet/tools/checkpoint_epoch40.pth'

    #test_img_dir='E:/Nick/Desktop/SpineDataset/data/f01/segment/19_6.png'
    #pic_save_dir='E:/Nick/Desktop/SpineDataset/'

    net=UNet(n_channels=3,n_classes=1)
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_model=model_path
    scale=0.5
    threshold=0.6
    logging.info(f'Loading model {load_model}')
    logging.info(f'Using device {device}')
    
    net.to(device=device)
    net.load_state_dict(torch.load(load_model,map_location=device))
    
    logging.info('Model loaded!')
    
        #for i,filename in enumerate(in_files):
    #logging.info(f'\nPrediction image {filename} ...')
    #print('i and filename',i,filename)
    
    Input_img_dir=img_dir  
                   
    img_list=list(os.listdir(os.path.join(Input_img_dir,"segment")))
    print(img_list)
    mask_list=list(os.listdir(os.path.join(Input_img_dir,"segment_mask")))
    print(mask_list)
    
    DC=[]
    
    for i in range(len(img_list)):
        
        img_path=os.path.join(Input_img_dir,"segment",img_list[i])
        mask_path=os.path.join(Input_img_dir,"segment_mask",mask_list[i])
        
        
        
        img=Image.open(img_path).convert("RGB")
        mask_img=Image.open(mask_path).convert("I")
        mask_img=SpineDatasetUnet.preprocess(mask_img,1,is_mask=True)
        
        mask_tensor=torch.as_tensor(mask_img.copy()).long().contiguous()
        
        mask_true = mask_tensor.to(device=device, dtype=torch.float)
        mask_true=mask_true.squeeze(2)
        #print('mask_true',mask_true.shape)
        #mask_true = F.one_hot(mask_true, net.n_classes).permute(0,3, 1, 2).float() #no batch index
        
        
        
        
        
        mask=predict_img(net=net,full_img=img,scale_factor=scale,
                             out_threshold=threshold,device=device)
        #print('mask',mask.shape)  

        mask_temp=torch.from_numpy(mask)
        mask_coeff=mask_temp.to(device=device, dtype=torch.float)
        DC.append(DC_coeff(mask_coeff,mask_true))
        #DC[i]= dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
        
        stri=str(i)
        if i<10:
            stri='0'+str(i)
        elif i>=10:
            stri=str(i)
        
        ouput_filename=stri+'.jpg' 
        
        result=mask_to_image(mask)
        
        result.save(pic_save_dir+ouput_filename)
                
    
        #plot_img_and_mask(img,mask)
            
            
    return DC    
        
        
        
    
    
    
    
    
    
    
    
        
        
        