# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 12:41:34 2021

@author: Nick
"""

import logging

from os import listdir
from os.path import splitext
from pathlib import Path

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class SpineDataset(Dataset):
    def __init__(self,images_dir:str,masks_dir:str,scale:float=1.0,mask_suffix:str=''):
        self.images_dir=Path(images_dir)
        self.masks_dir=Path(masks_dir)
        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        
        self.imgs=list(os.listdir(os.path.join(self.images_dir,"segment")))
        self.masks=list(os.listdir(os.path.join(self.masks_dir,"segment_mask")))
        print("imgs",self.imgs)
        print("masks",self.masks)
        assert 0<scale<=1, 'Scale must be between 0 and 1'
        self.scale=scale
        self.mask_suffix=mask_suffix
        
        
    
    def __len__(self):
        return len(self.imgs)
    
    
    @classmethod
    def preprocess(cls,pil_img,scale,is_mask):
        w,h=pil_img.size
        newW,newH=int(scale*w),int(scale*h)
        assert newW>0 and newH>0,'Scale is too small, resized images would have no pixel'
        pil_img=pil_img.resize((newW,newH),resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray=np.asarray(pil_img)
        #print('img_ndarray.shape',img_ndarray.shape)  img(600,250,3) mask(600,250)
        if img_ndarray.ndim==2 and not is_mask:
            img_ndarray=img_ndarray[np.newaxis, ...]
        elif not is_mask:   
            img_ndarray=img_ndarray.transpose((2,0,1)) #from(H,W,C) To (C,W,H)
        
        #if not is_mask:
        img_ndarray=img_ndarray/255
        
        if is_mask:
            img_ndarray = img_ndarray[:,:,None]
        
        return img_ndarray    
    
    @classmethod
    def load(cls,filename):
        ext=splitext(filename)[1]
        if ext in ['.npz','.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt','.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        
        else:
            return Image.open(filename)
    
    
    def __getitem__(self,idx):
        #name=self.ids[idx]
        #mask_file=list(self.masks_dir.glob(name+self.mask_suffix+'.*'))
        #mask_file=list(self.masks_dir.glob(str(idx)+'.*'))
        #img_file=list(self.images_dir.glob(str(idx)+'.*'))
        
        img_file=os.path.join(self.images_dir,"segment",self.imgs[idx])
        mask_file=os.path.join(self.masks_dir,"segment_mask",self.masks[idx])
        #print("\n mask_file",mask_file)
        #print(len(mask_file))
        # assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        #assert len(img_file) == 1, f'Either no image or multiple images found for the ID {idx}: {img_file}'
        
        #mask=self.load(mask_file[0])
        mask=Image.open(mask_file).convert("I")         #Image.open(img_path).convert("RGB")
        img=Image.open(img_file).convert("RGB")
        #img=self.load(img_file[0])
        
        assert img.size==mask.size,\
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        img=self.preprocess(img,self.scale,is_mask=False)
        mask=self.preprocess(mask,self.scale,is_mask=True)
        '''
        for x in range(len(mask)):
            for y in range(len(mask[x])):
                for z in range(len(mask[x][y])):
                    
                    if mask[x][y][z]!=0:
                        print('x=',y,'y=',x,'result=',mask[x][y][z])
        '''
        #print('mask:',mask)
        #print(len(mask))
        
        return{
            'image':torch.as_tensor(img.copy()).float().contiguous(),
            'mask':torch.as_tensor(mask.copy()).long().contiguous()
            }
        
        