# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:25:40 2021

@author: Nick
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

from PIL import Image,ImageOps,ImageEnhance

def segmentBox(model,img,idx,root):
    
    
    if "f02" in root:
        idx=idx+20
    elif "f03" in root:
        if idx>=20:
            idx=idx+20
    
    if idx<9:
            s='000'
    else:
            s='00'
    
    path=root+'/image/'+s+str(idx+1)+'.png'
    path_mask=root+'/label/'+s+str(idx+1)+'.png'
    img_segment=Image.open(path)
    img_mask=Image.open(path_mask)
    
    '''
    path=root+'/image/'+s+str(idx+1)+'.png'
    path_mask=root+'/label/'+s+str(idx+1)+'.png'
    img_segment=Image.open(path)
    img_mask=Image.open(path_mask)
    '''
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    with torch.no_grad():
        #'labels':tensor([1,1],device='cuda:0')
        prediction=model([img.to(device)])
    
    
    
    img=img.permute(1,2,0)
    img=(img*255).byte().data.cpu()
    img=np.array(img)
    
    i=0
    
    path_segment=root+"/segment/"
    path_mask_segment=root+"/segment_mask/"
    sortedbox=[]
    for i in range(prediction[0]['boxes'].cpu().shape[0]):
        xmin=round(prediction[0]['boxes'][i][0].item())
        ymin=round(prediction[0]['boxes'][i][1].item())
        xmax=round(prediction[0]['boxes'][i][2].item())
        ymax=round(prediction[0]['boxes'][i][3].item())
        
        label=prediction[0]['labels'][i].item()
        score=prediction[0]['scores'][i].item()
        
        
        if label==1: # ==1
            if score>0.8:  
                if ymax-ymin<135:
                    #cv.rectangle(img, (xmin,ymin),(xmax,ymax),(255,0,0) , thickness=2)
                    #cv.putText(img, "spine", (xmin,ymin),cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),thickness=2)
                    sortedbox.append([xmin,ymin,xmax,ymax])

    #plt.figure(figsize=(20,15))
    #plt.imshow(img)
    sortedbox=sorted(sortedbox,key= lambda s:s[1])
    
    for i in range(len(sortedbox)):
                segment=img_segment.crop((sortedbox[i][0],sortedbox[i][1],sortedbox[i][2],sortedbox[i][3]))
                
                
                #segment= ImageEnhance.Sharpness(segment)
                #sharpness = 10.0
                #segment= segment.enhance(sharpness)
                #segment=ImageOps.autocontrast(segment)    #add some contrast
                segment=ImageOps.equalize(segment, mask = None)
                #enh_con = ImageEnhance.Contrast(segment)
                #contrast = 1.1
                #segment = enh_con.enhance(contrast)
                
                
                
                index=str(idx+1)
                stri=str(i)
                
                if idx<9:
                    index='0'+str(idx+1)
                elif idx>9:
                    index=str(idx+1)
                
                if i <10:
                    stri='0'+str(i)
                elif i>=10:
                    stri=str(i)
                
                segment.save(path_segment+index+'_'+stri+".png")       #str(idx+1)
                
                mask=img_mask.crop(((sortedbox[i][0],sortedbox[i][1],sortedbox[i][2],sortedbox[i][3])))
                mask_name=index+'_'+stri
                mask.save(path_mask_segment + mask_name+'_mask.png')
        
        
    return sortedbox
    #plt.figure(figsize=(20,15))
    #plt.imshow(img)

def segmentBox2(model,img,root,img_path):       #predict: segementation & showIMG 
    #path=root+'/image/'+s+str(idx+1)+'.png'
    #path_mask=root+'/label/'+s+str(idx+1)+'.png'
    img_segment=Image.open(img_path)
    path_mask=img_path.replace('image','label')
    img_mask=Image.open(path_mask)
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    with torch.no_grad():
        #'labels':tensor([1,1],device='cuda:0')
        prediction=model([img.to(device)])
    
    
    
    img=img.permute(1,2,0)
    img=(img*255).byte().data.cpu()
    img=np.array(img)
    
    i=0
    
    path_segment=root+"/segment/"
    path_mask_segment=root+"/segment_mask/"
    sortedbox=[]
    
    for i in range(prediction[0]['boxes'].cpu().shape[0]):
        xmin=round(prediction[0]['boxes'][i][0].item())
        ymin=round(prediction[0]['boxes'][i][1].item())
        xmax=round(prediction[0]['boxes'][i][2].item())
        ymax=round(prediction[0]['boxes'][i][3].item())
        
        label=prediction[0]['labels'][i].item()
        score=prediction[0]['scores'][i].item()
        
        
        if label==1: # ==1
            if score>0.8:
                if ymax-ymin<135:
                    cv.rectangle(img, (xmin,ymin),(xmax,ymax),(0,0,255) , thickness=2)
                    cv.putText(img, "spine", (xmin,ymin),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),thickness=2)
                    sortedbox.append([xmin,ymin,xmax,ymax])
    #plt.figure(figsize=(20,15))
    #plt.imshow(img)
    print('sortedbox_before:',sortedbox) 
    sortedbox=sorted(sortedbox,key= lambda s:s[1])
    print('sortedbox_after:',sortedbox) 
    for i in range(len(sortedbox)):
                segment=img_segment.crop((sortedbox[i][0],sortedbox[i][1],sortedbox[i][2],sortedbox[i][3]))
                
                #segment= ImageEnhance.Sharpness(segment)
                #sharpness = 10.0
                #segment= segment.enhance(sharpness)
                #segment=ImageOps.autocontrast(segment)
                
                #enh_con = ImageEnhance.Contrast(segment)
                #contrast = 1.1
                #segment = enh_con.enhance(contrast)
                segment=ImageOps.equalize(segment, mask = None)
                
                stri=str(i)
                

                if i <10:
                    stri='0'+str(i)
                elif i>=10:
                    stri=str(i)
                
                segment.save(path_segment+'01_'+stri+".png")
                mask=img_mask.crop(((sortedbox[i][0],sortedbox[i][1],sortedbox[i][2],sortedbox[i][3])))
                mask.save(path_mask_segment+'01_'+stri+'_mask.png')

        
    return sortedbox,img
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    