# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 10:21:47 2021

@author: willy
"""

import torch
import os
import numpy as np

import cv2 as cv
import matplotlib as plt

from torchvision import datasets,transforms
from PIL import Image

from xml.dom.minidom import parse

import xml.etree.ElementTree as ET


def read_content(xml_file:str): #https://stackoverflow.com/questions/53317592/reading-pascal-voc-annotations-in-python
    tree=ET.parse(xml_file)
    root=tree.getroot()
    #print(root)
    list_with_all_boxes=[]
    label_all_boxes=[]
    for boxes in root.iter('object'): #iter object label
        
        filename=root.find('filename').text
        #print(boxes)
        ymin,xmin,ymax,xmax=None,None,None,None
        
        #print(name)
        ymin=int(boxes.find("bndbox/ymin").text)
        xmin=int(boxes.find("bndbox/xmin").text)
        ymax=int(boxes.find("bndbox/ymax").text)
        xmax=int(boxes.find("bndbox/xmax").text)
                
        list_with_single_boxes=[xmin,ymin,xmax,ymax]
        label=boxes.find('name').text
        
        list_with_all_boxes.append(list_with_single_boxes)
        label_all_boxes.append((label))
        
        
    return label_all_boxes,list_with_all_boxes

class SpineDataset(torch.utils.data.Dataset):
    def __init__(self,root,transforms=None):
        self.root=root
        self.transforms=transforms
        self.imgs=list(os.listdir(os.path.join(root,"image")))
        self.bbox_xml=list(os.listdir(os.path.join(root,"notation")))
    
    def __getitem__(self,idx):
        
        img_path=os.oath.join(self.root,"image",self.imgs[idx])
        bbox_xml_path=os.path.join(self.root,"notation")
        img=Image.open(img_path).convert("RGB")
        bbox_xml_path=bbox_xml_path+"00"+(idx+1)
        
        label,box=read_content(bbox_xml_path)
        box=torch.as_tensor(box,dtype=torch.float32)
        image_id=torch.tensor([idx])
        area=(box[:,3]-boxes[:,1])*(box[:,2]-boxes[:,0])
        iscrowd = torch.zeros((len(box),), dtype=torch.int64)
        
        target={}
        target["boxes"]=box
        target["label"]=label
        target["image_id"]=image_id
        target["area"]=area
        target["iscrowd"]=iscrowd
        
        if self.transforms is not None:
            img,target=self.transforms(img,target)
            
        return img,target
        
    def __len__(self):
        return len(self.imgs)
        #dom=parse(bbox_xml_path) #open .xml file
        


name,boxes=read_content("C:/SpineDataset/data/f01/notation/0001.xml")

    
