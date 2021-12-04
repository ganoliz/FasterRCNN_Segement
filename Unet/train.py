# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 17:59:37 2021

@author: Nick
"""

import argparse
import logging

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np

from torch import optim
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm

from Unet.data_loading import SpineDatasetUnet
from Unet.dice_score import dice_loss
from Unet.evaluate import evaluate
from Unet.unet_model import UNet


#dir_img=Path('E:/Nick/Desktop/SpineDataset/data/f02')
#dir_mask=Path('E:/Nick/Desktop/SpineDataset/data/f02')
#save_path=r'E:/Nick/Desktop/SpineDataset/Unet/tools/'



def train_net(dir_img,dir_mask,save_path,net,device,epochs:int=5,batch_size:int =1,
              learning_rate: float =0.001,val_percent: float=0.1,
              save_checkpoint: bool=True,
              img_scale: float=0.5,
              amp: bool=False):
    # 1. Create dataset
    #try:
    dataset=SpineDatasetUnet (dir_img,dir_mask,img_scale)
    
    #dir3='E:/Nick/Desktop/SpineDataset/data/f03'
    test_dataset=SpineDatasetUnet(dir_img,dir_img,img_scale)
    #except(AssertionError,RuntimeError):
    #    dataset=BasicDataset(dir_img,dir_mask,img_scale)
   
 
    #loader_args=dict(batch_size=batch_size,num_workers=0,pin_memory=True)
    
    train_loader=DataLoader(dataset,shuffle=True,batch_size=batch_size)
    val_loader=DataLoader(test_dataset,shuffle=False,drop_last=True,batch_size=batch_size)
    
    #'''
    experiment=wandb.init(project='U-Net',resume='allow',anonymous='must')
    experiment.config.update(dict(epochs=epochs,batch_size=batch_size,
                                  learning_rate=learning_rate,
                                  val_percent=val_percent,save_checkpoint=save_checkpoint,
                                  img_scale=img_scale,amp=amp))
    
    logging.info(f'''Starting training:
                 Epochs:    {epochs}
                 Batch size:{batch_size}
                 Learning rate: {learning_rate}
                 
                 Device:       {device.type}
                 Images scaling: {img_scale}
                 Mixed Precision:{amp}
                 
                 
                 Training size: {len(dataset)}
                 Validation size:{int(len(dataset)*val_percent)}
                 Checkpoints:   {save_checkpoint}
                 ''')

    #'''
    optimizer=optim.RMSprop(net.parameters(),lr=learning_rate,weight_decay=1e-8,momentum=0.9)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',patience=2)
    grad_scaler=torch.cuda.amp.GradScaler(enabled=amp)
    #criterion=nn.CrossEntropyLoss()
    criterion=nn.BCEWithLogitsLoss()        #nn.BCELoss()
    global_step=0
    
    for epoch in range(epochs):
        net.train()
        epoch_loss=0
        
        with tqdm(total=len(dataset),desc=f'Epoch {epoch+1}/{epochs}',unit='img') as pbar:
            for batch in train_loader:
                images=batch['image']
                true_masks=batch['mask']
                
                assert images.shape[1]==net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                
                images=images.to(device=device,dtype=torch.float32)
                true_masks=true_masks.to(device=device,dtype=torch.long) #masks=[1,600,250] (batch_size,H,W)
                
                #print('images',images)
                print('images_shape',images.shape)
                #print('true_masks',true_masks)
                print('true_masks_shape',true_masks.shape)
                masks_permute=true_masks.permute(0,3,1,2).float()

                
                
                
                print('masks_permute',masks_permute.shape)
                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred=net(images)                              #pred=[1,1,600,250]=(batch,Channel,H,W)
                    #print('masks_pred',masks_pred)
                    print('masks_pred',masks_pred.shape)
                    loss=criterion(masks_pred,masks_permute)+dice_loss(F.softmax(masks_pred,dim=1).float(), 
                                masks_permute,multiclass=False)         # we only have 1 type:spine to detect
                         
                         
                          
                        
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                    
                pbar.update(images.shape[0])
                global_step +=1
                epoch_loss +=loss.item()
                
                experiment.log({
                    'train loss':loss.item(),
                    'step':global_step,
                    'epoch':epoch
                    })
               
                # Evaluation round
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
                division_step=len(dataset)                    #  (n_train // (10* batch_size))
                if division_step>0:
                    if division_step %70==0:                                         #global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            
                                #if np.isfinite(value.data.cpu()) and np.isfinite(value.grad.data.cpu()):
                                tag = tag.replace('/', '.')
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                #'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })
                        
        #if save_checkpoint:
        #Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    #torch.save(net.state_dict(), str(save_path+'checkpoint_epoch{}.pth'.format(epoch + 1)))
    torch.save(net.state_dict(),str(save_path))
    logging.info(f'Checkpoint {epoch + 1} saved!')


def UnetTrain(save_RCNNData_dir,num_epoch,save_Unetmodel,UnetSegmentPath):
    #if __name__ == '__main__':
    #args = get_args()
    
    
    #dir_img=Path('E:/Nick/Desktop/SpineDataset/data/f02')
    #dir_mask=Path('E:/Nick/Desktop/SpineDataset/data/f02')
    #save_path=r'E:/Nick/Desktop/SpineDataset/Unet/tools/'
    
    dir_img=save_RCNNData_dir
    dir_mask=save_RCNNData_dir
    save_path=save_Unetmodel
    
    num_epochs=num_epoch
    lr=0.001
    scale=1
    
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=1, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')



    #if args.load: (model_load)
    #    net.load_state_dict(torch.load(args.load, map_location=device))
    #    logging.info(f'Model loaded from {args.load}')


    net.to(device=device)
    #try:
    train_net(dir_img,dir_mask,save_path,net=net,  
                  epochs=num_epochs,               #need load the data
                  batch_size=1,
                  learning_rate=lr,
                  device=device,
                  img_scale=scale,
                  val_percent=0.1,
                  amp=True)
    '''
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
    '''
                    
                
                
                
    
    
    
    
    
    
    
    
    