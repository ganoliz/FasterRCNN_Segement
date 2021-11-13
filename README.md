# FasterRCNN_Segement




Practice Faster RCNN and Unet implement using Pytorch.

I use  and  modify code from   the website https://www.cnblogs.com/wildgoose/p/12905004.html  and  https://github.com/milesial/Pytorch-UNet/tree/master/unet

FasterRCNN need torchvision reference tools  at https://github.com/pytorch/vision/tree/main/references/detection

Files Directory need modify manually at lots of .py code.

There are lots of problem need solve. As you can see train_example(FasterRCNN) batch size is 1 and CAN'T be 16,32. 

Because  bounding box number is different in lots of Image, the Misaligned of labels will cause torch.stack error .



