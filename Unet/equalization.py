# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 01:19:23 2021

@author: willy
"""

from PIL import Image,ImageOps,ImageEnhance


im1 = Image.open(r'C:/SpineDataset/data/f03/image/0042.png')
im1.show() 
enhancer = ImageEnhance.Sharpness(im1)
#im2 = ImageOps.equalize(im1, mask = None)
  
#im2.show() 
factor=10
enhancer.enhance(factor).show()