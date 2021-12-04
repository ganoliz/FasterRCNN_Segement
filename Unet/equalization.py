# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 01:19:23 2021

@author: willy
"""

from PIL import Image,ImageOps,ImageEnhance


#im1 = Image.open(r'E:/Nick/Desktop/SpineDataset/data/f01/segment/04_05.png')
im1 = Image.open(r'E:/Nick/Desktop/SpineDataset/data/f01/image/0006.png')
#enhancer = ImageEnhance.Sharpness(im1)
im = ImageOps.equalize(im1, mask = None)
  


enh_con = ImageEnhance.Contrast(im1)
contrast = 1.1
im = enh_con.enhance(contrast)

#enh_bri = ImageEnhance.Brightness(im1)
#brightness =1.1
#im = enh_bri.enhance(brightness)

#im = ImageOps.equalize(im, mask = None)

#enh_sha = ImageEnhance.Sharpness(im1)
#sharpness = 5.0
#im = enh_sha.enhance(sharpness)
#im = ImageOps.autocontrast(im)
#im1.show()
im.show()

#im2.show() 
#factor=10
#enhancer.enhance(factor).show()

#im.show() 
#im.save( "E:/Nick/Desktop/SpineDataset/data/predict/0002.png", "PNG" )