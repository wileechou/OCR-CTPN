# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:47:12 2018

@author: zphsc
"""
from PIL import Image
import PIL.ImageOps
import cv2
import numpy as np  
import matplotlib.pyplot as plt
from mser import non_max_suppression_slow
from binary import binary
import os

infile='2.jpg'
img=cv2.imread(infile)
im1=Image.open(infile)
im2=binary(im1)


img_blur = cv2.medianBlur(img,3)#降噪

a=50
b=int(im2.size[0]*im2.size[1]/2)
mser=cv2.MSER_create(_min_area=a,_max_area=b)
gray = cv2.cvtColor(img_blur,cv2.COLOR_BGR2GRAY)
regions, boxes = mser.detectRegions(gray)
boxes1=non_max_suppression_slow(boxes,0.2)
boxes2 = boxes1[np.lexsort(boxes1[:,::-1].T)] #排序
print(boxes2)

###分割字符
(shotname,extension) = os.path.splitext(infile);
pth=shotname
os.mkdir(pth)  

i=0
for box in boxes2:
    x,y,w,h = box
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0, 0),2)#绘图框选
    
    name="im_parts"+str(i)
    box1=(x,0,x+w,im2.size[1]) 
    part=im2.crop(box1)
    part.save(pth+"/"+name+".jpg") 
    i=i+1
    
plt.imshow(img,'brg')
plt.show()
