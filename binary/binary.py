# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 20:04:22 2018

@author: zphsc
"""


def binary(im):
    img=im.convert('L')
    threshold = 148
    a = []
    for i in range(256):
        if i < threshold:
            a.append(0)
        else :
            a.append( 1 )
    imb =img.point(a,'1')
    return imb