# -*- coding: utf-8 -*-
"""
    Implementation of Loss functions, Padding
"""

import numpy as np

def crossEntropyLossBackprop(inputValues , correctLabel):
    loss = np.zeros(inputValues.shape)
    row = np.arange(len(inputValues))
    loss[row, correctLabel] = -1/inputValues[row, correctLabel]
    return loss

def crossEntropyLossForward(out , correctLabel):
    row = np.arange(len(out))
    loss = -np.log(out[row, correctLabel])
    return loss  

def addPadding(img, z):
    (y,x) = img.shape[-2:]
    #print(y,x)
    
    newShape = img.shape[:-2] + (y+z+1, x+z+1)
    temp = np.zeros(newShape)
    
    temp[:,:,z:y+z, z:x+z] = img
    
    return(temp) 

def removePadding(img,z):
    (*_, y,x) = img.shape
    
    return img[:,:, z:y-1, z:x-1]
