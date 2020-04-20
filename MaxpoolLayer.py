# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 23:16:10 2020

@author: Dinoel
"""

import numpy as np

class Maxpool:
    
    def iter2x2_regions(self,img):
        (*_, yDim, xDim) = img.shape
        
        newH = yDim//2
        newW = xDim//2
        
        for y in range(newH):
            for x in range(newW):   
                if(img.ndim == 4):
                    region = img[:,:, (y*2):(y*2+2), (x*2):(x*2+2)]
                
                if(img.ndim == 3):
                    region = img[:, (y*2):(y*2+2), (x*2):(x*2+2)]                    

                if(img.ndim == 2):
                    region = img[(y*2):(y*2+2), (x*2):(x*2+2)]  
                
                yield y,x, region
                
    def forward(self, inputVal):
        self.inputVal = inputVal
        
        (*_, depthDim, yDim, xDim) = inputVal.shape
        outShape = (inputVal.shape[:-2]) + (yDim//2, xDim//2)

        out = np.zeros(outShape)
        
        for y, x, region in self.iter2x2_regions(self.inputVal):
            out[:,:,y,x] = np.amax(region, axis = (2,3))

        return out
    
    def backprop(self, dLdOut):
        #(depthDim, yDim, xDim) = dLdOut.shape
        
        dLdInput = np.zeros(self.inputVal.shape)
        
        for i, img in enumerate(self.inputVal):
            for j, arr in enumerate(img):
                for y, x, region in self.iter2x2_regions(arr):
                    mask = np.where(region == np.amax(region),1,0)
                    dLdInput[i,j, (y*2):(y*2+2), (x*2):(x*2+2)] = dLdOut[i,j,y,x] * mask
        
        return dLdInput
    

# =============================================================================
# a = np.arange(600).reshape(2,3,10,10)
# c = Maxpool()
# 
# out = c.forward(a)
# print(out.shape)
# print(out)
# out2 = c.backprop(out/2)
# print(out2.shape)
# print(out2)
# =============================================================================
        
# =============================================================================
# dLdOut = out/100
# 
# dLdIn = c.backProp(dLdOut)
# print(dLdIn)
# =============================================================================

