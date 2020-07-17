# -*- coding: utf-8 -*-
"""
    Implementation of Maxpool Layer
"""

import numpy as np

class Maxpool:
    
    def iter2x2_regions(self,img):
        #This function returns the 2x2 region from the image and the y,x coords
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
        #This function iterates over the 2x2 regions in the input image and
        #saves the max element into a new output matrix, that it returns
        self.inputVal = inputVal
        
        (*_, depthDim, yDim, xDim) = inputVal.shape
        outShape = (inputVal.shape[:-2]) + (yDim//2, xDim//2)

        out = np.zeros(outShape)
        
        for y, x, region in self.iter2x2_regions(self.inputVal):
            out[:,:,y,x] = np.amax(region, axis = (2,3))

        return out
    
    def backprop(self, dLdOut):
        #This function iterates over the 2x2 image regions
        #   [ 0  1  4]   It saves the max value  [0 0 4]
        #   [ 2  1  1]   other values being      [0 0 0]
        #   [ 1  0  1]   equal to zero           [0 0 0]
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

