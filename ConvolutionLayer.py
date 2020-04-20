# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 23:49:23 2019

@author: Dinoel
"""
import numpy as np
from math import sqrt

class Conv3x3:
    
    def __init__(self, nrOfFilters, inputShape, batchSize = 1, lRate=0.001, depth = 3):
        #Initialize filters
        #See Kaiming Initialization
        self.nrOfFilters = nrOfFilters
        self.depth = depth
        self.W = np.random.randn(nrOfFilters, self.depth ,3,3) / sqrt(9*self.depth/2)

        self.dLdW = np.zeros(self.W.shape)
        self.lRate = lRate
        
        self.batchCount = 0
        self.batchSize = batchSize
        self.inputShape = inputShape
        
        self.out_dim_x = inputShape[-1] - 2
        self.out_dim_y = inputShape[-2] - 2
        
        self.out_shape = (inputShape[0], self.nrOfFilters, self.out_dim_y, self.out_dim_x)
        
        self.updateCount = 0
        
    def forward(self, inputValue):
        
        (filterNr, filter_depth, filter_y, filter_x) = self.W.shape
        #print("Filter shape:", self.W.shape)
        if(inputValue.ndim == 3):    
            inputValue = np.expand_dims(inputValue, 0)
        (nrImages, img_depth, img_y, img_x) = inputValue.shape
                
        self.lastInput = inputValue

        out = np.zeros(self.out_shape)
        reg_size = self.W.shape[-1]
        for n, image in enumerate(inputValue):
            for f,filter in enumerate(self.W):
                for y,x, region in self.iter_regions(image, reg_size):
                    out[n,f,y,x] = np.sum(np.multiply(region, filter))
        
        (nr, d, y,x) = self.out_shape            
        out = out.ravel().reshape(nr*d,1,y,x)
        return out
        
        
    
    def iter_regions(self, img, reg_size, fw = 1):
        (*_, yDim, xDim) = img.shape
        
        # OutDim = (imgDim - fDim)/s + 1; s=1, fDim=3 => OutDim = imgDim-2
        
        if(fw == 1):   
            #Output dimensions
            dim_x = self.out_dim_x
            dim_y = self.out_dim_y
        else:
            #Filter dimensions
            dim_x = 3
            dim_y = 3
            
        for y in range(dim_y):
            for x in range(dim_x):
                if(img.ndim == 3):    
                    region = img[:, y:y+reg_size , x:x+reg_size]
                    
                if(img.ndim == 2):
                    region = img[y:y+reg_size , x:x+reg_size]
                yield y,x, region
    
    
    def calc_dLdW(self, dLdOut):
        #Input must be of shape (NrImages, Depth, Y , X), where Depth = 1 or 3      
        temp_dLdW = np.zeros(self.W.shape)

        reg_size = dLdOut.shape[-1]

        for f in range(self.nrOfFilters):    
            #print("Filter",f)
            for i, img in enumerate(self.lastInput):
                for y, x, region in (self.iter_regions(img, reg_size,fw=0)):                    
                    temp_dLdW[f,:,y,x] += np.sum(np.multiply(region, dLdOut[i,f]), axis = (1,2))
        #print("Conv1:\n", temp_dLdW)
        return temp_dLdW
    
    def calc_dLdX(self, dLdOut):
        (nrFilters, filterDepth, filter_y, filter_x) = self.W.shape
        (*_, img_x, img_y) = self.lastInput.shape
        
        out_x = img_x - 2
        out_y = img_y - 2 
        
        out_size = int(out_x * out_y)
        img_size = int(img_x * img_y)
        
        
        dOutdX = np.zeros((nrFilters,filterDepth, out_size, img_size))
        for f, filter in enumerate(self.W):
            i=0
            for y in range(out_y):
                for x in range(out_x):
                    zeroMask = np.zeros((filterDepth, img_y, img_x))
                    zeroMask[:, y:y+3, x:x+3] = filter
                    
                    dOutdX[f,:,i,:] += zeroMask.flatten().reshape(filterDepth, img_y*img_x)
                    i += 1
                    
        dOutdX = np.transpose(dOutdX, (0,1,3,2))
        
        #Concat the last two elements of shape
        sh = dLdOut.shape
        sh = sh[:-2] +(1,1,sh[-2]*sh[-1])
        dLdOut = dLdOut.ravel().reshape(sh)        
        
        dLdX = np.zeros(self.inputShape)

        for n, dLdO in enumerate(dLdOut):
            temp = np.sum( np.multiply(dOutdX, dLdO), axis = (3,0))
            dLdX[n] = temp.ravel().reshape(self.lastInput.shape[-3:])
        
        return dLdX

    def updateWeights(self):
        #print("Update:", self.dLdW * self.lRate)
        self.W -= self.dLdW * self.lRate
        self.dLdW = np.zeros(self.W.shape)
        
        
    def backprop(self, dLdOut):
        
        dLdOut = dLdOut.ravel().reshape(self.out_shape)
        self.dLdW += self.calc_dLdW(dLdOut)
        #print("\nBack\n",self.dLdW)
        #self.W -= self.calc_dLdW(dLdOut) * self.lRate
        
        self.batchCount += 1
        if(self.batchCount == self.batchSize):
            self.updateCount += 1
            #sprint("Uc:", self.updateCount)
            self.updateWeights()
            self.batchCount = 0
        
        return self.calc_dLdX(dLdOut)

# =============================================================================
# #2 layers
# inputShape = (nr,d,y,x) = (2,1,6,6)
# inp = np.arange(nr*d*y*x).reshape(inputShape)/100
# conv1 = Conv3x3(2,inputShape,lRate=0.003,depth=1)
# 
# conv2 = Conv3x3(3,(4,1,4,4), lRate=0.003,depth=1)
# 
# out = conv1.forward(inp)
# correct = conv2.forward(out)/2
# 
# 
# #print("Output:\n",out)s
# 
# for i in range(5000):
#     out = conv1.forward(inp)
#     out = conv2.forward(out)
#     
#     #print("out", out)
#     
#     loss = out - correct   
#     print(i, np.sum(loss))
# 
#     #print("loss:", np.sum(loss**2)/np.size(loss))
#     out = conv2.backprop(loss)
#     out = conv1.backprop(out)
# 
# print("Success")
# =============================================================================

# =============================================================================
# print("1")
# inp = np.arange(16).reshape((1,1,4,4))
# print("\nInput:\n", inp)
# 
# conv = Conv3x3(2,(1,1,4,4), depth=1)
# 
# out = conv.forward(inp)
# 
# print(out)
# print("\nOutput:\n",out)
# 
# conv.backprop(out/2)
# =============================================================================

# =============================================================================
# inputShape = (nr,d,y,x) = (2,1,4,4)
# inp = np.arange(nr*d*y*x).reshape(inputShape)/32
# conv = Conv3x3(2,inputShape,depth=1)
# 
# #print("Input:\n", inp)
# 
# #correct = np.arange(4*2*2).reshape(4,1,2,2)
# 
# correct = conv.forward(inp)/2
# #print("Final", correct)
# 
# 
# 
# #print("Output:\n",out)
# 
# for i in range(10000):
#     out = conv.forward(inp)
#     #print("out", out)
#     
#     loss = out - correct   
# 
# 
#     print("loss:", np.sum(loss**2)/np.size(loss))
#     inpB = conv.backprop(loss)
# =============================================================================

# =============================================================================
# #Test filter
# self.filter[0]=[[[-1,-1,-1],
#                         [2,2,2],
#                         [-1,-1,-1]],
#                            
#                            [[-1,-1,-1],
#                         [2,2,2],
#                         [-1,-1,-1]],
#                            
#                            [[-1,-1,-1],
#                         [2,2,2],
#                         [-1,-1,-1]]]
# self.filter[0] = self.filter[0].transpose(0,2,1)
# =============================================================================
