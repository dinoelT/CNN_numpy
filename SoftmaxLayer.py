# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 17:52:09 2020

@author: Dinoel
"""
import numpy as np


class Softmax:       
        
    def forward(self, inputArray):
        self.inputArray = np.exp(inputArray)
        
        self.inputSize = inputArray.size
        self.lastTotal = np.sum(self.inputArray)

        self.output = self.inputArray/self.lastTotal
        return self.output
            
    def backprop(self, dLdOut):
        for i, loss in enumerate(dLdOut):
            if(loss == 0):
                continue
            
            S = self.lastTotal

            dLdInput = (-self.inputArray * self.inputArray[i]) / (S**2)
            dLdInput[i] = self.inputArray[i] * (S - self.inputArray[i])/(S**2)
            
            dLdInput = -dLdInput/self.output[i] 
            return dLdInput
        

# =============================================================================
# a = Softmax()
# 
# inputArr = np.array([1,4,3,7,8])
# 
# out = a.forward(inputArr)
# 
# out = np.around(out,2)
# 
# print("Output: ",out)
# 
# correctLabel = 1
# #Calculate CrossEntropy loss
# Loss = np.zeros(inputArr.size)
# Loss[correctLabel] = -1/out[correctLabel]
# print("Loss:",Loss)
# resp = a.backprop(Loss)
# resp = np.around(resp,3)
# print("BackProp:", resp)
# =============================================================================

