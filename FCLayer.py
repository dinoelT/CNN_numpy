# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:43:07 2020

@author: Dinoel
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:56:16 2020
Fully Conected Layer 
@author: Dinoel
"""
import numpy as np
import sparse

class FC:
    '''
    beta - constant for optimizer updates
    * none -> beta not used
    * momentum -> beta used
    '''
    
    def __init__(self, nrInputs, nrOutputs, optimizer = 'none', beta1 = 0.9, beta2 = 0.999, batchSize = 1):
        
        self.batchSize = batchSize
        #self.optimizer = np.char.lower(optimizer)
        self.optimizer = optimizer
        (self.nrOutputs, self.nrInputs) = (nrOutputs, nrInputs)
        
        self.W = np.random.randn(nrOutputs, nrInputs)/ np.sqrt(nrInputs/2)
        #print("W:\n", self.W)
        self.B = np.zeros(nrOutputs)
        #print("B:\n", self.B)
        
        self.dLdW = np.zeros(self.W.shape)
        self.dLdB = np.zeros(self.B.shape)
        
        #Init optimizer values
        self.VdW = np.zeros(self.W.shape)
        self.VdB = np.zeros(self.B.shape)
  
        #Init optimizer values
        self.SdW = np.zeros(self.W.shape)
        self.SdB = np.zeros(self.B.shape)
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = 1/10**8
        
        self.iter_count = 1
        
        
    def forward(self, inputArray):

        self.lastInput = inputArray
        
        self.out = np.tensordot(self.W, inputArray, axes = (1,1)).T + self.B      

        return self.out   
    
    
    def sparse_dLdW(self, dLdOut):
        inp = self.lastInput
        (batchSize, nrInputs) = inp.shape

        nrCol = self.nrOutputs
        
        row = np.arange(nrInputs * nrCol)
        
        column = np.repeat(np.arange(nrCol), nrInputs)

        rowCol = np.array([row,column])

        data = np.tile(inp, nrCol).flatten()

        thirdDim = np.repeat(np.arange(batchSize), len(row))

        rowCol = np.tile(rowCol, batchSize)
        coords = np.array([thirdDim, *rowCol])
        print(rowCol.shape, coords.shape)
        x = sparse.COO(coords, data)
        
        dLdW = sparse.tensordot(x, dLdOut, axes = ([0,2],[0,1]))
        return dLdW.reshape(self.W.shape)
    
    #The function calculates the derivatives with respect to the dLdOut loss
    def backprop(self, dLdOut, lr):        
        self.dLdW = self.sparse_dLdW(dLdOut)

        self.dLdB = np.sum( np.dot( np.identity(self.nrOutputs) , dLdOut.T), axis = 1) 
        #print("dLdB:\n",self.dLdB)
        dLdX = np.tensordot(self.W, dLdOut, axes = (0,1)).T

        #Update values
            
        if(self.optimizer == 'none'):
            self.W -= self.dLdW * lr
            self.B -= self.dLdB * lr
            
        elif(self.optimizer == 'momentum'):
            self.VdW = self.VdW * self.beta1 + (1 - self.beta1)*self.dLdW
            self.VdB = self.VdB * self.beta1 + (1 - self.beta1)*self.dLdB                
            
            self.W -= self.VdW * lr
            self.B -= self.VdB * lr

        elif(self.optimizer == 'rmsprop'):
            self.SdW = self.SdW * self.beta2 + (1 - self.beta2)*np.power(self.dLdW,2)
            self.SdB = self.SdB * self.beta2 + (1 - self.beta2)*np.power(self.dLdB,2)                   
 
            self.W -= (self.dLdW * lr) / (np.sqrt(self.SdW)+ self.eps)
            self.B -= (self.dLdB * lr) / (np.sqrt(self.SdB)+ self.eps)
        
        elif(self.optimizer == 'adam'):
            self.VdW = self.VdW * self.beta1 + (1 - self.beta1)*self.dLdW
            self.VdB = self.VdB * self.beta1 + (1 - self.beta1)*self.dLdB 

            self.SdW = self.SdW * self.beta2 + (1 - self.beta2)*np.power(self.dLdW,2)
            self.SdB = self.SdB * self.beta2 + (1 - self.beta2)*np.power(self.dLdB,2) 
            
            #Perform Bias Correction
            self.VdW /= (1 - self.beta1**self.iter_count)
            self.VdB /= (1 - self.beta1**self.iter_count)

            self.SdW /= (1 - self.beta2**self.iter_count)
            self.SdB /= (1 - self.beta2**self.iter_count)            
            
            self.W -= (self.VdW * lr) / (np.sqrt(self.SdW) + self.eps)
            self.B -= (self.VdB * lr) / (np.sqrt(self.SdB) + self.eps) 
            
            self.iter_count +=1
            
        return dLdX
    
    def printWeights(self):
        print("\nW:\n", self.W)
        print("\nB:\n", self.B)

    def printInfo(self):
        print("-----------------------------------")
        print("Fully Connected Layer")
        print(self.nrInputs,"inputs, ",self.nrOutputs, " outputs")
        print("Weights shape: ", self.W.shape)
        print("Bias shape: ",self.B.shape)
        print("-----------------------------------")        

# =============================================================================
# #Big Inputs-Output values <<Laptop was blocked>>
# fc = FC(56000,1000, batchSize = 10)
# 
# inp = np.random.randn(560000).reshape(10,56000) 
# out = fc.forward(inp)
# print(out.shape)
# back = fc.backprop(out, 0.005)
# =============================================================================




# =============================================================================
# #for a nn 4x6 inputs and 4x2 outputs
# np.random.seed(1)
# 
# inp = np.random.randn(3,10)
# 
# FC1 = FC(10,6, batchSize = 3)
# FC2 = FC(6,2, batchSize = 3)
# 
# out = FC1.forward(inp)
# out = FC2.forward(out)
# correct = np.zeros(out.shape)
# correct[0] = out[0]/2
# correct[1] = out[1]*2
# correct[2] = out[2]*0.25
# 
# for i in range(1):
#     out = FC1.forward(inp)
#     out = FC2.forward(out)
#     
#     loss = out - correct
#     
#     print(np.sum(loss))
#     
#     out = FC2.backprop(loss, 0.001)
#     #out = FC1.backprop(out, 0.001)    
# =============================================================================
    

# =============================================================================
# excel.write(inp, "INPUT")
# excel.write(weights, "Weights")
# excel.write(bias, "Bias")
# excel.write(out, "OUTPUT")
# 
# excel.save()
# =============================================================================








# =============================================================================
# #Simple example 2 Layers
# 
# np.random.seed(1)
#     
# lr = 0.01
# 
# fc1 = FC(3,5, batchSize = 5) 
# fc2 = FC(5,2, batchSize = 5)  
# # =============================================================================
# # print("Initial Weights")
# # fc1.printWeights()
# # =============================================================================
# 
# 
# inp = np.array([0.7, 0.1, 0.5])
# 
# out = fc1.forward(inp)
# out = fc2.forward(out)
# print("\nOutput:",out) 
# 
# correct = out/2
# 
# #fc1.printInfo()
# 
# for i in range(100):
#     out = fc1.forward(inp)
#     out = fc2.forward(out)
#     
#     loss = out - correct
#     print(np.sum(loss))
#     
#     out = fc2.backprop(loss, lr)
#     out = fc1.backprop(out, lr)
# =============================================================================



# =============================================================================
# #Simple example( 2 inputs, 1 output) to verify the layer
# 
# np.random.seed(1)
# lr = 0.01
# 
# fc1 = FC(2,1, batchSize = 5)  
# # =============================================================================
# # print("Initial Weights")
# # fc1.printWeights()
# # =============================================================================
# 
# 
# inp = np.array([0.7, 0.1])
# 
# out = fc1.forward(inp)
# 
# print("\nOutput:",out) 
# 
# correct = out/2
# 
# for i in range(10):
#     out = fc1.forward(inp)
#     loss = out - correct
#     
#     out = fc1.backprop(loss,lr)
#     #print("finalWeights")
#     #fc1.printWeights()
# =============================================================================

