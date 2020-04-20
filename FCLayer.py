# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:56:16 2020
Fully Conected Layer 
@author: Dinoel
"""
import numpy as np
import math

class FC:
    
    def __init__(self, nrInputs, nrOutputs, learningRate,batchSize = 1,  activation = 'relu'):
        
        self.batchSize = batchSize
        self.countPasses = 0
        
        (self.nrOutputs, self.nrInputs) = (nrOutputs, nrInputs)
        
        self.W = np.random.randn(nrOutputs, nrInputs)/ math.sqrt(1.1*nrInputs/2)

        self.B = np.random.randn(nrOutputs) / math.sqrt(nrInputs/2)
        
        self.lr = learningRate
        self.activation = activation
        
        self.dLdW = np.zeros(self.W.shape)
        self.dLdB = np.zeros(self.B.shape)
# =============================================================================
#         print("W:\n",self.W)
#         print("B:\n",self.B)
# =============================================================================
               
    def Relu(self, x, deriv = 'False'):
        if(deriv == 'False'):
            res = np.maximum(0,x)
        else:
            res = np.where(x>0, 1, 0)
        return res
    
    def Leaky_Relu(self, x, deriv = 'False'):
        if(deriv == 'False'):    
            return np.maximum(x*0.1, x)
        else:
            return np.where(x>0, 1, 0.1)       
    
    def forward(self, inputArray):
        assert (inputArray.size == self.W[0].size),"FC: Incorrect nr of inputs"

        self.lastInput = inputArray
        
        self.Z = np.dot(self.W, inputArray) + self.B
        #print("Z:", self.Z)
        if(self.activation == 'relu'):
            self.out = self.Relu(self.Z)
        elif(self.activation == 'leaky_relu'):
            self.out = self.Leaky_Relu(self.Z)
        
        #print("Out:\n",self.out)
        return self.out   
    
    def backprop(self, dLdOut, test = '0'):
        
        self.countPasses += 1 
        
        temp_dLdW = np.zeros((self.W.size, dLdOut.size))
        
        step = self.nrInputs

        if(self.activation == 'relu'):
            dOutdZ = self.Relu(self.Z, deriv = 'True')
        elif(self.activation == 'leaky_relu'):
            dOutdZ = self.Leaky_Relu(self.Z, deriv = 'True')
        
        for n in range(dLdOut.size):
            temp_dLdW[n*step:(n+1)*step, n] = self.lastInput
        
        
        dLdZ = np.multiply(dOutdZ, dLdOut)
                
        self.dLdW += np.dot(temp_dLdW, dLdZ).reshape(self.W.shape)
        self.dLdB += np.dot( np.identity(self.nrOutputs) , dLdZ) 
        
        dLdX = np.dot(self.W.T, dLdZ)
        
# =============================================================================
#         print(np.sum(self.dLdW))
#         print(np.sum(self.dLdB))
# =============================================================================
        
        if(test == '1'):
            print("dLdZ:",self.dLdZ)
            print("dLdW:\n",self.dLdW)
            print("dLdX:\n",self.dLdX)
            print("dLdB:", self.dLdB)
        
        #print(self.countPasses," Calculated")
        
        if(self.countPasses == self.batchSize):          
            #Update values
            self.W -= self.dLdW * self.lr
            self.B -= self.dLdB * self.lr
            #print(" Updated")
            self.dLdW = np.zeros(self.W.shape)
            self.dLdB = np.zeros(self.B.shape)
            
            self.countPasses = 0   
        return dLdX
    
    def printWeights(self):
        print("\nW:\n", self.W)
        print("\nB:\n", self.B)

# =============================================================================
# #Simple example 2 Layers
# 
# np.random.seed(1)
#     
# fc1 = FC(3,5, 0.01, batchSize = 5, activation = 'relu') 
# fc2 = FC(5,2, 0.01, batchSize = 5, activation = 'relu')  
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
# for i in range(1000):
#     out = fc1.forward(inp)
#     out = fc2.forward(out)
#     
#     loss = out - correct
#     print(np.sum(loss))
#     
#     out = fc2.backprop(loss)
#     out = fc1.backprop(out)
#     #print("finalWeights")
#     #fc1.printWeights()
# =============================================================================



# =============================================================================
# #Simple example( 2 inputs, 1 output) to verify the layer
# 
# np.random.seed(1)
#     
# fc1 = FC(2,1, 0.01, batchSize = 5, activation = 'relu')  
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
#     out = fc1.backprop(loss)
#     #print("finalWeights")
#     #fc1.printWeights()
# =============================================================================

