# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:56:16 2020
Fully Conected Layer 
@author: Dinoel
"""
import numpy as np

class FC:
    
    def __init__(self, nrInputs, nrOutputs, batchSize = 1):
        
        self.batchSize = batchSize
        self.countPasses = 0
        
        (self.nrOutputs, self.nrInputs) = (nrOutputs, nrInputs)
        
        self.W = np.random.randn(nrOutputs, nrInputs)/ np.sqrt(nrInputs/2)
        #print("W:\n", self.W)
        self.B = np.zeros(nrOutputs)
        #print("B:\n", self.B)
        
        self.dLdW = np.zeros(self.W.shape)
        self.dLdB = np.zeros(self.B.shape)
        
    
    def forward(self, inputArray):
        assert (inputArray.size == self.W[0].size),"FC: Incorrect nr of inputs"

        self.lastInput = inputArray
        
        self.out = np.dot(self.W, inputArray) + self.B      
        #print("Out:\n",self.out)
        return self.out   
    
    #The function calculates the derivatives with respect to the dLdOut loss
    def backprop(self, dLdOut, lr, test = '0'):
        
        self.countPasses += 1 
        
        temp_dLdW = np.zeros((self.W.size, dLdOut.size))
        
        step = self.nrInputs
        
        for n in range(dLdOut.size):
            temp_dLdW[n*step:(n+1)*step, n] = self.lastInput
                
        self.dLdW += np.dot(temp_dLdW, dLdOut).reshape(self.W.shape)
        self.dLdB += np.dot( np.identity(self.nrOutputs) , dLdOut) 
        
        dLdX = np.dot(self.W.T, dLdOut)
        
        if(test == '1'):
            print("dLdZ:",self.dLdZ)
            print("dLdW:\n",self.dLdW)
            print("dLdX:\n",self.dLdX)
            print("dLdB:", self.dLdB)
        
        #print(self.countPasses," Calculated")
        
        if(self.countPasses == self.batchSize):          
            #Update values
            self.W -= self.dLdW * lr
            self.B -= self.dLdB * lr
            
            self.dLdW = np.zeros(self.W.shape)
            self.dLdB = np.zeros(self.B.shape)

            self.countPasses = 0   
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

