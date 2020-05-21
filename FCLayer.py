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

class FC:
    '''
    beta - constant for optimizer updates
    * none -> beta not used
    * momentum -> beta used
    '''
    
    def __init__(self, nrInputs, nrOutputs, optimizer = 'none', beta1 = 0.9, beta2 = 0.01, batchSize = 1):
        
        self.batchSize = batchSize
        self.optimizer = np.char.lower(optimizer)
        
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
        
        self.iter_count = 1
        
        
    def forward(self, inputArray):

        self.lastInput = inputArray
        
        self.out = np.tensordot(self.W, inputArray, axes = (1,1)).T + self.B      

        return self.out   
    
    #The function calculates the derivatives with respect to the dLdOut loss
    def backprop(self, dLdOut, lr):
        #from arrayToExcel import ArrayToExcel
        
        #excel = ArrayToExcel()
        temp_dLdW = np.zeros((self.batchSize, self.W.size, dLdOut[0].size))
        
        step = self.nrInputs
        #print("Last input:\n", self.lastInput)
        for n in range(self.nrOutputs):
             temp_dLdW[:,n*step:(n+1)*step, n] = self.lastInput
        #print("temp_dLdW\n",temp_dLdW)
        #excel.write(dLdOut,"dLdOut")
        #excel.write(temp_dLdW[0],"temp_dLdW")
        #excel.write(temp_dLdW[1])
        self.dLdW = np.tensordot(temp_dLdW, dLdOut, axes = ([0,2],[0,1])).reshape(self.W.shape)
        #print("dLdW:\n",self.dLdW)
        #excel.write(self.dLdW, "dLdW")
        self.dLdB = np.sum( np.dot( np.identity(self.nrOutputs) , dLdOut.T), axis = 1) 
        #excel.write(self.dLdB, "dLdB")
        #print("dLdB:\n",self.dLdB)
        dLdX = np.tensordot(self.W, dLdOut, axes = (0,1)).T
        #excel.save("fc_sm_text.xls")
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
 
            self.W -= (self.dLdW * lr) / np.sqrt(self.SdW)
            self.B -= (self.dLdB * lr) / np.sqrt(self.SdB)
        
        elif(self.optimizer == 'adam'):
            self.VdW = self.VdW * self.beta1 + (1 - self.beta1)*self.dLdW
            self.VdB = self.VdB * self.beta1 + (1 - self.beta1)*self.dLdB 

            self.SdW = self.SdW * self.beta2 + (1 - self.beta2)*np.power(self.dLdW,2)
            self.SdB = self.SdB * self.beta2 + (1 - self.beta2)*np.power(self.dLdB,2) 
            
            #Perform Bias Correction
            self.VdW = self.VdW/(1 - self.beta1**self.iter_count)
            self.VdB = self.VdB/(1 - self.beta1**self.iter_count)

            self.SdW = self.SdW/(1 - self.beta1**self.iter_count)
            self.SdB = self.SdB/(1 - self.beta1**self.iter_count)            
            
            
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
# from arrayToExcel import ArrayToExcel
# 
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
# for i in range(10000):
#     out = FC1.forward(inp)
#     out = FC2.forward(out)
#     
#     loss = out - correct
#     
#     print(np.sum(loss))
#     
#     out = FC2.backprop(loss, 0.001)
#     out = FC1.backprop(out, 0.001)    
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

