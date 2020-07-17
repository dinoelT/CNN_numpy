# -*- coding: utf-8 -*-
"""
    Implementation of Softmax layer
"""
import numpy as np

class Softmax:       
        
    def forward(self, inputArray):
        ''' 
        This function calculates the exponential of all inputs and the sum of
        the exponentials. Finally, it returns the ratio of the exponential
        of the inputs and the sum of the exponentials
        '''
        #Calculate the exp for each element
        self.inputArray = np.exp(inputArray)

        #Calculate the Sum of the exp for each row
        self.lastTotal = np.sum(self.inputArray, axis = 1)
        self.output = self.inputArray/self.lastTotal[:, None]

        return self.output
            
    def backprop(self, dLdOut):
        '''
        This function calculates the gradients for each of the outputs
        '''
        (corr_row, corr_col) = np.nonzero(dLdOut)

        # Sum of all exponential terms
        S = self.lastTotal

        # Sum of all exponential terms squared
        S2 = S**2
        #Correct classes
        correct = self.inputArray[corr_row, corr_col]
        # c - correct class. Calculate where k != c
        dLdInput = -self.inputArray.T * correct/S2
        # Calculate where k=c
        dLdInput[corr_col, corr_row] = correct * (S - correct)/S2
        
        dLdInput = - dLdInput * 1/self.output[corr_row, corr_col]
        dLdInput = dLdInput.T

        return dLdInput




# =============================================================================
# np.random.seed(0)
# 
# a = Softmax()
# 
# inputArray = np.array([1,3,5,1,2]).reshape(1,5)
# 
# out = a.forward(inputArray)
# print("Output", out)
# print(np.sum(out))
# 
# correctLabel = np.array([1]).reshape(1,1)
# 
# #Calculate CrossEntropy loss
# Loss = np.zeros(inputArray.shape)
# 
# for i,el in enumerate(correctLabel):
#     Loss[i,el] = -1/out[i,el]
# #print(np.nonzero(Loss))
# 
# resp = a.backprop(Loss)
# print(resp)
# =============================================================================




# =============================================================================
# np.random.seed(0)
# 
# softmax = Softmax()
# 
# inputArray = np.random.randint(1,10,30).reshape(3,10)
# 
# out = softmax.forward(inputArray)
# #print(out)
# correctLabel = np.array([1,3,5])
# 
# l = CrossEntropyLossForward(out, correctLabel)
# print("L:", l)
# 
# #Calculate CrossEntropy loss
# Loss = np.zeros(inputArray.shape)
# 
# for i,el in enumerate(correctLabel):
#     Loss[i,el] = -1/out[i,el]
# print(Loss)
# 
# r = CrossEntropyLossBackprop(out, correctLabel)
# print(r - Loss)
# 
# back = softmax.backprop(r)
# 
# print(back)
# =============================================================================

# =============================================================================
# from arrayToExcel import ArrayToExcel
# 
# np.random.seed(0)
# 
# a = Softmax()
# 
# excel = ArrayToExcel()
# 
# 
# inputArray = np.random.randint(1,10,30).reshape(3,10)
# 
# out = a.forward(inputArray)
# #print(out)
# 
# correctLabel = np.array([1,3,5])
# 
# #Calculate CrossEntropy loss
# Loss = np.zeros(inputArray.shape)
# 
# for i,el in enumerate(correctLabel):
#     Loss[i,el] = -1/out[i,el]
# #print(np.nonzero(Loss))
# 
# resp = a.backprop(Loss)
# 
# excel.write(resp, "backprop")
# 
# excel.save("softmax3.xls")
# =============================================================================

# =============================================================================
# for i in range(3):   
#     out = a.forward(inputArray[i])
#     excel.write(inputArray[i],"Input"+str(i))
#     print("Output: ",out)
#     excel.write(out, "Out"+str(i))
#     correctLabel = 1
#     
#     #Calculate CrossEntropy loss
#     Loss = np.zeros(inputArr.size)
#     Loss[correctLabel] = -1/out[correctLabel]
#     print("Loss:",Loss)
#     resp = a.backprop(Loss)
#     excel.write(resp, "backprop"+str(i))
#     print("BackProp:", resp)
# 
# excel.save("softmax.xls")
# =============================================================================
