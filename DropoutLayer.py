# -*- coding: utf-8 -*-
"""
    Implementation of Dropout layer
"""

import numpy as np
#from FCLayer import FC

class Dropout:

    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
        
    def forward(self, x):
        self.dropMatrix = np.random.rand(*x.shape) < self.keep_prob
        return np.multiply(x, self.dropMatrix) / self.keep_prob

    def backprop(self, dLdOut):
        return np.multiply(dLdOut, self.dropMatrix) / self.keep_prob   
# =============================================================================
#    def forward(self, x, keep_prob):
#        dropMatrix = np.random.rand(*x.shape) < keep_prob
#        
#        return np.multiply(x, dropMatrix) / keep_prob
# =============================================================================


# =============================================================================
# fc = FC(25,3)
# drop = Dropout()
# 
# nrIter = 100000
# 
# meanDifference = np.zeros(nrIter)
#         
# for i in range(nrIter):
#       
#     inp = np.random.randn(25)
#     #print("Input:\n",inp)
#     outDrop = drop.forward(inp, 0.2)
#     
#     #print("Output:\n", outDrop)
#     
#     out1 = fc.forward(inp)
#     #print("Without dropout:\n", out1)
#     mean1 = np.mean(out1)
#     out2 = fc.forward(outDrop)
#     #print("With dropout:\n", out2)
#     mean2 = np.mean(out2)
#     
#     meanDifference[i] = mean1 - mean2
#     
#     
# print(np.mean(meanDifference))
# =============================================================================
