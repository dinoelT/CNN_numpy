# -*- coding: utf-8 -*-
"""
    Implementation of the Activation Layer(Non-Liniarity)
"""
import numpy as np

class Activation:
    
    def __init__(self, activation = 'relu'):
        self.activation = activation
    
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
        
    def forward(self, input):
        if(self.activation == 'relu'):
            self.Z =  self.Relu(input)
        elif(self.activation == 'leaky_relu'):
            self.Z = self.Leaky_Relu(input) 
        return self.Z
        
    def backprop(self, dLdOut):
        
        if(self.activation == 'relu'):
            dOutdZ = self.Relu(self.Z, deriv = 'True')
        elif(self.activation == 'leaky_relu'):
            dOutdZ = self.Leaky_Relu(self.Z, deriv = 'True')
            
        dLdZ = np.multiply(dOutdZ, dLdOut)
        
        return dLdZ
        
        
        