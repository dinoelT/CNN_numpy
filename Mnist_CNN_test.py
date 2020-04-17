
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:50:09 2020

@author: Dinoel
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:41:07 2020

@author: Dinoel
"""
import h5py
import numpy as np
from ConvolutionLayer import Conv3x3
from MaxpoolLayer import Maxpool
from FCLayer import FC
from SoftmaxLayer import Softmax
#nimport matplotlib.pyplot as plt
#import cv2

def crossEntropyLossBackprop(out, correctLabel):
    dLdOut = np.zeros(out.size)
    dLdOut[correctLabel] = -1/out[correctLabel]
    return dLdOut

def crossEntropyLoss(out, correctLabel):
    loss = np.zeros(out.size)
    loss[correctLabel] = - np.log(out[correctLabel])
    return loss 

def addPadding(img, z):
    (y,x) = img.shape[-2:]
    #print(y,x)
    
    newShape = img.shape[:-2] + (y+z+1, x+z+1)
    temp = np.zeros(newShape)
    
    temp[:,:,z:y+z, z:x+z] = img
    
    return(temp) 

def removePadding(img,z):
    (*_, y,x) = img.shape
    
    return img[:,:, z:y-1, z:x-1]


imgShape = (1,1,28,28)

# =============================================================================
# data_path = "MnistData/"
# train_data = np.loadtxt(data_path + "mnist_train.csv", 
#                         delimiter=",")
# 
# test_data = np.loadtxt(data_path + "mnist_test.csv", 
#                        delimiter=",")
# 
# print("Train data shape:", train_data.shape)
# =============================================================================

with h5py.File('mnist_dataset.h5','r') as f:
    ls = list(f.keys())
    #print(ls)
    train_data = np.array(f.get('train_data'))  
    test_data = np.array(f.get('test_data'))


#Init layers
conv1 = Conv3x3(3, imgShape, batchSize=5,lRate=0.01, depth = 1)
maxpool1 = Maxpool()
conv2 = Conv3x3(3, (3,1,13,13),batchSize=5,lRate=0.01, depth = 1)
maxpool2 = Maxpool()
fc1 = FC(324, 100, 0.01, activation = 'leaky_relu')
fc2 = FC(100, 10, 0.01, activation = 'leaky_relu')
softmax = Softmax()


with h5py.File('savedMnistDigit.h5','r') as f:
    ls = list(f.keys())
    print(ls)

    conv1.W = np.array(f.get('Conv1_W'))
    conv2.W = np.array(f.get('Conv2_W'))
    
    fc1.W = np.array(f.get('FC1_W'))
    fc1.B = np.array(f.get('FC1_B'))
    
    fc2.W = np.array(f.get('FC2_W'))
    fc2.B = np.array(f.get('FC2_B'))
    
correctNr = 0
incorrectNr = 0

for i,img in enumerate(test_data):     
    label = int(img[0])
    inp = img[1:785].reshape((1,1,28,28))/255

    out = conv1.forward(inp)
    
    out = maxpool1.forward(out)

    out = conv2.forward(out)

    out = addPadding(out,0)
    
    out = maxpool2.forward(out).flatten()
    
    out = fc1.forward(out)
    
    out = fc2.forward(out)
    
    out = softmax.forward(out)    
    
    if(np.argmax(out) == label):
        correctNr += 1
    else: incorrectNr += 1
    
    if(i % 25 == 0):
        acc = (correctNr*100)/(correctNr + incorrectNr)
        acc = round(acc,2)
        print("Accuracy: ", acc, "%")
    
    
    if((i==100) or (i==1000)):
        print("Do you want to stop? y/n")
        c = input()
        if(c == 'y'):
            break
    
print("Success")

