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
import matplotlib.pyplot as plt
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

print(train_data.shape)
print(test_data.shape)

#Init layers
conv1 = Conv3x3(8, imgShape, batchSize=1,lRate=0.005, depth = 1)
maxpool1 = Maxpool()
conv2 = Conv3x3(8, (8,1,13,13),batchSize=1,lRate=0.005, depth = 1)
maxpool2 = Maxpool()
#After Maxpool, the output is(8x8x6x6) =  2304
fc1 = FC(2304, 200, 0.001, activation = 'leaky_relu')
fc2 = FC(200, 10, 0.001, activation = 'leaky_relu')
softmax = Softmax()

saveCNN_checkpoint = 0

def saveNetwork():   
    path = "CNN_Checkpoints/savedMnistDigit"+str(saveCNN_checkpoint)+".h5"
    with h5py.File(path,'w') as f:
        f.create_dataset('Conv1_W', data = conv1.W)
        f.create_dataset('Conv2_W', data = conv2.W)
        
        f.create_dataset('FC1_W', data = fc1.W)
        f.create_dataset('FC1_B', data = fc1.B)
        
        f.create_dataset('FC2_W', data = fc2.W)
        f.create_dataset('FC2_B', data = fc2.B)

with h5py.File('CNN_Checkpoints/savedMnistDigit2_sp.h5','r') as f:
    ls = list(f.keys())
    print(ls)

    conv1.W = np.array(f.get('Conv1_W'))
    conv2.W = np.array(f.get('Conv2_W'))
    
    fc1.W = np.array(f.get('FC1_W'))
    fc1.B = np.array(f.get('FC1_B'))
    
    fc2.W = np.array(f.get('FC2_W'))
    fc2.B = np.array(f.get('FC2_B'))
    
errAvg = 2.2

f= open("error.txt","w+")

avg = list()  
# =============================================================================
# plt.gca().set_ylim(bottom = -0.1)
# plt.gca().set_ylim(top = 2.5)
# =============================================================================

checkPoint = 1000



for i,img in enumerate(train_data, start=1001): 
    if(i % 500 == 0):
        saveNetwork()
        saveCNN_checkpoint += 1
        
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

    loss = crossEntropyLoss(out, label)

    errAvg = 0.99 * errAvg + 0.01 * np.sum(loss)
    
    if(i % 5 == 0):
        f.write("%f\n" % errAvg)
        avg.append(errAvg)
        plt.plot(avg)
        plt.draw()
        plt.pause(0.5)
      
    #print(i,np.sum(loss))
    print(i,errAvg)
    
    dLdOut = crossEntropyLossBackprop(out, label)
    
    out = softmax.backprop(dLdOut)

    out = fc2.backprop(out)
    
    out = fc1.backprop(out).reshape((64,1,6,6))
    
    out = maxpool2.backprop(out)
    
    out = removePadding(out,0)

    out = conv2.backprop(out)
    
    out = maxpool1.backprop(out)
    
    out = conv1.backprop(out)
     
    if(i==checkPoint):
        print(i," examples processed")
        print("Choose the next checkpoint? 0 = Stop")
        checkPoint = int(input())
        if(checkPoint == 0):
            break
        

    
print("Success")

