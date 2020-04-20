
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

with h5py.File('mnist_train_2000.h5','r') as f:
    ls = list(f.keys())
    #print(ls)
    train_data = np.array(f.get('train_data'))  

print(train_data.shape)


#Init layers
conv1 = Conv3x3(8, imgShape, batchSize=10,lRate=0.001, depth = 1)
maxpool1 = Maxpool()

#After Maxpool, the output is(8x13x13) =  1352
fc1 = FC(1352, 10, 0.001, batchSize=10,activation = 'leaky_relu')
softmax = Softmax()



saveCNN_checkpoint = 0

def saveNetwork():   
    path = "SimpleCNN_CheckPoints/simpleCNN"+str(saveCNN_checkpoint)+".h5"

    with h5py.File(path,'w') as f:
        f.create_dataset('Conv1_W', data = conv1.W)
        
        f.create_dataset('FC1_W', data = fc1.W)
        f.create_dataset('FC1_B', data = fc1.B)
    

plt.gca().set_ylim(bottom = -0.1)
plt.gca().set_ylim(top = 2.5)
     
avg = list()  
errAvg = 2.3
checkPoint = 2001
trainEpoch = 1

nrIncorrectExamples = 0
nrCorrectExamples=0

epoch = 0

while(trainEpoch == 1):
    for i,img in enumerate(train_data): 
        
        label = int(img[0])
        inp = img[1:785].reshape((1,1,28,28))/255 
        
        out = conv1.forward(inp)
        
        out = maxpool1.forward(out).flatten()
        
        out = fc1.forward(out)
        
        out = softmax.forward(out)
        
        if(label == np.argmax(out)):
            nrCorrectExamples += 1
        else:
            nrIncorrectExamples += 1
            
        loss = crossEntropyLoss(out, label)
    
        errAvg = 0.9 * errAvg + 0.1 * np.sum(loss)
        
        if(i % 5 == 0):
            avg.append(errAvg)
            plt.plot(avg, color = 'blue')
            plt.draw()
            plt.pause(0.01)
          
        #print(i,np.sum(loss))
        print(i,errAvg)
        
        dLdOut = crossEntropyLossBackprop(out, label)
        
        out = softmax.backprop(dLdOut)
        
        out = fc1.backprop(out).reshape((8,1,13,13))
        
        out = maxpool1.backprop(out)
        
        out = conv1.backprop(out)
         
        if(i==checkPoint):
            print(i," examples processed")
            print("Choose the next checkpoint? 0 = Stop")
            checkPoint = int(input())
            if(checkPoint == 0):
                break
            
            
    print("Epoch", epoch )
    accuracy = (nrCorrectExamples*100)/(nrCorrectExamples + nrIncorrectExamples)
    print("Accuracy: ", accuracy,"%")
    saveNetwork()
    saveCNN_checkpoint += 1
    checkPoint = 500
    trainEpoch = int(input("Do you want to train for another epoch? No(0) Yes(1): "))
    epoch += 1
print("Training is finished!")