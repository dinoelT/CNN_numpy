
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

train_img = train_data[:1000, 1:785]
train_label = train_data[:1000, 0]

print("Images shape:",train_img.shape)
print("Labels shape:",train_label.shape)


lr = 0.005

#Init layers
conv1 = Conv3x3(8, imgShape, batchSize=1, depth = 1)
maxpool1 = Maxpool()

#After Maxpool, the output is(8x13x13) =  1352
fc1 = FC(1352, 10, batchSize=1)
softmax = Softmax()

saveCNN_checkpoint = 0

def saveNetwork():   
    path = "SimpleCNN_CheckPoints/simpleCNN"+str(saveCNN_checkpoint)+".h5"

    with h5py.File(path,'w') as f:
        f.create_dataset('Conv1_W', data = conv1.W)
        f.create_dataset('FC1_W', data = fc1.W)
        f.create_dataset('FC1_B', data = fc1.B)
    
     
avg = list()  
errAvg = 2.3
trainEpoch = 1

nrIncorrectExamples = 0
nrCorrectExamples=0

epoch = 0

for a in range(3):
    permutation = np.random.permutation(len(train_img))
    
    train_img = train_img[permutation]
    train_label = train_label[permutation]
    
    for i, img in enumerate(train_img): 
        
        label = int(train_label[i])
        inp = img.reshape((1,1,28,28))/255 - 0.5
        
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
        
        if(i % 100 == 0):
            accuracy = (nrCorrectExamples*100)/(nrCorrectExamples + nrIncorrectExamples)
            print("Processed",i,"Accuracy: ", accuracy,"%")
            
            nrIncorrectExamples = 0
            nrCorrectExamples=0
            
            plt.plot(avg, color = 'blue')
            plt.draw()
            plt.pause(0.01)
                   
        #print(i,np.sum(loss))
        #print(i,errAvg)
        
        dLdOut = crossEntropyLossBackprop(out, label)
        
        out = softmax.backprop(dLdOut)
        
        out = fc1.backprop(out, lr).reshape((8,1,13,13))
        
        out = maxpool1.backprop(out)
        
        out = conv1.backprop(out, lr)
            
    print("Epoch", epoch+1 )
    acc = (nrCorrectExamples*100)/(nrCorrectExamples + nrIncorrectExamples)
    print("Accuracy: ", acc,"%")
    
    saveCNN_checkpoint += 1
    trainEpoch = int(input("Do you want to train for another epoch? No(0) Yes(1): "))
    
    epoch += 1

plt.plot(avg, color = 'blue')
saveNetwork()

print("Training is finished!")

