
import h5py
import numpy as np
from ConvolutionLayer import Conv3x3
from MaxpoolLayer import Maxpool
from FCLayer import FC
from SoftmaxLayer import Softmax
import matplotlib.pyplot as plt



def crossEntropyLossBackprop(input , correctLabel):
    loss = np.zeros(input.shape)
    row = np.arange(len(input))
    loss[row, correctLabel] = -1/input[row, correctLabel]
    return loss

def crossEntropyLossForward(out , correctLabel):
    row = np.arange(len(out))
    loss = -np.log(out[row, correctLabel])
    #print(loss)
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




with h5py.File('mnist_dataset.h5','r') as f:
    ls = list(f.keys())
    print(ls)
    train_data = np.array(f.get('train_data')) 
    #test_data = np.array(f.get('test_data'))  

train_img = train_data[:, 1:785]
train_label = train_data[:, 0].astype(int)

# =============================================================================
# test_img = test_data[:, 1:785]
# test_label = test_data[:, 0].astype(int)
# =============================================================================


batchSize = 20

imgShape = (batchSize,1,28,28)

#reshape, squish between 1 and 0 and center
train_img = (train_img.reshape((-1,batchSize,1,28,28)) / 255) - 0.5
train_label = train_label.reshape(-1,batchSize)

# =============================================================================
# test_img = (test_img.reshape((-1,batchSize,1,28,28)) / 255) - 0.5
# test_label = test_label.reshape(-1,batchSize)
# =============================================================================

print("Images shape:",train_img.shape)
print("Labels shape:",train_label.shape)


lr = 0.0005



#Init layers
conv1 = Conv3x3(8, imgShape, batchSize=1, depth = 1,optimizer = 'rmsprop')
#Output (n_filt * batchSize,1,26,26)
maxpool1 = Maxpool()
#Output (n_filt * batchSize,1,13,13)
#After Maxpool, the output is(8x13x13) =  1352
fc1 = FC(1352, 10, batchSize=batchSize,optimizer = 'rmsprop')
softmax = Softmax()

saveCNN_checkpoint = 0

def saveNetwork(fname = 'simpleCNN'):   
    global saveCNN_checkpoint
    path = "SimpleCNN_CheckPoints/"+fname+str(saveCNN_checkpoint)+".h5"

    with h5py.File(path,'w') as f:
        f.create_dataset('Conv1_W', data = conv1.W)
        f.create_dataset('FC1_W', data = fc1.W)
        f.create_dataset('FC1_B', data = fc1.B)
 
    saveCNN_checkpoint += 1
     
avg = list()  
errAvg = 110

nrTotalExamples = 0
nrCorrectExamples=0


# =============================================================================
# fig, ax = plt.subplots(1,2)
# plt.show(block = False)
# 
# =============================================================================
def cnn_forward(img, label):
    out = conv1.forward(img)
    
    out = maxpool1.forward(out).reshape(batchSize,-1)
    
    out = fc1.forward(out)
    
    out = softmax.forward(out)
    
    predicted = np.argmax(out,axis = 1)
    
    acc = batchSize - np.count_nonzero(predicted - label)
        
    loss = crossEntropyLossForward(out, label)
    
    return acc, loss, out


def cnn_backprop(dLdOut, lr):
    out = softmax.backprop(dLdOut)

    out = fc1.backprop(out, lr).reshape((8 * batchSize,1,13,13))

    out = maxpool1.backprop(out)
    
    out = conv1.backprop(out, lr)
    
train_processed = 0
test_processed = 0

train_next_ChkPoint = 1000

IsTraining = 1

trainOn = 1

while(trainOn == 1):
    if(train_processed == 1000):
        break
    if(IsTraining == 1):
        #Training

        label = train_label[train_processed]

        acc, loss, out = cnn_forward(train_img[train_processed], label)
        
        errAvg = 0.9 * errAvg + 0.1 * np.sum(loss)
        print("loss:",errAvg)
        nrCorrectExamples += acc
        nrTotalExamples += batchSize
        
# =============================================================================
#         if(i % 5 == 0):
#             avg.append(errAvg)
#             ax[0].plot(errAvg)
#     
#             fig.canvas.draw()
# =============================================================================
        
        if(train_processed % 5 == 0): 
            accuracy = (nrCorrectExamples*100)/nrTotalExamples
            print("Processed",train_processed * batchSize,"Accuracy: ", accuracy,"%")
                 
            nrTotalExamples = 0
            nrCorrectExamples=0
              
        dLdOut = crossEntropyLossBackprop(out, label)
        
        cnn_backprop(dLdOut, lr)
        
        train_processed += 1
        
        if(train_processed == train_next_ChkPoint):
            doTest = int(input("Do you want to test? No(0) Yes(1): "))
            
            if(doTest == 1):
                nrTestIter = int(input("Number of test iterations:"))
                IsTraining = 0
                
                #Reset count variables
                nrTotalExamples = 0
                nrCorrectExamples = 0
            else:
                train_next_ChkPoint = int(input("Next checkpoint:"))
    else:
        for i in range(nrTestIter):
            #Testing
            label = int(test_label[train_processed])
            
            acc, loss, out = cnn_forward(train_img[train_processed], label)
            
            errAvg = 0.9 * errAvg + 0.1 * np.sum(loss)
            
            nrCorrectExamples += acc
            nrTotalExamples += batchSize
        
        accuracy = (nrCorrectExamples*100)/nrTotalExamples
        print("Processed",i,"Accuracy: ", accuracy,"%")
        
        nrTotalExamples = 0
        nrCorrectExamples = 0
        
        IsTraining = 1
        
        trainOn = int(input("Do you want to continue training? No(0) Yes(1):"))
        
        doSave = int(input("Do you want to save a checkpoint? No(0) Yes(1):"))
        
        if(doSave == 1):
           saveNetwork() 
# =============================================================================
# print("Epoch", epoch+1 )
# acc = (nrCorrectExamples*100)/(nrCorrectExamples + nrIncorrectExamples)
# print("Accuracy: ", acc,"%")
# 
# saveCNN_checkpoint += 1
# trainEpoch = int(input("Do you want to train for another epoch? No(0) Yes(1): "))
# 
# epoch += 1
# =============================================================================

#plt.plot(avg, color = 'blue')
saveNetwork("FinalCNN")

print("Training is finished!")


