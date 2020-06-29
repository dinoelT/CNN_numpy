import numpy as np
import sparse

class Conv:
    
    def __init__(self, nrOfFilters, inputShape, batchSize = 1,optimizer = 'none',beta1 = 0.9, beta2 = 0.999, depth = 3, filterSize = 3,stride=1,isFirstLayer = 1):

        self.optimizer = np.char.lower(optimizer)
        self.nrOfFilters = nrOfFilters
        self.depth = depth
        
        self.filtS = filterSize
        #Initialize filters
        #See Kaiming Initialization        
        self.W = np.random.randn(nrOfFilters, self.depth, self.filtS, self.filtS) 
        self.W /= np.sqrt((filterSize**2)*self.depth/2)

        self.dLdW = np.zeros(self.W.shape)
        
        self.batchCount = 0
        self.batchSize = batchSize
        self.inputShape = inputShape
        
        self.stride = stride
        
        self.out_dim_x = (inputShape[-1] - self.filtS)//self.stride + 1
        self.out_dim_y = (inputShape[-2] - self.filtS)//self.stride + 1
        
        self.out_shape = (inputShape[0], self.nrOfFilters, self.out_dim_y, self.out_dim_x)

        self.VdW = np.zeros(self.W.shape)
        self.SdW = np.zeros(self.W.shape)
        
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.eps = 1/10**8
        
        self.iter_count = 1
        
        self.updateCount = 0
        
        self.isFirstLayer = isFirstLayer
        
    def forward(self, inputValue):
        
        (filterNr, filter_depth, filter_y, filter_x) = self.W.shape
        #print("Filter shape:", self.W.shape)
        if(inputValue.ndim == 3):    
            inputValue = np.expand_dims(inputValue, 0)
        (nrImages, img_depth, img_y, img_x) = inputValue.shape
                
        self.lastInput = inputValue

        out = np.zeros(self.out_shape)
        reg_size = self.W.shape[-1]
        for n, image in enumerate(inputValue):
            for f,filter in enumerate(self.W):
                if(self.stride != 1):
                    for y,x, region in self.iter_regions_stride(image,reg_size):
                        out[n,f,y,x] = np.sum(np.multiply(region, filter))                    
                else:
                    for y,x, region in self.iter_regions(image, reg_size):
                        out[n,f,y,x] = np.sum(np.multiply(region, filter))
        
        (nr, d, y,x) = self.out_shape  
          
        out = out.ravel().reshape(nr*d,1,y,x)
        return out
        
    def iter_regions_stride(self, img,reg_size):
        (*_, yDim, xDim) = img.shape
        
        # OutDim = (imgDim - fDim)/s + 1; s=1, fDim=3 => OutDim = imgDim-2
        
        
        #Output dimensions
        dim_x = self.out_dim_x
        dim_y = self.out_dim_y
        
        for y in range(dim_y):
            i = int(y * self.stride)
            for x in range(dim_x):
                j = int(x * self.stride)
                region = img[:,i:i+reg_size, j:j+reg_size]
                
                yield y,x, region

    
    def iter_regions(self, img, reg_size, fw = 1):
        (*_, yDim, xDim) = img.shape
        
        # OutDim = (imgDim - fDim)/s + 1; s=1, fDim=3 => OutDim = imgDim-2
        
        if(fw == 1):   
            #Output dimensions
            dim_x = self.out_dim_x
            dim_y = self.out_dim_y
        else:
            #Filter dimensions
            dim_x = self.filtS
            dim_y = self.filtS
            
        for y in range(dim_y):
            for x in range(dim_x):
                if(img.ndim == 3):   
                    region = img[:, y:y+reg_size , x:x+reg_size]
                    
                if(img.ndim == 2):
                    region = img[y:y+reg_size , x:x+reg_size]
                yield y,x, region
    
    def calc_dLdW_stride(self, dLdOut):
        temp_dLdW = np.zeros(self.W.shape)
        
        (filt_y, filt_x) = self.W.shape[-2:]
        (input_y, input_x) = self.lastInput.shape[-2:]
        
        x_range = np.arange(0,input_x - filt_x +1, self.stride)
        y_range = np.arange(0,input_y - filt_y +1, self.stride)
        #print(x_range, y_range)
        for f in range(self.nrOfFilters):    
            for i, img in enumerate(self.lastInput):
                #print(img.shape)
                for y in range(filt_y):
                    for x in range(filt_x):
                        #print((img[:,y_range+y][:,:,x_range+x]))
                        temp_dLdW[f,:,y,x] += np.sum(np.multiply(img[:,y_range+y][:,:,x_range+x], dLdOut[i,f]), axis = (1,2))
        return temp_dLdW
        
    def calc_dLdW(self, dLdOut):
     
        temp_dLdW = np.zeros(self.W.shape)

        reg_size = dLdOut.shape[-1]
        
        for f in range(self.nrOfFilters):    
            for i, img in enumerate(self.lastInput):
                for y, x, region in (self.iter_regions(img, reg_size,fw=0)):                    
                    temp_dLdW[f,:,y,x] += np.sum(np.multiply(region, dLdOut[i,f]), axis = (1,2))
        return temp_dLdW
    
    
    
    def calc_dOutdX_COO(self):
        (nrFilters, filterDepth, filter_y, filter_x) = self.W.shape
        
        (*_, img_y, img_x) = self.lastInput.shape
        
        #Correct the values. Write properties
        out_x = img_x - self.filtS + 1
        out_y = img_y - self.filtS + 1 
        
        out_size = int(out_x * out_y)
        
        filter_size = filter_y * filter_x
        
        rect = np.zeros((out_size, filter_size))            

        x = 0
        for i in range(out_y):
            for j in range(out_x):
                a = np.zeros((img_y, img_x))
                a[i:i+filter_y, j:j+filter_x] = 1
                rect[x] = np.where(a.flatten())[0]
                x += 1
        
        index = np.arange(len(rect))
        index = np.repeat(index, len(rect[0]))
        
        rect_noVals = np.stack((index, rect.flatten()))
        
        depthIndex = np.arange(filterDepth)
        depthIndex = np.repeat(depthIndex, len(rect_noVals[0]))
        depthIndex = np.expand_dims(depthIndex, axis=0)

        rect_noVals = np.tile(rect_noVals, filterDepth)

        rect_depth = np.concatenate((depthIndex,rect_noVals), axis = 0)
        
        filtIndex = np.arange(nrFilters)
        filtIndex = np.repeat(filtIndex, len(rect_depth[0]))
        filtIndex = np.expand_dims(filtIndex, axis = 0)
        rect_depth = np.tile(rect_depth, nrFilters)
        
        rect_filt = np.concatenate((filtIndex, rect_depth), axis = 0).astype(int)
        
        W = self.W.reshape(nrFilters, filterDepth, filter_y * filter_x)
        W = np.tile(W, out_size).flatten()

        x = sparse.COO(rect_filt, W)
        return x

        
        
    def calc_dLdX(self, dLdOut):                   
        dOutdX = self.calc_dOutdX_COO()
        dOutdX = np.transpose(dOutdX, (0,1,3,2))

        #Concat the last two elements of shape
        sh = dLdOut.shape
        sh = sh[:-2] +(1,1,sh[-2]*sh[-1])
        dLdOut = dLdOut.reshape(sh)        
        
        dLdX = np.zeros(self.inputShape)
        
        for n, dLdO in enumerate(dLdOut):
            temp = np.sum( np.multiply(dOutdX, dLdO), axis = (3,0))
            dLdX[n] = temp.todense().ravel().reshape(self.lastInput.shape[-3:])

        return dLdX
    
        
    def backprop(self, dLdOut, lRate):
        dLdOut = dLdOut.ravel().reshape(self.out_shape)    
        
        if(self.stride != 1):
            self.dLdW = self.calc_dLdW_stride(dLdOut)
        else:
            self.dLdW = self.calc_dLdW(dLdOut)
        
        
        if(self.optimizer == 'none'):
            self.W -= self.dLdW * lRate
            
        elif(self.optimizer == 'momentum'):
            self.VdW = self.VdW * self.beta1 + (1 - self.beta1)*self.dLdW               
            
            self.W -= self.VdW * lRate


        elif(self.optimizer == 'rmsprop'):
            self.VdW = self.VdW * self.beta2 + (1 - self.beta2)*np.power(self.dLdW,2)                   
 
            self.W -= (self.dLdW * lRate) / (np.sqrt(self.VdW) + self.eps)

        elif(self.optimizer == 'adam'):
            self.VdW = self.VdW * self.beta1 + (1 - self.beta1)*self.dLdW

            self.SdW = self.SdW * self.beta2 + (1 - self.beta2)*np.power(self.dLdW,2)
            
            #Perform Bias Correction
            self.VdW /= (1 - self.beta1**self.iter_count)

            self.SdW /= (1 - self.beta2**self.iter_count)          
            
            self.W -= (self.VdW * lRate) / (np.sqrt(self.SdW) + self.eps)  
        
            self.iter_count +=1
            
        if(self.isFirstLayer != 1):
            return self.calc_dLdX(dLdOut)


# =============================================================================
# np.random.seed(0)
# img = np.random.randn(2*3*10*10).reshape(2, 3, 10,10)
# 
# conv1 = Conv(2, img.shape, batchSize=1, depth = 3,optimizer = 'none', filterSize = 3, stride = 1, isFirstLayer = 1)
# conv2 = Conv(2, (4,1,8,8), batchSize=1, depth = 1,optimizer = 'none', filterSize = 3, stride = 1, isFirstLayer = 0)
# 
# out1 = conv1.forward(img)
# out2 = conv2.forward(out1)
# 
# correct = np.random.randn(8,1,6,6)
# #correct = out2/4
# 
# for n in range(1000):
#     out1 = conv1.forward(img)
#     out2 = conv2.forward(out1)
#     
#     loss = out2 - correct
#     print(np.sum(loss))
#     
#     back = conv2.backprop(loss, 0.001)
#     conv1.backprop(back, 0.001)
# =============================================================================


# =============================================================================
# #Just dLdW verification
# np.random.seed(0)
# img = np.random.rand(2*3*10*10).reshape(2, 3, 10,10)
# 
# conv1 = Conv(10, img.shape, batchSize=1, depth = 3,optimizer = 'none', stride = 1, isFirstLayer=1)
# out = conv1.forward(img)
# 
# correct = out/2
# 
# for n in range(2000):
#     out1 = conv1.forward(img)
#     
#     loss = out1 - correct
#     print(np.sum(loss))
#     conv1.backprop(loss, 0.001)
# #result should be: 9.985330091133796e-06
# =============================================================================

# =============================================================================
# np.random.seed(0)
# img = np.random.randn(2*3*11*11).reshape(2, 3, 11,11)
# 
# conv1 = Conv(2, img.shape, batchSize=1, depth = 3,optimizer = 'none', filterSize = 3, stride = 2, isFirstLayer = 1)
# 
# correct = np.random.randn(4,1,5,5)
# 
# for n in range(5000):
#     out1 = conv1.forward(img)
# 
#     loss = out1 - correct
#     print(np.sum(loss))
#     conv1.backprop(loss, 0.0001)
# =============================================================================


# =============================================================================
# np.random.seed(0)
# img = np.random.randn(2*3*10*10).reshape(2, 3, 10,10)
# 
# conv1 = Conv(2, img.shape, batchSize=1, depth = 3,optimizer = 'none', filterSize = 3, stride = 1, isFirstLayer = 1)
# 
# conv2 = Conv(2, (4,1,8,8), batchSize=1, depth = 1,optimizer = 'none', filterSize = 3, stride = 1, isFirstLayer = 0)
# # =============================================================================
# # out1 = conv1.forward(img)
# # #print(out1.shape)
# # out2 = conv2.forward(out1)
# # =============================================================================
# correct = np.random.randn(8,1,6,6)
# 
# for n in range(1000):
#     out1 = conv1.forward(img)
#     #print(out1.shape)
#     out2 = conv2.forward(out1)
#     loss = out2 - correct
#     print(np.sum(loss))
#     
#     back2 = conv2.backprop(loss, 0.001)
#     back1 = conv1.backprop(back2, 0.001)
# =============================================================================
    


#back = conv1.backprop(out1, 0.005)

# =============================================================================
# np.random.seed(0)
# img = np.random.randn(2*3*5*5).reshape(2, 3, 5,5)
# 
# conv1 = Conv(2, img.shape, batchSize=1, depth = 3,optimizer = 'momentum', filterSize = 3, stride = 1, isFirstLayer = 0)
# 
# out1 = conv1.forward(img)
# 
# back = conv1.backprop(out1, 0.005)
# =============================================================================


#import timeit 
#print(timeit.timeit(setup = setup, stmt = code, number = 5))

# =============================================================================
# img = np.random.randn(100).reshape(1,1,10,10)
# print("Input shape:", img.shape)
# print("Input: \n", img)
# conv1 = Conv(1, img.shape, depth=1, filterSize = 4, stride = 2)
# 
# out1 = conv1.forward(img)
# 
# correct = np.random.randn(out1.size).reshape(out1.shape)
# 
# print("Output1:\n", out1)
# for i in range(10000):
#     out1 = conv1.forward(img)
#     
#     loss = out1 - correct
#     print(np.sum(loss))
#     
#     back = conv1.backprop(loss, 0.005)
# =============================================================================


# =============================================================================
# #2 layers, filterSize = 4
# inputShape = (nr,d,y,x) = (2,1,10,10)
# inp = np.arange(nr*d*y*x).reshape(inputShape)/100
# conv1 = Conv3x3(2,inputShape,depth=1, filterSize = 4)
# 
# conv2 = Conv3x3(3,(4,1,7,7),depth=1,filterSize = 4)
# 
# out = conv1.forward(inp)
# out = conv2.forward(out)
# correct = np.random.random(out.shape)
# 
# 
# #print("Output:\n",out)s
# 
# for i in range(3000):
#     out = conv1.forward(inp)
#     out = conv2.forward(out)
#     
#     #print("out", out)
#     
#     loss = out - correct   
#     print(i, np.sum(loss))
# 
#     #print("loss:", np.sum(loss**2)/np.size(loss))
#     out = conv2.backprop(loss,0.0001)
#     out = conv1.backprop(out,0.0001)
# 
# print("Success")
# =============================================================================


# =============================================================================
# inputSize = 9
# filtSize = 3
# stride = 2
# 
# input = np.arange(inputSize**2).reshape(inputSize, inputSize)
# filt = np.arange(filtSize**2).reshape(filtSize, filtSize)
# print("Input:\n", input)
# (y_input, x_input) = input.shape
# (y_filt, x_filt) = filt.shape
# 
# out_y = (y_input - y_filt)/stride + 1 
# out_x = (x_input - x_filt)/stride + 1 
# 
# if((out_y != int(out_y)) or (out_x != int(out_x))):
#     print("Not perfect input/output ratio")
# out_y = int(out_y)
# out_x = int(out_x)
# print("Output shape: ", out_y)
# x_range = np.arange(0,x_input - x_filt +1, stride)
# y_range = np.arange(0,y_input - y_filt +1, stride)
# 
# 
# for y in range(y_filt):
#     for x in range(x_filt):
#         #print(x_range+x)
#         print(input[y_range+y][:,x_range+x])
#         print()
# =============================================================================



# =============================================================================
# #2 layers
# inputShape = (nr,d,y,x) = (2,1,6,6)
# inp = np.arange(nr*d*y*x).reshape(inputShape)/100
# conv1 = Conv3x3(2,inputShape,depth=1,optimizer = 'momentum')
# 
# conv2 = Conv3x3(3,(4,1,4,4),depth=1,optimizer = 'momentum')
# 
# out = conv1.forward(inp)
# correct = conv2.forward(out)/2
# 
# 
# #print("Output:\n",out)s
# 
# for i in range(300):
#     out = conv1.forward(inp)
#     out = conv2.forward(out)
#     
#     #print("out", out)
#     
#     loss = out - correct   
#     print(i, np.sum(loss))
# 
#     #print("loss:", np.sum(loss**2)/np.size(loss))
#     out = conv2.backprop(loss,0.005)
#     out = conv1.backprop(out,0.005)
# 
# print("Success")
# =============================================================================



# =============================================================================
# input = np.arange(16).reshape(1,1,4,4)
# 
# conv = Conv3x3(1, input.shape, depth = 1)
# 
# conv.W = np.arange(9).reshape(1,1,3,3)
# 
# out = conv.forward(input)
# 
# print("Out:\n", out)
# 
# dLdOut = out/2
# 
# back = conv.backprop(dLdOut, 0.1)
# =============================================================================


# =============================================================================
# print("1")
# inp = np.arange(16).reshape((1,1,4,4))
# print("\nInput:\n", inp)
# 
# conv = Conv3x3(2,(1,1,4,4), depth=1)
# 
# out = conv.forward(inp)
# 
# print(out)
# print("\nOutput:\n",out)
# 
# conv.backprop(out/2)
# =============================================================================

# =============================================================================
# inputShape = (nr,d,y,x) = (2,1,4,4)
# inp = np.arange(nr*d*y*x).reshape(inputShape)/32
# conv = Conv3x3(2,inputShape,depth=1)
# 
# #print("Input:\n", inp)
# 
# #correct = np.arange(4*2*2).reshape(4,1,2,2)
# 
# correct = conv.forward(inp)/2
# #print("Final", correct)
# 
# 
# 
# #print("Output:\n",out)
# 
# for i in range(10000):
#     out = conv.forward(inp)
#     #print("out", out)
#     
#     loss = out - correct   
# 
# 
#     print("loss:", np.sum(loss**2)/np.size(loss))
#     inpB = conv.backprop(loss)
# =============================================================================

# =============================================================================
# #Test filter
# self.filter[0]=[[[-1,-1,-1],
#                         [2,2,2],
#                         [-1,-1,-1]],
#                            
#                            [[-1,-1,-1],
#                         [2,2,2],
#                         [-1,-1,-1]],
#                            
#                            [[-1,-1,-1],
#                         [2,2,2],
#                         [-1,-1,-1]]]
# self.filter[0] = self.filter[0].transpose(0,2,1)
# =============================================================================
