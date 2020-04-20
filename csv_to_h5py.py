# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:30:15 2020

@author: Dinoel
"""

import numpy as np
import h5py

# Load data from csv to numpy array

data_path = "MnistData/"
train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")

test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",")

print("Train data:", train_data.shape)
print("Test data:", test_data.shape)


# Save data as h5py file
with h5py.File("mnist_train_2000.h5",'w') as f:
    f.create_dataset('train_data', data = train_data[:2000])
    
# Load the data from h5py file     
with h5py.File("mnist_train_2000.h5",'r') as f:
    read_data = np.array(f.get('train_data'))
    print("Read data:",read_data.shape)
    