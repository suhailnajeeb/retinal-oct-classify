# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 10:03:49 2018

@author: Suhail
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import h5py

#%%

homedir = os.getcwd()
imageSize = 128

train = 0

if(train):
   datadir = homedir + '/segmented/train/' 
   h5file = homedir + '/data/segmented_train'+str(imageSize)+'.hdf5'
else:
   datadir = homedir + '/segmented/teset/' 
   h5file = homedir + '/data/segmented_test'+str(imageSize)+'.hdf5'


#%%

def shuffleData(X,y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X_shuffled = X[randomize]
    y_shuffled = y[randomize]
    return X_shuffled,y_shuffled

#%%
def get_data(folder,imageSize):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['NORMAL']:
                label = 0
            elif folderName in ['CNV']:
                label = 1
            elif folderName in ['DME']:
                label = 2
            elif folderName in ['DRUSEN']:
                label = 3
            else:
                label = 4
            for image_filename in tqdm(os.listdir(folder + folderName)):
                img_file = cv2.imread((folder + folderName + '/' + image_filename),cv2.IMREAD_GRAYSCALE)
                if img_file is not None:
                    dim = (imageSize,imageSize)
                    img_file = cv2.resize(img_file,dim,cv2.INTER_CUBIC)
                    img_arr = np.asarray(img_file)
                    img_arr = img_arr
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y

#%%

X,y = get_data(datadir,imageSize)
X,y = shuffleData(X,y)

if(train):
    with h5py.File(h5file, 'w') as f:
        f.create_dataset("X_train", data=X)
        f.create_dataset("y_train", data=y)
else:
    with h5py.File(h5file, 'w') as f:
        f.create_dataset("X_test", data=X)
        f.create_dataset("y_test", data=y)

#%%


    
