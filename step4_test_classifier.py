# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 09:29:49 2018

@author: Suhail
"""

#%%


import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers.convolutional import Convolution2D,MaxPooling2D
import cv2
import matplotlib.pyplot as plt

#%%

homedir = os.getcwd()

num_classes = 4
imageSize = 128


model_name = 'segmented_128'

model_dir = homedir + '/model/' + model_name
weightFile_best = model_dir + '/best.hdf5'

image_path = 'test_normal.jpeg'
segment_path = 'segmented.jpeg'

#%%

def segment(image_path):
    #while True:
    #job = q.get()
    #image_path = path[0]
    #store_path = path[1]
    print("Processing image: %s"%image_path)
    # Read the image
    img = cv2.imread(image_path,0)
    rows, cols = img.shape
    # Defining Ranges
    x = range(rows)
    y = range(cols)
    # Fill the White spots
    for xx in x:
        for yy in y:
            if (img[xx,yy]>245):
                img[xx,yy]=0
    # Apply Otsu's Binary Threshold    
    ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # Opening Operation ( Noise Reduction)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # Dilation Operation
    mask = cv2.dilate(opening,kernel,iterations=7)
    # initialize indice for locationg ROI
    indices = []
    # Loop through pixels to generate indices
    for xx in x:
        for yy in y:
            mm = mask[xx,yy]
            if (mm==0):
                indices.append(xx)
    # Find maximum and minimum region of interest
    minInd = min(indices)
    maxInd = max(indices)
    # Define spacing factor
    spacing = 60
    # Apply spacing factor to region of interest
    minInd = minInd - spacing
    # Round lower boundary
    if(minInd<0): minInd = 0
    # Upper boundary, Round
    maxInd = maxInd + spacing
    if(maxInd>rows): maxInd = rows
    
    # Read another clean instance of the image   
    backup = cv2.imread(image_path,0)
    # Crop the image
    cropped = backup[minInd:maxInd, :]
    # Store the image
    #cv2.imwrite(store_path,cropped)
    print("image segmented.")
    return img, cropped
    
#%%
    
img, segmented = segment(image_path)

resized = cv2.resize(segmented,(imageSize,imageSize))

X = []
X.append(resized)
X = np.asarray(X)
X = X/255

X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

#%%

model = Sequential()

model = Sequential()
model.add(Convolution2D(32, 3, 3 , input_shape=(imageSize,imageSize,1),activation= 'relu' ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation= 'relu' ))
model.add(Dense(num_classes, activation= 'softmax' ))

#%%

model.load_weights(weightFile_best)

y = model.predict_classes(X)
classNo = np.ndarray.tolist(y)

dict = {0:'NORMAL', 1: 'CNV', 2: 'DME', 3:'DRUSEN'}

diseaseClass = dict[classNo[0]]

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,diseaseClass, (150,50),font,2,(200,255,0),5,cv2.LINE_AA)

plt.imshow(img,cmap = 'gray')


