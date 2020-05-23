import os
import cv2
import matplotlib.pyplot as plt
import skimage.io
import numpy as np
from numpy import array
import random
from random import randint
from sklearn.preprocessing import normalize


def read(path, file_count):
    window = 28 * 8
    
    filename_list = os.listdir(path)
    
    data_size = file_count * 49
    
    band_first = False
    Xtrain = np.zeros((data_size, window, window, 3))
    Ytrain = np.zeros((data_size, window, window))
    
    
    index=0
    for filename in filename_list:
    
        if not ".tiff" in filename:
            continue
        
        image1 = skimage.io.imread(path + filename,plugin='tifffile') / 255
        
        gt =  cv2.imread(path + "gt/" + filename[:-1],0)
        gt[gt>0] = 1
        
        height = image1.shape[0]
        width = image1.shape[1]
         
        stepx = int(width / window) + 1
        stepy = int(height / window) + 1
        
        for i in range (stepx):
            for j in range(stepy):
                
                coorx = i * window
                coory = j * window
                
                
                if coorx + window > width:
                    coorx = width - window 
                if coory + window > height:
                    coory = height - window
                
            
                image_patch = image1[coory:coory+window,coorx:coorx+window]
                image_label = gt[coory:coory+window,coorx:coorx+window]
        
    #            cv2.imshow("patch", image_patch)
    #            cv2.imshow("patch GT", image_label)
                # print("index: ", index)
    #            cv2.waitKey(0)
                Ytrain[index,:,:] = image_label.copy()
                Xtrain[index,:,:,:] = image_patch.copy()
             
                index +=1
        
    
        
    Xtrain =Xtrain.astype(np.float32) 
    Ytrain =Ytrain.astype(np.uint8) 
    
    if band_first:
        x_train = Xtrain.reshape((len(Xtrain), 3, window, window))
        y_train = Ytrain.reshape((len(Ytrain), 1, window, window))
    else:
        x_train = Xtrain.reshape((len(Xtrain), window, window, 3))
        y_train = Ytrain.reshape((len(Ytrain), 1, window, window))
        
    return x_train, y_train


