# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import confusion_matrix
import skimage.io
import pickle
import matplotlib.pyplot as plt
import cv2

def find_metrics(true_values, estimated):
    
    confusion = confusion_matrix(true_values.ravel(),estimated.ravel())
    if confusion.shape[0] == 1 and confusion.shape[1] == 1:
        return 1,1,1
    
    tn = confusion[0,0]
    fn = confusion[1,0]
    tp = confusion[1,1]
    fp = confusion[0,1]
    
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1 = (2*precision*recall) / (precision+recall)
    
    return precision,recall,f1


def convert_result(result):

    result[result>=0.5] = 1
    result[result<0.5] = 0
    return result


def read_tif(file):
    
    img = skimage.io.imread(file,plugin='tifffile')
  
    x,y,d = img.shape
    imRGB = np.ones((x , y, 3)) 
 
    imRGB[:,:,0] = img[:,:,2]
    imRGB[:,:,1] = img[:,:,1]
    imRGB[:,:,2] = img[:,:,0]
#    print( np.max(imRGB)
    image = np.uint8(imRGB/8) / 255
    
    nir =  np.uint8(img[:,:,3]/8) / 255
    
#    imFull = np.zeros((x , y, 4)) 
#    imFull[:,:,0:3] = image
#    imFull[:,:,3] = np.uint8(img[:,:,3]/8)
    
    return image,nir

def normalize_img(xxx):
    
    mean = np.mean(xxx)  
    std = np.std(xxx)  
    xxx -= mean
    xxx /= std
    
    return xxx

def save_history(hist, filename):
    
    with open(filename, 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)


def read_history(filename):
       
    hist = pickle.load( open( filename, "rb" ) )
    return hist
        
def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    
def combine (res1, resYolo):
    
    n = 5
    kernel = np.ones((n,n),np.uint8)
    resYolo = cv2.morphologyEx(resYolo, cv2.MORPH_CLOSE, kernel)
    
    im2, contours1, hierarchy = cv2.findContours(res1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im2, contours2, hierarchy = cv2.findContours(resYolo,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    blank = np.zeros(res1.shape, np.uint8)
    
    for cnt in contours1:
        
        (x,y),radius = cv2.minEnclosingCircle(cnt)
#        area1 = cv2.contourArea(cnt)
#        temp1 = cv2.drawContours(blank.copy(), cnt, 0, 1)

        for c in contours2:
            
            area2 = cv2.contourArea(c)
            if area2<10:
                continue
            
            temp2 =blank.copy()
            cv2.fillPoly(temp2, pts =[c], color=(255))
#            cv2.imshow("temp2",cv2.resize(temp2,(448,448)))
#            cv2.waitKey(0)
#            print("temp2 shape: ", temp2.shape)
            temp = cv2.bitwise_and(res1,temp2)
            intersect = len(temp[temp>0]) 
            ratio = intersect/area2
#            print("intersect:", intersect, "  area2:",area2, "ratio: ",ratio)
            if ratio > 0.2:
#                print("del")
                cv2.fillPoly(resYolo, pts =[c], color=(0))
                
#            dist = cv2.pointPolygonTest(c,(x,y),True)
#            if dist >= 0:
#                cv2.fillPoly(resYolo, pts =[c], color=(0))
            
        
    
    resCombined = cv2.bitwise_or(resYolo,res1)
    
    
    resCombined = np.uint8(resCombined / 255)
    return resCombined
        
        
        