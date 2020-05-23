# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 10:36:33 2018

@author: IbrahimD
"""

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

import glob
import sys
import matplotlib.pyplot as plt
import numpy as np

import unet, Inception, unetV2
import utils
import read_data

from keras.models import load_model  
from keras.preprocessing.image import ImageDataGenerator

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

smooth = 1.


window = 28 * 8


path = 'Massachusetts/train/'
x_train, y_train = read_data.read(path, 110)
print("len train ", len(x_train))

path = 'Massachusetts/validation/'
x_valid, y_valid = read_data.read(path, 4)
print("len valid ", len(x_valid))


# model = unet.get_unet()
# model = Inception.get_unet()
model = unetV2.get_unet_plus_inception()


model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])


# model_name = "Unet"
# model_name = "Inception-Unet"
model_name = "UnetV2"

# save_weights_path = "results-1/unet/"
# save_weights_path = "results-1/inception unet/"
save_weights_path = "results-1/UnetV2/"


orig_stdout = sys.stdout
f = open('ModelSummary.txt', 'w')
sys.stdout = f
print(model.summary())
sys.stdout = orig_stdout
f.close()


epoch = 100;


strTemp = save_weights_path + model_name + ".h5"


mc = ModelCheckpoint(strTemp.replace('.h5','.weights'), monitor='loss', mode='min', save_best_only=True, save_weights_only=True)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)


callbacks_list = [mc, es]


history = model.fit([x_train], [y_train],
                               validation_data=(x_valid, y_valid), 
                               callbacks=callbacks_list,
                               batch_size=5,
                               epochs=epoch)



strTemp = save_weights_path + model_name + ".history"

save_history(history,strTemp)





