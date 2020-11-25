from keras.applications.vgg16 import VGG16
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D



def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def conv_block_simple_no_bn(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def block(prevlayer, a, b, pooling):
    conva = Conv2D(a, (3, 3), activation='relu', padding='same')(prevlayer)
    conva = BatchNormalization()(conva)
    conva = Conv2D(b, (3, 3), activation='relu', padding='same')(conva)
    conva = BatchNormalization()(conva)
    if True == pooling:
        conva = MaxPooling2D(pool_size=(2, 2))(conva)
    
    
    convb = Conv2D(a, (5, 5), activation='relu', padding='same')(prevlayer)
    convb = BatchNormalization()(convb)
    convb = Conv2D(b, (5, 5), activation='relu', padding='same')(convb)
    convb = BatchNormalization()(convb)
    if True == pooling:
        convb = MaxPooling2D(pool_size=(2, 2))(convb)

    convc = Conv2D(b, (1, 1), activation='relu', padding='same')(prevlayer)
    convc = BatchNormalization()(convc)
    if True == pooling:
        convc = MaxPooling2D(pool_size=(2, 2))(convc)
        
    convd = Conv2D(a, (3, 3), activation='relu', padding='same')(prevlayer)
    convd = BatchNormalization()(convd)
    convd = Conv2D(b, (1, 1), activation='relu', padding='same')(convd)
    convd = BatchNormalization()(convd)
    if True == pooling:
        convd = MaxPooling2D(pool_size=(2, 2))(convd)
        
    up = concatenate([conva, convb, convc, convd])
    return up


img_rows = 224
img_cols = 224
depth = 3

def get_unet():
    inputs = Input((img_rows, img_cols, depth))
    
    conv1 = block(inputs, 16, 32, True)
    
    conv2 = block(conv1, 32, 64, True)

    conv3 = block(conv2, 64, 128, True)
    
    conv4 = block(conv3, 128, 256, True)
    
    conv5 = block(conv4, 256, 512, True)
    
    # **** decoding ****
    xx = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)    
    up1 = block(xx, 512, 128, False)
    
    xx = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(up1), conv3], axis=3)    
    up2 = block(xx, 256, 64, False)
    
    xx = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up2), conv2], axis=3)   
    up3 = block(xx, 128, 32, False)
    

    xx = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(up3), conv1], axis=3)   
    up4 = block(xx, 64, 16, False)

    xx = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(up4), inputs], axis=3)

    xx = Conv2D(32, (3, 3), activation='relu', padding='same')(xx)
#    xx = concatenate([xx, conv1a]) 
    

    xx = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(xx)


    model = Model(inputs=[inputs], outputs=[xx])

    
    return model
