# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:56:54 2019

@author: s120116
The CNN for the detection
"""
from keras.optimizers import Adam, SGD
from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D, LeakyReLU, BatchNormalization, Flatten, Dense, Dropout, ZeroPadding3D, AveragePooling3D, Activation
from keras.models import Model, load_model, model_from_json
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error

from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
import keras


def get_net_detection(input_shape=(16,32,32,1)):
    cnn = keras.models.Sequential()
    
    # 1st layer group
    cnn.add(Convolution3D(64,3,3,3, activation= 'relu', input_shape=input_shape, 
                          padding= 'same', name='conv1', subsample=(1,1,1)))

    cnn.add(BatchNormalization())
 
    cnn.add(MaxPooling3D(pool_size=(1,2,2), strides= (1,2,2) ,  padding= 'valid',
                        name='pool1'))

    
    #2nd layer group
    cnn.add(Convolution3D(128,3,3,3, activation= 'relu', 
                          padding= 'same', name='conv2', subsample=(1,1,1)))
 
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2) ,
                         padding= 'valid', name='pool2'))
    
    
    #3th layer group
    cnn.add(Convolution3D(256,3,3,3, activation= 'relu', 
                          padding= 'same', name='conv3a', subsample=(1,1,1)))
 
#   
    cnn.add(BatchNormalization()) 
    cnn.add(Convolution3D(256,3,3,3, activation= 'relu', 
                          padding= 'same', name='conv3b', subsample=(1,1,1)))
    
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2) ,
                         padding= 'valid', name='pool3'))
    
    
    #4th layer group
    cnn.add(Convolution3D(512,3,3,3, activation= 'relu', 
                         padding= 'same', name='conv4a', subsample=(1,1,1)))
 
    cnn.add(BatchNormalization())
#    
    cnn.add(Convolution3D(512,3,3,3, activation= 'relu', 
                          padding= 'same', name='conv4b', subsample=(1,1,1)))
    
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2) ,
                         padding= 'valid', name='pool4'))
    

    
    cnn.add(Flatten())
    
    cnn.add(Dense(64))
    cnn.add(BatchNormalization())
    cnn.add(Activation("relu"))
    cnn.add(Dropout(0.5))
	

    cnn.add(Dense(1,activation='sigmoid'))

    
    Adam= keras.optimizers.Adam(lr=0.0001)
    cnn.compile(loss='binary_crossentropy', optimizer=Adam)
    
    print(cnn.summary())
    return cnn