# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:56:54 2019

@author: s120116
The CNN for the detection
"""

from keras.layers import  Convolution3D, MaxPooling3D,  BatchNormalization, Flatten, Dense, Dropout, Activation
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