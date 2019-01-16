# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:30:51 2018


This script can be used to extract the features from nodule crops using a trained CNN.
The features are saved to the correct folders together with the label of the nodule
@author: s120116
"""
from keras import backend as K
from radio import dataset as ds
from radio.dataset import Pipeline
from CTImagesCustomBatch import CTImagesCustomBatch as CTICB
import keras
import helper_functions as helper
import sklearn.decomposition as decom
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as measure
import os
#cnn = keras.models.load_model('C:/Users/s120116/Documents/FinalModels/neuralnet_final(32x32x64)')

import tensorflow as tf
import keras.backend.tensorflow_backend
#import EvaluationScript27_08 as im_eval
#import classificationCNN as model
import keras


#configure GPU
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.tensorflow_backend.set_session(session)

cnn=keras.models.load_model('C:/Users/linde/OneDrive - TU Eindhoven/TUE/Afstuderen/FinalModelClassification/neuralnet_final.h5')
#cnn.summary()

flatten=cnn.get_layer(name="flatten_1")

inputs = [K.learning_phase()] + cnn.inputs


_flat_out = K.function(inputs, [flatten.output])


def flat_out_f(X):
    # The [0] is to disable the training phase flag
    return _flat_out([0] + [X])

#path='C:/Users/s120116/Documents/Preprocessed_Images/FinalCrops(32x64x64)/Crops(16x32x32)CancerwithMalignancy/'
cancer_folder='C:/Users/linde/Documents/Crops(16x32x32)Utrecht_conv/*' #server path



pre_savepath='C:/Users/linde/Documents/balbaltest/' #change this lines into pe

sample_line, dataset =helper.load_line_folder(cancer_folder)

features_total=[]

for i in range(len(dataset)):
   cbatch=sample_line.next_batch(batch_size=5, drop_last=False,shuffle=True)
   cim=cbatch.unpack(component='images', data_format='channels_last')
   features=flat_out_f(cim)
   
   for j in range(len(cim)):
       feat=features[0][j]
       #feature_red=measure.block_reduce(feat, block_size=(2,2,2,1), func=np.max )
       totalpath=cbatch.index.get_fullpath(cbatch.indices[j])
       splits=totalpath.split(os.sep)

     #  savepath=pre_savepath+ '/'+ splits[-2]  +'/'+ splits[-1]
       
       savepath=pre_savepath+ '/'+  splits[-1]
       if not os.path.exists(savepath):
           os.makedirs(savepath)
       np.save(savepath+'/features.npy',feat) 
       features_total.append(feat)
       nodule_info=np.load(totalpath+'/nodules.npy')
       np.save(savepath+ '/nodules.npy',nodule_info)
   print(i)
   
   

#
#ar=np.array(feature_array)
#
#pca_fun=decom.PCA(n_components=5)
#pca_fun.fit(ar)
#
#new_ary=pca_fun.transform(ar)
   
features=np.array(features_total   )
import sklearn
pca=sklearn.decomposition.PCA(n_components=10)
features_pca=pca.fit_transform(features)

plt.figure()
# Plot the training points
plt.scatter(features_pca[:, 0],features_pca[:, 1], c=np.ravel(labels),   edgecolor='k')
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig, elev=-150, azim=110)

ax.scatter(features_pca[:, 0], features_pca[:, 1], features_pca[:, 2], 
           cmap=plt.cm.Set1, edgecolor='k', s=40)