# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 12:07:31 2018

@author: linde

In this script the features are extracted from the created nodule crops. An earlier trained model (CNN) is used for this, and the features are per crop saved.
"""


from keras import backend as K
import keras
import numpy as np
import os
import sys
from CTImagesCustomBatch import CTImagesCustomBatch as CTICB
from radio import dataset as ds
from radio.dataset import Pipeline
sys.path.append("../")


def make_dataset(folder):
    index=ds.FilesIndex(path=folder,dirs=True)
    dataset=ds.Dataset(index=index,batch_class=CTICB)
    return dataset      

#load classification model
cnn=keras.models.load_model('../models/neuralnet_final.h5')

#get correct output from model (feature vector)
flatten=cnn.get_layer(name="flatten_1")
inputs = [K.learning_phase()] + cnn.inputs
_flat_out = K.function(inputs, [flatten.output])


def flat_out_f(X):
    # The [0] is to disable the training phase flag
    return _flat_out([0] + [X])


#define names for folders
crops_folder='../ResultingData/NoduleCrops' #server path
pre_savepath='../ResultingData/NoduleFeatures' #change this lines into pe





#make dataset, and give dtaset to pipeline
pipeline_load= Pipeline().load(fmt='blosc', components=['spacing', 'origin', 'images'])
dataset= make_dataset(os.path.join(crops_folder, '*'))
sample_line=(dataset >> pipeline_load)


#for each scan in batch, load scan, and compute features from scan. Next, each batch is saved
for i in range(int(np.ceil(len(dataset)/5))):
   cbatch=sample_line.next_batch(batch_size=5, drop_last=False,shuffle=True)
   cim=cbatch.unpack(component='images',data_format='channels_last')
   features=flat_out_f(cim)
   for j in range(len(cim)):
       feat=features[0][j]
       totalpath=cbatch.index.get_fullpath(cbatch.indices[j])
       splits=totalpath.split(os.sep)  
       savepath=pre_savepath+ '/'+  splits[-1]
       if not os.path.exists(savepath):
           os.makedirs(savepath)
       np.save(savepath+'/features.npy',feat)        

   
   
 