# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 13:58:01 2018

@author: s120116
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 11:52:04 2018

@author: s120116
"""

import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/s120116/OneDrive - TU Eindhoven/TUE/Afstuderen/lung-nodule-msc-2018')
import numpy as np
import keras
from CTImagesCustomBatch import CTImagesCustomBatch as CTICB
from radio import dataset as ds
from radio.dataset import Pipeline
from datetime import datetime
startTime = datetime.now()
import random
import CNNfile as model
import CTsliceViewer as slice
import matplotlib.patches as mpatches
import os
import keras.backend as K 
import pandas as pd
from radio.dataset import F
from skimage import measure, morphology
import time
import tensorflow as tf
import keras.backend.tensorflow_backend
import evalJulianFunction as im_eval
import math

# Define folders containing train and validation crops

path='/home/lshesse/Datasets/Crops(16x32x32)/'


#scales
possible_scales=np.arange(0.5,1.6, 0.1)


cancer_folder=path+'*/training/cancer/*/*'
val_cancer_folder=path+'*/validate/cancer/*/*'

save_folder_train= path + 'augmentedCropsFlip/training/cancer/'
save_folder_test = path + 'augmentedCropsFlip/validate/cancer/'


foldernames=['leftright', 'updown', 'frontback']
for folder in foldernames:
    if not os.path.exists(save_folder_train + folder):
        os.makedirs(save_folder_train+ folder)

    
#function to create dataset from folder name
def make_dataset(folder):
    index=ds.FilesIndex(path=folder,dirs=True)
    dataset=ds.Dataset(index=index,batch_class=CTICB)
    return dataset  

def flipData(batch, axis): #for now, whole batch same spacing change, each scan 50% change to be flipped

    new_batch=batch.deepcopy()
    #flip left right for each of the scans in the batch seperately
    for i in range(len(new_batch)):
            new_batch[i].images=np.flip(new_batch[i].images,axis)
            new_batch[i].masks=np.flip(new_batch[i].masks,axis)
         
    return new_batch
# create pipeline to load images and give dataset structures to pipeline
pipeline_load= Pipeline().load(fmt='blosc', components=['spacing', 'origin', 'images','masks'])


cancer_trainset= make_dataset(cancer_folder)
sample_cancer_train=(cancer_trainset >> pipeline_load)



batch_size=5
for i in range (math.ceil(len(sample_cancer_train)/batch_size)):
    batch=sample_cancer_train.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1)
   
    #flip left right
    batch_lr=flipData(batch,2) 
    batch_lr.dump(dst=save_folder_train+'leftright', components=['spacing', 'origin', 'images','masks'])
    
    #flip left right
    batch_updown=flipData(batch,1) 
    batch_updown.dump(dst=save_folder_train+'updown', components=['spacing', 'origin', 'images','masks'])
        
    #flip left right
    batch_front=flipData(batch,1) 
    batch_front.dump(dst=save_folder_train+'frontback', components=['spacing', 'origin', 'images','masks'])

    #no flip
    batch.dump(dst=save_folder_train+'noflips', components=['spacing', 'origin', 'images','masks'])


#for flipping
        


