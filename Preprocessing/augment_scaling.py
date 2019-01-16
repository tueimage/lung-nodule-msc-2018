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
path='C:/Users/s120116/Documents/Preprocessed_Images/Crops(16x32x32)RandomBbox/'
#path='/home/lshesse/' #server path


#scales
possible_scales=np.arange(0.5,1.6, 0.1)


cancer_folder=path+'*/training/cancer/*/*'
val_cancer_folder=path+'*/validate/cancer/*/*'

save_folder_train= path + 'augmentedCrops/training/cancer/'
save_folder_test = path + 'augmentedCrops/validate/cancer/'

for i in possible_scales:
    if not os.path.exists(save_folder_train + f'{i:.1f}'):
        os.makedirs(save_folder_train+ f'{i:.1f}')
    
    
for i in possible_scales:
    if not os.path.exists(save_folder_test+ f'{i:.1f}'):
        os.makedirs(save_folder_test+ f'{i:.1f}')
    
#function to create dataset from folder name
def make_dataset(folder):
    index=ds.FilesIndex(path=folder,dirs=True)
    dataset=ds.Dataset(index=index,batch_class=CTICB)
    return dataset  

def augmentdata(batch, Scale): #for now, whole batch same spacing change, each scan 50% change to be flipped
    new_batch=batch.deepcopy()
    #scale whole batch with factor between 0.8 and 1.2
    spacing=new_batch.get(0,'spacing')
    spacing_randomizer=lambda *args: tuple(Scale* np.squeeze(spacing)) 
    
    new_batch.unify_spacing_withmask(spacing=spacing_randomizer(), shape=(new_batch.get(0,'images').shape))
    
    value = np.stack([self.get(i, component) for i in range(len(self))])

    #flip left right for each of the scans in the batch seperately
    for i in range(len(new_batch)):
        if np.random.choice(np.arange(2)) == 0: #50% change of flip batch
            new_batch[i].images=np.flip(new_batch[i].images,2)
            new_batch[i].masks=np.flip(new_batch[i].masks,2)
         
    return new_batch
# create pipeline to load images and give dataset structures to pipeline
pipeline_load= Pipeline().load(fmt='blosc', components=['spacing', 'origin', 'images','masks'])


cancer_trainset= make_dataset(cancer_folder)
cancer_testset= make_dataset(val_cancer_folder)


sample_cancer_train=(cancer_trainset >> pipeline_load)
sample_cancer_test=(cancer_testset >> pipeline_load)


batch_size=5
for i in range (math.ceil(len(sample_cancer_train)/batch_size)):
    batch=sample_cancer_train.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1)
    batch.segmentation=np.zeros_like(batch.images)

    for scale in possible_scales:
        new_batch=augmentdata(batch,scale) 
        new_batch.dump(dst=save_folder_train+f'{scale:.1f}', components=['spacing', 'origin', 'images','masks',])
        
        

for i in range (math.ceil(len(sample_cancer_test)/batch_size)):
    batch=sample_cancer_test.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1)
    batch.segmentation=np.zeros_like(batch.images)

    for scale in possible_scales:
        new_batch=augmentdata(batch,scale) 
        new_batch.dump(dst=save_folder_test+f'{scale:.1f}', components=['spacing', 'origin', 'images','masks',])        




