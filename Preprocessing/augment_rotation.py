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
import scipy
# Define folders containing train and validation crops

path='/home/lshesse/Datasets/Crops(16x32x32)/'


#scales



cancer_folder=path+'*/training/cancer/*/*'
val_cancer_folder=path+'*/validate/cancer/*/*'

save_folder_train= path + 'augmentedRotations/training/cancer/'




    
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


def rotateData(batch,axes,angle):
    new_batch=batch.deepcopy()
    for i in range(len(new_batch)):
        new_batch[i].images=scipy.ndimage.rotate(new_batch[i].images, angle, axes)
        new_batch[i].segmentation=scipy.ndimage.rotate(new_batch[i].segmentation, angle, axes)
         
    return new_batch


# create pipeline to load images and give dataset structures to pipeline
pipeline_load= Pipeline().load(fmt='blosc', components=['spacing', 'origin', 'images','masks'])


cancer_trainset= make_dataset(cancer_folder)
sample_cancer_train=(cancer_trainset >> pipeline_load)



batch_size=5
for i in range (math.ceil(len(sample_cancer_train)/batch_size)):
    batch=sample_cancer_train.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1)
   
    #first axis = {0,1}
    for k in range(3):
        angle=np.random.randint(-180, 180)
        new_batch=rotateData(batch,(0,1), angle)
        new_batch.dump(dst=save_folder_train, components=['spacing', 'origin', 'images','masks'])
    
    for k in range(3):
        angle=np.random.randint(-180, 180)
        new_batch=rotateData(batch,(1,2), angle)
        new_batch.dump(dst=save_folder_train, components=['spacing', 'origin', 'images','masks'])
        
    for k in range(3):
        angle=np.random.randint(-180, 180)
        new_batch=rotateData(batch,(0,2), angle)
        new_batch.dump(dst=save_folder_train, components=['spacing', 'origin', 'images','masks'])

    


#for flipping
        


