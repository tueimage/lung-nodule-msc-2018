# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 09:42:42 2018

@author: s120116
"""

import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/s120116/OneDrive - TU Eindhoven/TUE/Afstuderen/lung-nodule-msc-2018')

from CTImagesCustomBatch import CTImagesCustomBatch as CTICB
from radio import dataset as ds
from radio.dataset import Pipeline
from datetime import datetime
startTime = datetime.now()


import pandas as pd

import os

sublist=['subset0', 'subset1', 'subset2', 'subset3', 'subset4','subset5', 'subset6', 'subset7']
for sub in sublist:


#LUNA_test='C:/Users/s120116/Documents/subset* - split/testing/*' 
#LUNA_train='C:/Users/s120116/Documents/subset* - split/training/*' 

    LUNA_test='/home/lshesse/'+ sub+' - split/testing/*' 
    LUNA_train='/home/lshesse/'+ sub+' - split/training/*' 
    nodules_df = pd.read_csv('C:/Users/s120116/OneDrive - TU Eindhoven/TUE/Afstuderen/CSVFILES/annotations.csv')
    #nodules_df = pd.read_csv('/home/lshesse/annotations.csv')
    
    
    luna_index_test = ds.FilesIndex(path=LUNA_test, dirs=True)      # preparing indexing structure
    luna_dataset_test = ds.Dataset(index=luna_index_test, batch_class=CTICB)
    
    luna_index_train = ds.FilesIndex(path=LUNA_train, dirs=True)      # preparing indexing structure
    luna_dataset_train = ds.Dataset(index=luna_index_train, batch_class=CTICB)
    
    
    
    #save folders
    
    path='/hpme/lshesse/'
    SaveFolder= 'preprocessed_files/'
    
    test_folder=path+SaveFolder+sub+'testing'
    train_folder=path+SaveFolder+sub+'training'
    

        
    folderlist=[test_folder,train_folder]
    
    for folder in folderlist:
         if not os.path.exists(folder):
             os.makedirs(folder)
             
    
    #this pipeline does the preprocessing and gets the ground truth for the image
    preprocessing	       =     (Pipeline()
                                  .load(fmt='blosc', components=['spacing', 'origin', 'images','masks']) 
                                
                                  .unify_spacing_withmask(shape=(400,512,512), spacing=(2.0,1.0,1.0), padding='constant') #equalizes the spacings 
                    
                                  .normalize_hu(min_hu=-1200, max_hu=600) #clips the HU values and linearly rescales them, values from grt team
                                  .apply_lung_mask(padding=170)#eventueel nog bot weghalen)
                                  .fetch_nodules_info(nodules_df)
                                  .create_mask())
                                #  .predict_on_scans(cnn,strides=(8,16,16), crop_shape=(16,32,32), targets_mode='classification', model_type='keras'))
    #                              .create_mask()
    #                        
    
    
    
    preprocess_line_test=(luna_dataset_test >> preprocessing.dump(dst=test_folder)) 
    preprocess_line_train=(luna_dataset_train >> preprocessing.dump(dst=train_folder))
    
 
    
    
    for i in range(len(luna_dataset_test)):
        batch=preprocess_line_test.next_batch(batch_size=1)
        
    for i in range(len(luna_dataset_train)):
        batch=preprocess_line_train.next_batch(batch_size=1)
        
        
        
    
    
   
    