# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:11:28 2018

@author: s120116
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 14 15:31:07 2018

@author: s120116
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:18:06 2018

@author: s120116

In this script images from a folder are loaded, divided in a training and validation set, and subsequently 
segmented. These segmentations are saved in the mask component. Training and validation images are saved
to different folders. 
In the sublist, the names of the subsets can be listed, to run over all these folders
"""


import sys
sys.path.append('C:/Users/s120116/Documents/OneDrive - TU Eindhoven/TUE/Afstuderen/lung-nodule-msc-2018')
from radio.dataset import FilesIndex
from radio.dataset import Pipeline
import numpy as np
from radio import dataset as ds
import os
from CTImagesCustomBatch import CTImagesCustomBatch 

#import CTsliceViewer as slice

sublist=['subset8', 'subset9']

for subset in sublist:
    print(subset)
    Split=True #dataset indices are existent
    
    #Define data folder (LUNA_mask)
    LUNA_MASK = 'C:/Users/s120116/Documents/LUNAsubsets/'+subset+'/*.mhd'    # set glob-mask for scans from Luna-dataset here
    
    #makes folder for all savings
    LUNA_val='C:/Users/s120116/Documents/Preprocessed_Images/'+subset+' - split/validate' 
    LUNA_train= 'C:/Users/s120116/Documents/Preprocessed_Images/'+subset+' - split/training' 
    
    
    LUNA_test='C:/Users/s120116/Documents/Preprocessed_Images/validationData/'
    if not os.path.exists(LUNA_test):
        os.makedirs(LUNA_test)
    
#    if not os.path.exists(LUNA_val):
#        os.makedirs(LUNA_val)
#        
#    if not os.path.exists(LUNA_train):
#        os.makedirs(LUNA_train)    
    
  
 
    #set up dataset structure
    luna_index = FilesIndex(path=LUNA_MASK, no_ext=True)      # preparing indexing structure
    luna_dataset = ds.Dataset(index=luna_index, batch_class=CTImagesCustomBatch)
    
    #Split dataset in training and validation part ----------------------------------------------
    
    #define path to save or load index files 
#    if Split:
#        # If dataset has already been split: make two subsets from indices for testing vs training
#        path='C:/Users/s120116/Documents/'+subset+' - split/'
#
#        index_train=np.load(os.path.join(path, 'trainindex.npy'))
#        luna_index_train=luna_index.create_subset(index_train)
#        dataset_train= ds.Dataset(index=luna_index_train, batch_class=CTImagesCustomBatch)
#        
#        index_val=np.load(os.path.join(path,'testindex.npy'))
#        luna_index_val=luna_index.create_subset(index_val)
#        dataset_val= ds.Dataset(index=luna_index_val, batch_class=CTImagesCustomBatch)
#    
#    else:
#        #else split dataset and save indices
#        luna_dataset.cv_split(0.85, shuffle=True)
#        
#        #save indexfiles for later use if needed        
#        np.save(os.path.join(path,'trainindex.npy'),luna_dataset.train.indices)
#        np.save(os.path.join(path,'testindex.npy'), luna_dataset.test.indices)
#        
#        #give them seperate names 
#        dataset_val=luna_dataset.test
#        dataset_train=luna_dataset.train
#    
    #-----------------------------------------------------------------------
    
    #make pipeline to load and segment, saves segmentations in masks
    load_and_segment     = (Pipeline()
                            .load(fmt='raw')
                            .get_lung_mask(rad=15)
                          .unify_spacing_withmask(shape=(400,512,512), spacing=(2.0,1.0,1.0),padding='constant') #equalizes the spacings 
                                  #from both images and mask
                           .normalize_hu(min_hu=-1200, max_hu=600) #clips the HU values and linearly rescales them, values from grt team
                           .apply_lung_mask(padding=170))
    
    ## uncomment this for validation data
    lunaline_test=(luna_dataset >> load_and_segment.dump(dst=LUNA_test,components=['spacing', 'origin', 'images','segmentation']))

    batch_size=1
    for i in range(np.ceil(len(luna_dataset)/batch_size).astype(int)):
        batch=lunaline_test.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1)
        print(i)
       
    
#    
#    #pass training dataset through pipeline
#    lunaline_train=(dataset_train >> load_and_segment.dump(dst=LUNA_train,components=['spacing', 'origin', 'images','segmentation']))
#
#    batch_size=1
#    for i in range(np.ceil(len(dataset_train)/batch_size).astype(int)):
#        batch=lunaline_train.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1)
#        print(i)
#       
#    
#    
#    
#    #pass validation dataset through pipeline
# 
#
#    lunaline_val=(dataset_val >> load_and_segment.dump(dst=LUNA_val,components=['spacing', 'origin', 'images','segmentation']))
#    
#  
#    batch_size=1
#    for i in range(np.ceil(len(dataset_val)/batch_size).astype(int)):
#        batch=lunaline_val.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1)
#        print(i)
    
