# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:25:20 2018

@author: linde
"""

import numpy as np    # access to fast math
import matplotlib.pyplot as plt
from radio.dataset import FilesIndex, DatasetIndex
from radio.dataset import Pipeline
from radio import dataset as ds
import os
from CTImagesCustomBatch import CTImagesCustomBatch  
import CTsliceViewer as slices
import glob
import time

save_path='C:/Users/linde/Documents/PreprocessedImages_CS_PE/' 

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
    
for string in ['PE']   : #still do PE

    path_cs="C:/Users/linde/Documents/CS_PE_seperated/"+ string+ "/*"
    
    cs_index=FilesIndex(path=path_cs,dirs=True,sort=True)
    cs_dataset = ds.Dataset(index=cs_index, batch_class=CTImagesCustomBatch )
    
    
    #load and normalize these images
    load_and_normalize     = (Pipeline()
                            .load(fmt='blosc', components=['spacing', 'origin', 'images'])
                            .unify_spacing(shape=(400,512,512), spacing=(2.0,1.0,1.0),padding='constant') #equalizes the spacings 
                                  #from both images and mask
                           .normalize_hu(min_hu=-1200, max_hu=600)) #clips the HU values and linearly rescales them, values from grt team
                         #  .apply_lung_mask(paddi
        
        
        
            
    Path='C:/Users/linde/Documents/PreprocessedImages1008CorrectConvs/Spacing(2x1x1)/0*'
    loadSegm = (Pipeline()
                    .load(fmt='blosc', components=['segmentation', 'masks']))                
                            
    
    im_index=FilesIndex(path=Path,dirs=True)
    
    lunaline_train=(cs_dataset>> load_and_normalize)
    
    for i in range(len(cs_dataset)):
        batch_spect=lunaline_train.next_batch(batch_size=1,shuffle=False)
        
        index=batch_spect.index.indices
        
        index=im_index.create_subset(index)
        
        if os.path.exists(index.get_fullpath(index.indices[0])+'/masks' ):
            blosc_file = ds.Dataset(index=index, batch_class=CTImagesCustomBatch)
            blosc_line= (blosc_file >> loadSegm)
            
            batch_blosc= blosc_line.next_batch(batch_size=1)
            
            padding=170
            batch_spect.images = batch_spect.images * batch_blosc.masks + padding * (1-batch_blosc.masks)
            batch_spect.images[(batch_blosc.masks-batch_blosc.segmentation)*batch_spect.images >= 210] = 170
            
            batch_spect.segmentation=batch_blosc.segmentation
            batch_spect.masks=batch_blosc.masks
            
            dst=save_path+'/'+string+'/'
            if not os.path.exists(dst):
                os.makedirs(dst)
            batch_spect.dump(dst=dst,components=['spacing', 'origin', 'images', 'segmentation','masks'])
            
            
        
        else:
            dst=save_path+'/'+string+'/'
            batch_spect.dump(dst=dst, components=['spacing', 'origin', 'images'])
    
        print(i)