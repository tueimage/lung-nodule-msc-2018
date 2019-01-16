# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 12:21:32 2018

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

LUNA_pre='C:/Users/linde/Documents/PreprocessedImages_Spectral/' 

if not os.path.exists(LUNA_pre):
    os.makedirs(LUNA_pre)
    
  
string=['080', '120', '190', 'conv']

for num in string:   
#start with1 energy

    fileList=[]
    for i in range(0,300): #from 1 to number of scans
            number=str(i).zfill(6) 
            path='D:/DATA20181008/' + number+'/'+ num
            if os.path.isdir(path)==True:
                fileList.append(path+ '/' + os.listdir(path)[0])
    
    
    luna_index=FilesIndex(path=fileList,sort=True)
    luna_dataset = ds.Dataset(index=luna_index, batch_class=CTImagesCustomBatch)
        
    #load and normalize these images
    load_and_normalize     = (Pipeline()
                            .load(fmt='dicom')
                            .unify_spacing(shape=(400,512,512), spacing=(2.0,1.0,1.0),padding='constant') #equalizes the spacings 
                                  #from both images and mask
                           .normalize_hu(min_hu=-1200, max_hu=600)) #clips the HU values and linearly rescales them, values from grt team
                         #  .apply_lung_mask(paddi
        
        
    Path='C:/Users/linde/Documents/PreprocessedImages1008CorrectConvs/Spacing(2x1x1)/0*'
    loadblosc = (Pipeline()
                    .load(fmt='blosc', components=['segmentation', 'masks']))                
                            
    
    im_index=FilesIndex(path=Path,dirs=True)
    
        
    lunaline_train=(luna_dataset>> load_and_normalize)
    
    
    
    
    
    for i in range(len(luna_dataset)):
        batch_spect=lunaline_train.next_batch(batch_size=1)
        
        index=batch_spect.index.indices
        
        index=im_index.create_subset(index)
        
        if os.path.exists(index.get_fullpath(index.indices[0])+'/masks' ):
            blosc_file = ds.Dataset(index=index, batch_class=CTImagesCustomBatch)
            blosc_line= (blosc_file >> loadblosc)
            
            batch_blosc= blosc_line.next_batch(batch_size=1)
            
            padding=170
            batch_spect.images = batch_spect.images * batch_blosc.masks + padding * (1-batch_blosc.masks)
            batch_spect.images[(batch_blosc.masks-batch_blosc.segmentation)*batch_spect.images >= 210] = 170
            
            batch_spect.segmentation=batch_blosc.segmentation
            batch_spect.masks=batch_blosc.masks
            
            dst=LUNA_pre+'/'+np.array_str(batch_spect.index.indices)[2:-2]+'/'+ num
            if not os.path.exists(dst):
                os.makedirs(dst)
            batch_spect.dump(dst=dst,components=['spacing', 'origin', 'images', 'segmentation','masks'])
            
            elemlist=glob.glob(dst+'/*/*' )
            for elem in elemlist:
                os.rename(elem, dst+'/'+os.path.basename(elem))
            os.rmdir(dst+'/'+np.array_str(batch_spect.index.indices)[2:-2])
        
        else:
            dst=LUNA_pre+'/'+np.array_str(batch_spect.index.indices)[2:-2]+'/'+ num
            batch_spect.dump(dst=dst, components=['spacing', 'origin', 'images'])
            elemlist=glob.glob(dst+'/*/*' )
            for elem in elemlist:
                os.rename(elem, dst+'/'+os.path.basename(elem))
            os.rmdir(dst+'/'+np.array_str(batch_spect.index.indices)[2:-2])
        print(i)
#rename everyth   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# #code to get excluded out   
#save_excluded='D:/DATA20181008/Excluded'
#if not os.path.exists(save_excluded):
#    os.makedirs(save_excluded)  
#    
#path_convs=    'C:/Users/linde/Documents/PreprocessedImages1008CorrectConvs/Spacing(2x1x1)/Excluded/*'
#pathlist=glob.glob(path_convs)
#
#path_dicoms=    'D:/DATA20181008/Spacing/*'
#dicomlist=glob.glob(path_dicoms)
#
#convslist=[os.path.basename(conv)[:6] for conv in pathlist]
#for i in range(len(dicomlist)):
#    dicom=dicomlist[i]
#    if os.path.basename(dicom)  in convslist:
#        os.rename(dicom, os.path.dirname(dicom)+ '/Excluded/' + os.path.basename(dicom))
#        
#path_ex=    'D:/DATA20181008/Excluded/*'
#pathlist=glob.glob(path_ex)