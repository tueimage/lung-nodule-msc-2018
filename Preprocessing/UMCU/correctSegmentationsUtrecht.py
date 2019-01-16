# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 12:52:52 2018

@author: linde
"""
from radio.dataset import FilesIndex, DatasetIndex
import sys
sys.path.append('C:/Users/s120116/Documents/OneDrive - TU Eindhoven/TUE/Afstuderen/lung-nodule-msc-2018')
from radio import dataset as ds
import CTsliceViewer as slices
from radio.dataset import Pipeline
#from memory_profiler import profile
#import gc
import numpy as np
from CTImagesCustomBatch import CTImagesCustomBatch

import os

#get original images


#from each dicom folder, add one file to filesindex. This makes sure that with next batch, the next
#dicom scan is loaded and not the next slice (file)

load     = (Pipeline()
                        .load(fmt='dicom'))
                        
                        
lunaline_train=(luna_dataset>> load)#.dump(dst=LUNA_pre,components=['spacing', 'origin', 'images','segmentation', 'masks']))
#


#
batch_size=1
for i in range(279):#len(luna_dataset)):#np.ceil(len(dataset_train)/batch_size).astype(int)):
    start_time = time.time()
    batchnew=lunaline_train.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1, drop_last=False)
    batch.dump(dst=LUNA_pre,components=['spacing', 'origin', 'images','segmentation','masks'])
    print(i)
    print(np.round((time.time() - start_time)/60,2))       

slices.multi_slice_viewer(batchnew.images)
#get segmented images
Path='C:/Users/linde/Documents/PreprocessedImages1008CorrectConvs/Spacing(2x1x1)/SpacingNew/incorrect/*'
pipeline_loadblosc = (Pipeline()
                .load(fmt='blosc', components=['spacing', 'origin', 'images','segmentation', 'masks']))                
                        

im_index=FilesIndex(path=Path,dirs=True)


batch_size=1
ixs = np.array(['000274_IM000001'])
observed_scans = ds.Dataset(index=im_index.create_subset(ixs), batch_class=CTImagesCustomBatch)
observed_scans = ds.Dataset(index=im_index, batch_class=CTImagesCustomBatch)

lunaline_segm=(observed_scans>> pipeline_loadblosc)
batch_segm=lunaline_segm.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1, drop_last=False)
slices.multi_slice_viewer(batch_segm.masks)

slices.multi_slice_viewer(batch_segm.images)


fileList=[]
for i in range(12,13): #from 1 to number of scans

        number=str(i).zfill(6) 
        path='D:/DATA20181008/Spacing/' + number+'/conv'
        fileList.append(path+ '/' + os.listdir(path)[0])

luna_index=FilesIndex(path=fileList)
luna_dataset = ds.Dataset(index=luna_index, batch_class=CTImagesCustomBatch)



#segment newly
load_and_segment     = (Pipeline()
                        .load(fmt='dicom')
                       #.get_lung_mask(rad=15)
                       .unify_spacing(shape=(400,512,512), spacing=(2.0,1.0,1.0),padding='constant') #equalizes the spacings 
                              #from both images and mask
                       .normalize_hu(min_hu=-1200, max_hu=600)) #clips the HU values and linearly rescales them, values from grt team
                     #  .apply_lung_mask(padding=170))


lunaline_train=(luna_dataset>> load_and_segment)
import time
start_time=time.time()
batch=lunaline_train.next_batch(batch_size=1)
print(np.round((time.time() - start_time)/60,2))      


LUNA_pre='C:/Users/linde/Documents/PreprocessedImages1008CorrectConvs/Spacing(2x1x1)/SpacingNew/incorrect/updated' 
batch.dump(dst=LUNA_pre,components=['spacing', 'origin', 'images'])















