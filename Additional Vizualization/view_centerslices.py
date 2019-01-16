# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 10:15:31 2018

@author: s120116

This file loads the middle slice of each scan an and mask, and pictures these
for the whole batch in a viewer. This can be used to verify the segmentations
@author: s120116
"""
import sys
sys.path.append('C:/Users/s120116/Documents/OneDrive - TU Eindhoven/TUE/Afstuderen/lung-nodule-msc-2018')
from radio import dataset as ds
import CTsliceViewer as slice
from radio.dataset import Pipeline
#from memory_profiler import profile
#import gc
import numpy as np
from CTImagesCustomBatch import CTImagesCustomBatch as CTICB




Path='C:/Users/linde/Documents/PreprocessedImages1008/Spacing(2x1x1)/*'
        
        
luna_index=ds.FilesIndex(path=Path,dirs=True,sort=True)
luna_dataset = ds.Dataset(index=luna_index, batch_class=CTICB)




ixs = np.array([['1.3.6.1.4.1.14519.5.2.1.6279.6001.312127933722985204808706697221']])


#observed_scans = ds.Dataset(index=images_index.create_subset(ixs), batch_class=CTICB)
#create pipeline to load images, and spacing & origin information
pipeline_load = (Pipeline()
                .load(fmt='blosc', components=['spacing', 'origin', 'images','segmentation']))

#give dataset to pipline and run it per batch
load_line= luna_dataset >> pipeline_load
#load_line=observed_scans >> pipeline_load


#create lists for middle slices of masks and images, and list for index numbers
list_of_masks=[]
list_of_im=[]
list_of_indices=[]

#obtain for whole batch middle slice of image and mask and index in list
batch_size=1
for i in range(int(np.ceil(len(luna_dataset)/batch_size))):
 
    batch=load_line.next_batch(batch_size=batch_size,shuffle=False)
    arrayIm,arrayMask=batch.get_middle_slices() #function returns middle slices of batch
    list_of_masks.append(arrayMask)
    list_of_im.append(arrayIm)
    list_of_indices.append(batch.index.indices)
    print(i)

#flatten lists of lists to a 3D array   
CenterImages=np.array([item for sublist in list_of_im for item in sublist])
CenterMasks=np.array([item for sublist in list_of_masks for item in sublist])
Indices=np.array([item for sublist in list_of_indices for item in sublist])

#
slice.masks_images_viewer(CenterImages,CenterMasks,Indices)
   
#slice.multi_slice_viewer(batch.images)
   

#observing one image
#index=ds.FilesIndex(path='C:/Users/s120116/Documents/subset0 - split/training/1.3.6.1.4.1.14519.5.2.1.6279.6001.303421828981831854739626597495',dirs=True)
#batch = CTICB(index)
#im=batch.load(fmt='blosc')
#slice.multi_slice_viewer(im.masks)
