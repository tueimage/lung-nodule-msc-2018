# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:13:53 2018

@author: s120116
"""

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
#subset0
#1.3.6.1.4.1.14519.5.2.1.6279.6001.303421828981831854739626597495

#subset1
#1.3.6.1.4.1.14519.5.2.1.6279.6001.243094273518213382155770295147
#1.3.6.1.4.1.14519.5.2.1.6279.6001.503980049263254396021509831276
#1.3.6.1.4.1.14519.5.2.1.6279.6001.861997885565255340442123234170
#1.3.6.1.4.1.14519.5.2.1.6279.6001.935683764293840351008008793409

#subset4
#1.3.6.1.4.1.14519.5.2.1.6279.6001.339546614783708685476232944897
#1.3.6.1.4.1.14519.5.2.1.6279.6001.228511122591230092662900221600- validate


sublist=['subset1', 'subset2', 'subset3', 'subset4','subset5', 'subset6', 'subset7']
subset='subset4'

  #Define data folder (LUNA_mask)
LUNA_MASK = 'C:/Users/s120116/Documents/Allfolders/'+subset+'/*.mhd'    # set glob-mask for scans from Luna-dataset here

#makes folder for all savings
LUNA_val='C:/Users/s120116/Documents/Preprocessed_Images/'+subset+' - split/validate' 
LUNA_train= 'C:/Users/s120116/Documents/Preprocessed_Images/'+subset+' - split/training' 


luna_index = FilesIndex(path=LUNA_MASK, no_ext=True) 


ixs = np.array([

'1.3.6.1.4.1.14519.5.2.1.6279.6001.228511122591230092662900221600'])
fix_ds = ds.Dataset(index=luna_index.create_subset(ixs), batch_class=CTImagesCustomBatch) 

 #make pipeline to load and segment, saves segmentations in masks
load_and_segment     = (Pipeline()
                        .load(fmt='raw')
                        .get_lung_mask(rad=15))
                    #  .unify_spacing_withmask(shape=(400,512,512), spacing=(2.0,1.0,1.0),padding='constant') #equalizes the spacings 
                              #from both images and mask
                     # .normalize_hu(min_hu=-1200, max_hu=600) #clips the HU values and linearly rescales them, values from grt team
                      #.apply_lung_mask(padding=170))


#pass training dataset through pipeline
lunaline_train=(fix_ds >> load_and_segment )#.dump(dst=LUNA_train,components=['spacing', 'origin', 'images','segmentation']))
batch=lunaline_train.next_batch(batch_size=1, shuffle=False,n_epochs=1)

batch.dump(dst='C:/Users/s120116/Documents/Preprocessed_Images/subset1 - split', components=['spacing', 'origin', 'images','segmentation']) 



import CTsliceViewer as slices
slices.multi_slice_viewer(batch.images)
slices.multi_slice_viewer(batch.segmentation)

