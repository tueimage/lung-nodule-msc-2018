# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 10:40:19 2018

@author: linde

In this script all dicom files are loaded and preprocessed. For preprocessing intensity normalization takes place, and the pixel spacing is adapted for each scan. 
Once the dicoms are processed, they are saved as blosc files. The images, spacing, and origin are each saved together in a folder for each scan.
"""

import sys
sys.path.append("../")
sys.path.append("../HelperScripts")
from radio.dataset import FilesIndex
from radio.dataset import Pipeline
from radio import dataset as ds
import os
from CTImagesCustomBatch import CTImagesCustomBatch as CTICB
import CTsliceViewer as slices
import numpy as np    

#input --------------------------------------------------------------------------------
#put here path with data
data_path='D:/OnlyConv/' #folder containing for each dicom file a folder with all slices (files)

#--------------------------------------------------------------------------------------------


savepath='../ResultingData/PreprocessedImages' 

#makes folder for all savings
if not os.path.exists(savepath):
    os.makedirs(savepath)
    

#create filesindex to iterate over all files
folder_path=os.path.join(data_path, '*')
scan_index=FilesIndex(path=folder_path,dirs=True)
scan_dataset = ds.Dataset(index=scan_index, batch_class=CTICB)

#to check index / dataset use: luna_index.indices or scan_dataset.index.indices
#should contain list of names of folders (for each scan a folder), names should be different for each scan    

#make pipeline to load, equalize spacing and normalize the data
load_and_preprocess     = (Pipeline()
                        .load(fmt='dicom') #loads all slices from folder in dataset
                        .unify_spacing(shape=(400,512,512), spacing=(2.0,1.0,1.0),padding='constant')#equalizes the spacings
                       .normalize_hu(min_hu=-1200, max_hu=600) #clips the HU values and linearly rescales them
                      )

##pass training dataset through pipeline
preprocessing_pipeline=(scan_dataset>> load_and_preprocess.dump(dst=savepath,components=['images', 'spacing', 'origin' ]))


#get scans one by one through the pipeline
for i in range(len(scan_dataset)):
    print('prepoccesing scan nr:'+ str(i))
    batch=preprocessing_pipeline.next_batch(batch_size=1, shuffle=False,n_epochs=1, drop_last=False)



#to observe a certain scan after preprocessing:
#slices.multi_slice_viewer(batch.images) #scroll through slices with j and k
#to acces other components use: batch.origin / batch.spacing  
