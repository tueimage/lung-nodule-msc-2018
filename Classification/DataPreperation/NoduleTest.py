# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:36:46 2018

@author: linde

This script crops nodules in boxes from unpreprocessed files. It should not used in the final workflow, but can be used to check whether the annotations are correctly processed.
The crops which are produced should contain a nodule in the middle of the box, if this is not the case something is wrong. See the read me for suggestions on this. 
"""
import sys
sys.path.append("../")
sys.path.append("../HelperScripts")
from radio.dataset import FilesIndex
from radio.dataset import Pipeline
import pandas as pd
from radio import dataset as ds
from CTImagesCustomBatch import CTImagesCustomBatch as CTICB #custom batch class
import os
import CTsliceViewer as slices


nodules_path='C:/Users/linde/OneDrive - TU Eindhoven/TUE/Afstuderen/CSVFILES/AnnotatiesPim/nodule_data_adapted.xlsx'
data_path='D:/OnlyConv'

#get nodule info, the dtype preserves the leading zeros to to get folder names and this name equal, if  numbers are used this is not necessary
nodules_utrecht=pd.read_excel(nodules_path,dtype={'PatientID_new': str})


#make pipeline to load, and get the annotations
load_and_preprocess     = (Pipeline()
                         .load(fmt='dicom')
                         .fetch_nodules_info_general(nodules_utrecht) #loads nodule infomation into batch
                        .create_mask()
                        .sample_nodules(batch_size=None, nodule_size=(16,32,32), share=(1.0),variance=(0,0,0),data='Utrecht'))
                        


#create filesindex to iterate over all files
folder_path=os.path.join(data_path, '*')
scan_index=FilesIndex(path=folder_path,dirs=True)
scan_dataset = ds.Dataset(index=scan_index, batch_class=CTICB)

# get dataset to pipeline, and get a batch through the pipeline, for the next batch run the 2nd command multiple times
line=(scan_dataset >> load_and_preprocess )
batch=line.next_batch(batch_size=1)


slices.multi_slice_viewer(batch.images) #check if nodules are present in center of box, navigation with j and k


#if the boxes are not correct, one can do the load_and_preprocess pipeline without the sample_nodules. This results in a batch with the CT scan in the image component, and a mask
# with the nodules in the mask component. Then it can be observerd how the annotation deviates:

load_and_preprocess_lim    = (Pipeline()
                         .load(fmt='dicom')
                         .fetch_nodules_info_general(nodules_utrecht) #loads nodule infomation into batch
                        .create_mask())
line=(scan_dataset >> load_and_preprocess_lim )
batch=line.next_batch(batch_size=1, shuffle=False)

slices.multi_slice_viewer(batch.images)
slices.multi_slice_viewer(batch.masks)