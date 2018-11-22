# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 16:52:17 2018

@author: linde

This is the code for the complete data preperation , the code consist of three parts
1. Scan preprocessing
2. Cropping of nodule cubes
3. Feature extracture for each cube

The result of the code is for each nodule a feature vector saved as numpy asrray
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
import pandas as pd
import keras
from keras import backend as K

def make_dataset(folder):
    index=ds.FilesIndex(path=folder,dirs=True)
    dataset=ds.Dataset(index=index,batch_class=CTICB)
    return dataset    


#input --------------------------------------------------------------------------------
#put here path with data
data_path='D:/OnlyConv/' #folder containing for each dicom file a folder with all slices (files)
nodules_path='C:/Users/linde/OneDrive - TU Eindhoven/TUE/Afstuderen/CSVFILES/AnnotatiesPim/nodule_data_adapted.xlsx'

# Preprocessing Images--------------------------------------------------------------------------------------------


savepath_preprocess='../../../ResultingData/PreprocessedImages' 

#makes folder for all savings
if not os.path.exists(savepath_preprocess):
    os.makedirs(savepath_preprocess)
    

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
preprocessing_pipeline=(scan_dataset>> load_and_preprocess.dump(dst=savepath_preprocess,components=['images', 'spacing', 'origin' ]))


#get scans one by one through the pipeline
for i in range(len(scan_dataset)):
    print('prepoccesing scan nr:'+ str(i))
    batch=preprocessing_pipeline.next_batch(batch_size=1, shuffle=False,n_epochs=1, drop_last=False)
    
    
    
    
    


# Creating Nodule Crops ----------------------------------------------------------------------------------------------------------

#read annotations, dtype makes sure that the indexing item gets read as a string, such that leading zeros are not removed from name of path
nodules_info=pd.read_excel(nodules_path,dtype={'PatientID': str})

#name of savepath for nodulecrops
savepath_crops="../../../ResultingData/NoduleCrops"


#create folder for savings
if not os.path.exists(savepath_crops):
    os.makedirs(savepath_crops)


#set up dataset structure
luna_index = FilesIndex(path=savepath_preprocess+'/*', dirs=True)      # preparing indexing structure
luna_dataset= ds.Dataset(index=luna_index, batch_class=CTICB) 



#make pipeline for loading and sampling of the nodules
crop_pipeline=   (Pipeline()
            .load(fmt='blosc', components=['spacing', 'origin', 'images'])
            .fetch_nodules_info_general(nodules_info) #loads nodule infomation into batch
            .create_mask()
            .sample_nodules(batch_size=None, nodule_size=(16,32,32), share=(1.0),variance=(0,0,0),data='Utrecht'))


#pass data through pipeline
cancer_train=(luna_dataset >> crop_pipeline)#          
   

#apply pipeline for each scan in the dataset
for i in range (len(luna_dataset)):
    batch=cancer_train.next_batch(batch_size=1, shuffle=False,n_epochs=1)
    batch.dump(dst=savepath_crops, components=['origin','spacing',  'images'])
    print('crops made for scan:' + str(i))    
    
    
    
# Features Extraction -------------------------------------------------------------- 

#load classification model
cnn=keras.models.load_model('../models/neuralnet_final.h5')

#get correct output from model (feature vector)
flatten=cnn.get_layer(name="flatten_1")
inputs = [K.learning_phase()] + cnn.inputs
_flat_out = K.function(inputs, [flatten.output])


def flat_out_f(X):
    # The [0] is to disable the training phase flag
    return _flat_out([0] + [X])


#define names for folders
savepath_features='../../../ResultingData/NoduleFeatures' #change this lines into pe


#make dataset, and give dtaset to pipeline
pipeline_load= Pipeline().load(fmt='blosc', components=['spacing', 'origin', 'images'])
dataset= make_dataset(os.path.join(savepath_crops, '*'))
sample_line=(dataset >> pipeline_load)


#for each scan in batch, load scan, and compute features from scan. Next, each batch is saved
for i in range(int(np.ceil(len(dataset)/5))):
   cbatch=sample_line.next_batch(batch_size=5, drop_last=False,shuffle=False)
   cim=cbatch.unpack(component='images',data_format="channels_last")
   features=flat_out_f(cim)
   for j in range(len(cim)):
       feat=features[0][j]
       totalpath=cbatch.index.get_fullpath(cbatch.indices[j])
       splits=totalpath.split(os.sep)  
       savepath=savepath_features + '/'+  splits[-1]
       if not os.path.exists(savepath):
           os.makedirs(savepath)
       np.save(savepath+'/features.npy',feat)        
   print(i)
    
    