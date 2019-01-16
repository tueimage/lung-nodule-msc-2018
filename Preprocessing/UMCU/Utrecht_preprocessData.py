# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 10:53:41 2018

@author: linde
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
from radio.dataset import FilesIndex, DatasetIndex
from radio.dataset import Pipeline
from radio import dataset as ds
import os
from CTImagesCustomBatch import CTImagesCustomBatch  
import CTsliceViewer as slices
import glob



import numpy as np    # access to fast math
import time
#makes folder for all savings
LUNA_pre='C:/Users/linde/Documents/PreprocessedImages1008CorrectConvs/Spacing(2x1x1)/SpacingNew/' 

if not os.path.exists(LUNA_pre):
    os.makedirs(LUNA_pre)
    




#from each dicom folder, add one file to filesindex. This makes sure that with next batch, the next
#dicom scan is loaded and not the next slice (file)
fileList=[]
for i in range(279): #from 1 to number of scans

        number=str(i).zfill(6) 
        path='D:/DATA20181008//' + number+'/conv'
        if os.path.isdir(path):
            fileList.append(path+ '/' + os.listdir(path)[0])






pathlist=glob.glob(path)
for i in range(len(pathlist)):
    path=pathlist[i]

    path1, im=os.path.split(path)
    path2, conv=os.path.split(path1)
    path3, patient=os.path.split(path2)
    os.rename(path,path2+'/'+ conv+'/'+patient+'_'+im )     
        


#ixs = np.array([['1.3.6.1.4.1.14519.5.2.1.6279.6001.312127933722985204808706697221']])
luna_index=FilesIndex(path=fileList)
luna_dataset = ds.Dataset(index=luna_index, batch_class=CTImagesCustomBatch)

try:
    import pydicom as dicom # pydicom library was renamed in v1.0
except ImportError:
    import dicom as dicom

indexlist=np.zeros(214)
contrast=[]
for i in range(214):
    os_, index= os.path.split(fileList[i])
    list_of_dicoms = dicom.dcmread(fileList[i])
    print(i)
    indexlist[i]=(int(index[:6]))

    
    contrastagent= list_of_dicoms.BodyPartExamined
    contrast.append(contrastagent)
    
new_contrast=[] 
for i in range(len(contrast)):
    if 'BLANCO' in contrast[i]:
        new_contrast.append('No cont')
    else:
        new_contrast.append('contrast')
       
dataframe['Contrast']=new_contrast
dataframe.groupby('labels')['Contrast'].value_counts()

##make pipeline to load and segment, saves segmentations in masks
load_and_segment     = (Pipeline()
                        .load(fmt='dicom')
                      .get_lung_mask(rad=15)
                        .unify_spacing_withmask(shape=(400,512,512), spacing=(2.0,1.0,1.0),padding='constant') #equalizes the spacings )
                              #from both images and mask
                       .normalize_hu(min_hu=-1200, max_hu=600) #clips the HU values and linearly rescales them, values from grt team
                      .apply_lung_mask(padding=170))
#
#
##pass training dataset through pipeline
lunaline_train=(luna_dataset>> load_and_segment.dump(dst=LUNA_pre,components=['images', 'masks', 'segmentation', 'spacing', 'origin', ]))
#
batch_size=1
for i in range(279):#len(luna_dataset)):#np.ceil(len(dataset_train)/batch_size).astype(int)):
    start_time = time.time()
    batch=lunaline_train.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1, drop_last=False)
 
   # batch.dump(dst=LUNA_pre,components=['origin'])#['spacing', 'origin', 'images','segmentation','masks'])
    print(i)
    print(np.round((time.time() - start_time)/60,2))

#   

#for i in range(10):
#    batch=lunaline_train.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1, drop_last=False)
#    seg_fill=new_seg(batch.images)
#    slice.multi_slice_viewer(seg_fill)
#    slice.multi_slice_viewer(batch.images)    
#    print(batch.indices)    
    
    
#import pandas as pd    
#nodules_df = pd.read_csv('D:/AnnotatiesPim/nodule_data.csv')
#
#load_and_segment     = (Pipeline()
#                        .load(fmt='dicom')
#                       .fetch_nodules_info_Utrecht(nodules_df)
#                       .create_mask_utrecht()
#                       .sample_nodules(share=1, batch_size=None,nodule_size=(16, 32, 32), data='Utrecht'))
#
#
#lunaline_train=(luna_dataset>> load_and_segment)
#batch=lunaline_train.next_batch(batch_size=1, shuffle=False,n_epochs=1, drop_last=False)
#
#slices.multi_slice_viewer(batch.images)

