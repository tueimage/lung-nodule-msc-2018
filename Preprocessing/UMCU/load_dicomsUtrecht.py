# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np    # access to fast math
import matplotlib.pyplot as plt
from radio.dataset import FilesIndex
from radio.dataset import Pipeline
from radio import dataset as ds
import os
from CTImagesCustomBatch import CTImagesCustomBatch 
import CTsliceViewer as slices


save_folder= 'C:/Users/linde/Documents/testImages' 
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

#from each dicom folder, add one file to filesindex. This makes sure that with next batch, the next
#dicom scan is loaded and not the next slice (file)
fileList=[]
for i in range(1,3): #from 1 to number of scans
    number='00'+str(i)
    path='C:/Users/linde/Documents/DAta/DATA/Use/'+ number +'/conventional'
    fileList.append(path+ '/' +os.listdir(path)[0])
    

 
#set up dataset structure
luna_index = FilesIndex(path=fileList, no_ext=False,sort=True)      # preparing indexing structure


luna_dataset = ds.Dataset(index=luna_index, batch_class=CTImagesCustomBatch)

#load pipeline
load_LUNA   = (Pipeline()
               .load(fmt='dicom')
               .get_lung_mask(rad=10))



lunaline=luna_dataset >> load_LUNA.dump(dst=save_folder, components=['spacing', 'origin', 'images','masks'])

#get next batch
list_int=[]
i=0
while True:
    try:
        batch=lunaline.next_batch(batch_size=1, shuffle=False,n_epochs=1)
        im_array=batch.images
        [values, count] = np.unique(im_array, return_counts=True)
        list_int.append( [batch.index.indices, values, count])
        i=i+1
        
    except StopIteration:
        print('End of segmenting testing data')
        break


np.save('intensitie_counts_utrecht',np.array(list_int))


