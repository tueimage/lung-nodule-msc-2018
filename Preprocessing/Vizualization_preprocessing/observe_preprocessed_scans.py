# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 10:31:24 2018

@author: s120116
"""

import sys
sys.path.append('C:/Users/s120116/Documents/OneDrive - TU Eindhoven/TUE/Afstuderen/lung-nodule-msc-2018')
from radio.dataset import FilesIndex
from radio.dataset import Pipeline
import numpy as np
from radio import dataset as ds
import os
from CTImagesCustomBatch import CTImagesCustomBatch as CTICB
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import CTsliceViewer as slices
import scipy 



nodules_df_2 = pd.read_csv('C:/Users/s120116/OneDrive - TU Eindhoven/TUE/Afstuderen/CSVFILES/annotations.csv')
     
LUNA_MASK = 'C:/Users/s120116/Documents/LUNAsubsets/'+subset+'/*.mhd' 
path='C:/Users/s120116/Documents/LUNAsubsets/subset*/*.mhd'
path='C:/Users/s120116/Documents/Preprocessed_Images/subset2 - split/training/*'
sub='subset0'

   
luna_index_train = FilesIndex(path=path, no_ext=True)      


  # preparing indexing structure


ixs = np.array(['1.3.6.1.4.1.14519.5.2.1.6279.6001.750792629100457382099842515038'])


two_scans_dataset = ds.Dataset(index=luna_index_train.create_subset(ixs), batch_class=CTICB)
luna_dataset_train = ds.Dataset(index=luna_index_train, batch_class=CTICB)


nodules_malignancy=pd.read_excel('C:/Users/s120116/OneDrive - TU Eindhoven/TUE/Afstuderen/CSVFILES/all_info_averaged_observer_corrected2.xlsx')

pipeline=   (Pipeline()
            .load(fmt='raw'))
            .fetch_nodules_info(nodules_df_2)
            .create_mask())


def load_pipeline(nodules_df):
    pipeline=   (Pipeline()
            .load(fmt='blosc', components=['spacing', 'origin', 'images','segmentation'])
           
      .fetch_nodules_info_malignancy(nodules_df)
           .create_mask()) #creates mask component with nodules
    return  pipeline                      
                 
cancer_cropline=load_pipeline(nodules_malignancy)

cancer_train=(two_scans_dataset >> load_and_segment)

batch=cancer_train.next_batch(batch_size=2)
batch.dump(dst='C:/Users/s120116/OneDrive - TU Eindhoven/TUE/Afstuderen', components=['spacing', 'origin', 'images','masks','nodules'])
np.save('C:/Users/s120116/OneDrive - TU Eindhoven/TUE/Afstuderen/'+ batch.indices[0]+'/nodules.npy', batch.nodules)




pipeline=(Pipeline().load(fmt='blosc', components=['spacing', 'origin', 'images','masks',]))
ixs = np.array(['1.3.6.1.4.1.14519.5.2.1.6279.6001.124663713663969377020085460568'])

luna_index_train = FilesIndex(path='C:/Users/s120116/OneDrive - TU Eindhoven/TUE/Afstuderen/1.3.6.1.4.1.14519.5.2.1.6279.6001.124663713663969377020085460568', dirs=True)   
luna_dataset_train = ds.Dataset(index=luna_index_train, batch_class=CTICB)

cancer=(luna_dataset_train >> pipeline)

batch=cancer.next_batch(batch_size=1)
slices.multi_slice_viewer(batch.images)
slices.multi_slice_viewer(batch.masks)
