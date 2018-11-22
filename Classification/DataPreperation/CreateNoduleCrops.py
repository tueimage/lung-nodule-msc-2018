# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:09:52 2018

@author: linde
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:31:53 2018

@author: linde
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:32:55 2018

@author: s120116
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:16:31 2018

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

In this script boxes are cropped around the nodules.  The preprocessed images are loaded, together with an excel file of the annotations.
For specifications about the excel file, see the readme. The cropped boxes are then saved per nodule in a folder. 
"""
import sys
sys.path.append("../")
sys.path.append("../HelperScripts")
from radio.dataset import FilesIndex
from radio.dataset import Pipeline
import pandas as pd
from radio import dataset as ds
import math
from CTImagesCustomBatch import CTImagesCustomBatch as CTICB #custom batch class
import os
import CTsliceViewer as slices

# Input  ----------------------------------------------------------------------------------
#file name of xlsx annotation file:
nodules_path='C:/Users/linde/OneDrive - TU Eindhoven/TUE/Afstuderen/CSVFILES/AnnotatiesPim/nodule_data_adapted.xlsx'

# ----------------------------------------------------------------------------------------------------------

#read annotations, dtype makes sure that the indexing item gets read as a string, such that leading zeros are not removed from name of path
nodules_info=pd.read_excel(nodules_path,dtype={'PatientID': str})

#names of folders of preprocessed images / 
path_data="../../../ResultingData/PreprocessedImages/*"
SaveFolder="../../..//ResultingData/NoduleCrops"


#create folder for savings
if not os.path.exists(SaveFolder):
    os.makedirs(SaveFolder)


#set up dataset structure
luna_index = FilesIndex(path=path_data, dirs=True)      # preparing indexing structure
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
for i in range (math.ceil(len(luna_dataset))):
    batch=cancer_train.next_batch(batch_size=1, shuffle=False,n_epochs=1)
    batch.dump(dst=SaveFolder, components=['origin','spacing',  'images'])
    print('crops made for scan:' + str(i))

        
        
  





    