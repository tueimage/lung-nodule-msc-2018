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

This script is developed to get small crops of cancerous and non cancerous crops, which can be used to train the network.
The crops are created seperately and dumped in their own folder. For the cancerous crops, every nodules existent in the images (in the csv file) is cropped 
to and dumped. For the non-cancerous crops, 10 crops are created per batch (3) 

The process is performed for each subset, both for the validation and training set
"""
import sys
sys.path.append('C:/Users/s120116/Documents/OneDrive - TU Eindhoven/TUE/Afstuderen/lung-nodule-msc-2018')
from radio.dataset import FilesIndex
from radio.dataset import Pipeline
import pandas as pd

from radio import dataset as ds
import math
from CTImagesCustomBatch import CTImagesCustomBatch as CTICB #custom batch class
import random
import os
import numpy as np
import CTsliceViewer as slices
#from radio import CTImagesMaskedBatch as CTIMB
#names of folders in which images with segmentations are


#add subset 0 usually
CancerCropsMalignant= True
CancerCrops=False
RandomCrops=False
CandidateCrops=False
FalsePosCrops=False

Num_Cancer=1 #number of variations per nodule
crop_size=(16,32,32)

 #get nodule info
nodules_utrecht=pd.read_csv('C:/Users/linde/OneDrive - TU Eindhoven/TUE/Afstuderen/CSVFILES/AnnotatiesPim/nodule_data.csv')



path_data='C:/Users/linde/Documents/PreprocessedImages1008CorrectConvs/Spacing(2x1x1)/SpacingNew/0*'

SaveFolder='C:/Users/linde/Documents/Crops(16x32x32)Utrecht/'



#set up dataset structure
luna_index = FilesIndex(path=path_data, dirs=True)      # preparing indexing structure
luna_dataset= ds.Dataset(index=luna_index, batch_class=CTICB)




def make_folder(folderlist=[]):
    for folder in folderlist:
        if not os.path.exists(folder):
            os.makedirs(folder)
         

def load_pipeline(nodules_df):
    pipeline=   (Pipeline()
                .load(fmt='blosc', components=['spacing', 'origin', 'images'])
                .fetch_nodules_info_Utrecht(nodules_df)
                .create_mask()) #creates mask component with nodules
    return  pipeline                      
                          
if CancerCropsMalignant == True:
    print('Starting Malignancy Cancer Crops')
    

    
    spacing=(2,1,1)
   
    
    cancer_cropline=(load_pipeline(nodules_utrecht)
                    .sample_nodules(batch_size=None, nodule_size=crop_size, share=(1.0),variance=(0,0,0),data='Utrecht') #variance is max total shift 
                    )
    #take for variance size of mini evaluation box 

    batch_size=1
    #loop multiple times to produce crops with different variations
    for j in range(Num_Cancer):
        print(j)
        #make new folder for each run
        make_folder([(SaveFolder+ 'novariation'),(SaveFolder+ 'novariation') ])
        
        #train & test lines
        cancer_train=(luna_dataset >> cancer_cropline.dump(dst=(SaveFolder+ 'novariation'), components=['origin','spacing',  'images','masks']))#          
       

       
        #cancer training set
        for i in range (math.ceil(len(luna_dataset)/batch_size)):
            try:
                batch=cancer_train.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1)
#                for index in range(len(batch.indices)):
#                   # np.save((SaveFolder+ 'novariation')+'/'+batch.indices[index]+'/nodules.npy', batch[index].nodules)
#                    if len(batch[index].nodules)==0:
#                            print('zerolengthnodule')
#                            print(batch.indices[index])
                print(i)
            except StopIteration:
                break    
    print('Ending Cancer Crops')     




print('All crops are made and saved for one run')



