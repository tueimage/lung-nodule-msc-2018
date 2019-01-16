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
#from radio import CTImagesMaskedBatch as CTIMB
#names of folders in which images with segmentations are
sublist=['subset0', 'subset1', 'subset2', 'subset3', 'subset4','subset5', 'subset6', 'subset7']

#add subset 0 usually
CancerCropsMalignant= True
CancerCrops=False
RandomCrops=False
CandidateCrops=False
FalsePosCrops=False

Num_Cancer=20 #number of variations per nodule
crop_size=(16,32,32)

 #get nodule info
#nodules_df = pd.read_csv('C:/Users/s120116/OneDrive - TU Eindhoven/TUE/Afstuderen/CSVFILES/annotations.csv')
candidates_df = pd.read_csv('C:/Users/s120116/OneDrive - TU Eindhoven/TUE/Afstuderen/CSVFILES/candidates_v2.csv')
falsepositives_df = pd.read_excel('C:/Users/s120116/OneDrive - TU Eindhoven/TUE/Afstuderen/CSVFILES/FalsePositiveMiningList2.xlsx', sheet_name=0)
nodules_malignancy=pd.read_excel('C:/Users/s120116/OneDrive - TU Eindhoven/TUE/Afstuderen/CSVFILES/all_info_averaged_observer_corrected2.xlsx')



path='C:/Users/s120116/Documents/Preprocessed_Images/'
SaveFolder='Crops(16x32x32)CompleteDataset'

for sub in sublist:
    print(sub)
    
    #define folders in which validation and training data is
    LUNA_val=path + sub +' - split/validate/*' 
    LUNA_train= path+ sub +' - split/training/*' 
   
    
    #set up dataset structure
    luna_index_val = FilesIndex(path=LUNA_val, dirs=True)      # preparing indexing structure
    luna_dataset_val = ds.Dataset(index=luna_index_val, batch_class=CTICB)
    
    luna_index_train = FilesIndex(path=LUNA_train, dirs=True)      # preparing indexing structure
    luna_dataset_train = ds.Dataset(index=luna_index_train, batch_class=CTICB)
    
    
    def make_folder(folderlist=[]):
        for folder in folderlist:
            if not os.path.exists(folder):
                os.makedirs(folder)
             
    
    def load_pipeline(nodules_df):
        pipeline=   (Pipeline()
                    .load(fmt='blosc', components=['spacing', 'origin', 'images','segmentation'])
                    .fetch_nodules_info_malignancy(nodules_df)
                    .create_mask()) #creates mask component with nodules
        return  pipeline                      
                              
    if CancerCropsMalignant == True:
        print('Starting Malignancy Cancer Crops')
        
        cancer_folder_val=path+SaveFolder+'/'+sub+'/validate/cancer/cancer'
        cancer_folder_train=path+SaveFolder+'/'+sub+'/training/cancer/cancer'
        
        spacing=(2,1,1)
       # spacing_randomizer=lambda *args: tuple(random.uniform(0.8,1.2)* np.squeeze(spacing)) 
        
        cancer_cropline=(load_pipeline(nodules_malignancy)
                        .sample_nodules(batch_size=None, nodule_size=crop_size, share=(1.0),variance=(8,8,8)) #variance is max total shift 
                        )
        #take for variance size of mini evaluation box 
    
        batch_size=1
        #loop multiple times to produce crops with different variations
        for j in range(Num_Cancer):
            print(j)
            #make new folder for each run
            make_folder([(cancer_folder_val+ str(j)),(cancer_folder_train+ str(j)) ])
            
            #train & test lines
            cancer_train=(luna_dataset_train >> cancer_cropline.dump(dst=(cancer_folder_train+str(j)), components=['spacing', 'origin', 'images','masks']))#          
            cancer_val=(luna_dataset_val >> cancer_cropline.dump(dst=(cancer_folder_val+str(j)), components=['spacing', 'origin', 'images','masks']))

            #cancer validation set
            for i in range (math.ceil(len(luna_dataset_val)/batch_size)):
                try:
                    batch=cancer_val.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1)
                    for index in range(len(batch.indices)):
                        np.save((cancer_folder_val+str(j))+'/'+batch.indices[index]+'/nodules.npy', batch[index].nodules)
                except StopIteration:
                    break
           
            #cancer training set
            for i in range (math.ceil(len(luna_dataset_train)/batch_size)):
                try:
                    batch=cancer_train.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1)
                    for index in range(len(batch.indices)):
                        np.save((cancer_folder_train+str(j))+'/'+batch.indices[index]+'/nodules.npy', batch[index].nodules)
                except StopIteration:
                    break    
        print('Ending Cancer Crops')     
    
    if CancerCrops == True:
        print('Starting Cancer Crops')
        
        cancer_folder_val=path+SaveFolder+'/'+sub+'/validate/cancer_scaled/cancer'
        cancer_folder_train=path+SaveFolder+'/'+sub+'/training/cancer_scaled/cancer'
        
        spacing=(2,1,1)
        spacing_randomizer=lambda *args: tuple(random.uniform(0.8,1.2)* np.squeeze(spacing)) 
        
        cancer_cropline=(load_pipeline(nodules_malignancy)
                        .unify_spacing_withmask(spacing=spacing_randomizer(), shape=(400,512,512))
                        .sample_nodules(batch_size=None, nodule_size=crop_size, share=(1.0),variance=(8,8,8)) #variance is max total shift 
                        )
        #take for variance size of mini evaluation box 
    
        batch_size=1
        #loop multiple times to produce crops with different variations
        for j in range(Num_Cancer):
            print(j)
            #make new folder for each run
            make_folder([(cancer_folder_val+ str(j)),(cancer_folder_train+ str(j)) ])
            
            #train & test lines
            cancer_train=(luna_dataset_train >> cancer_cropline.dump(dst=(cancer_folder_train+str(j)), components=['spacing', 'origin', 'images','masks']))
            cancer_val=(luna_dataset_val >> cancer_cropline.dump(dst=(cancer_folder_val+str(j)), components=['spacing', 'origin', 'images','masks']))

            #cancer validation set
            for i in range (math.ceil(len(luna_dataset_val)/batch_size)):
                try:
                    batch=cancer_val.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1)
                   
                except StopIteration:
                    break
           
            #cancer training set
            for i in range (math.ceil(len(luna_dataset_train)/batch_size)):
                try:
                    batch=cancer_train.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1)
                
                except StopIteration:
                    break    
        print('Ending Cancer Crops')       
    
    if RandomCrops == True:
        print('Starting Random Crops')
        
        #make folders to save crops
        random_folder_val=path+SaveFolder+'/'+sub+'/validate/noncancer/random_bbox'
        random_folder_train=path+SaveFolder+'/'+sub+'/training/noncancer/random_bbox'
        make_folder([random_folder_val, random_folder_train])
        
        random_cropline=load_pipeline(nodules_malignancy).sample_nodules(batch_size=20, nodule_size=crop_size, share=(0.0)) 
   
        #train & test lines
        random_train=(luna_dataset_train >> random_cropline.dump(dst=random_folder_train, components=['spacing', 'origin', 'images','masks']))
        random_val=(luna_dataset_val >> random_cropline.dump(dst=random_folder_val, components=['spacing', 'origin', 'images','masks']))
        
        batch_size=1
        #random validation set
        for i in range (math.ceil(len(luna_dataset_val)/batch_size)):
            batch=random_val.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1)
           
          
       
        #random training set0
        for i in range (math.ceil(len(luna_dataset_train)/batch_size)):
            batch=random_train.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1)
           
         
        print('End of Random Crops')   


    if CandidateCrops == True:
        print('Starting Candidate Crops')
        
        #make folders to save crops
        candidate_folder_val=path+SaveFolder+'/'+sub+'/validate/noncancer/candidate'
        candidate_folder_train=path+SaveFolder+'/'+sub+'/training/noncancer/candidate'
        make_folder([candidate_folder_val, candidate_folder_train])
        
        candidate_cropline=(load_pipeline(nodules_malignancy)
                           .fetch_candidate_info(candidates_df)
                           .sample_candidates(batch_size=40, nodule_size=crop_size, type_cand='LunCand'))
        
    
   
        #train & test lines
        cand_train=(luna_dataset_train >> candidate_cropline.dump(dst=candidate_folder_train, components=['spacing', 'origin', 'images','masks']))
        cand_val=(luna_dataset_val >> candidate_cropline.dump(dst=candidate_folder_val, components=['spacing', 'origin', 'images','masks']))
        batch_size=1
        #random validation set
        for i in range (math.ceil(len(luna_dataset_val)/batch_size)):
            try:
                batch=cand_val.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1)
              
            except StopIteration:
                break
          
       
        #random training set0
        for i in range (math.ceil(len(luna_dataset_train)/batch_size)):
            try:
                batch=cand_train.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1)
                
            except StopIteration:
                break
         
        print('End of Candidate Crops')   


    if FalsePosCrops == True:
        print('Starting FP Crops')
         #make folders to save crops
       
        fp_folder_train=path+SaveFolder+'/'+sub+'/training/noncancer/falsepositives'
        make_folder([fp_folder_train])
        
        fp_cropline=  (load_pipeline(nodules_malignancy)
                        .fetch_candidate_info(falsepositives_df)
                        .sample_candidates(batch_size=10, nodule_size=crop_size, type_cand= 'FPred'))           
    

        #train & test lines
        fp_train=(luna_dataset_train >> fp_cropline.dump(dst=fp_folder_train, components=['spacing', 'origin', 'images','masks' ]))
      
        batch_size=1
        #random validation set
        
       
        #random training set0
        for i in range (math.ceil(len(luna_dataset_train)/batch_size)):
            try:
                batch=fp_train.next_batch(batch_size=batch_size, shuffle=False,n_epochs=1)
                print(i)
            except StopIteration:
                break
         
        print('End of FP Crops')   

    
    



print('All crops are made and saved for one run')



