# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 16:32:54 2018

@author: s120116
"""
import sys
sys.path.append('C:/Users/s120116/Documents/OneDrive - TU Eindhoven/TUE/Afstuderen/lung-nodule-msc-2018')
from radio.dataset import FilesIndex
from radio.dataset import Pipeline
import numpy as np
from radio import dataset as ds
import os
from CTImagesCustomBatch import CTImagesCustomBatch 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import CTsliceViewer as slices
import scipy 

LUNA_MASK = 'C:/Users/s120116/Documents/Preprocessed_Images/Crops(32x64x64)CancerwithMalignancy/subset0/training/cancer/cancer0/*' 
luna_index = FilesIndex(path=LUNA_MASK, dirs=True)      # preparing indexing structure
luna_dataset = ds.Dataset(index=luna_index, batch_class=CTImagesCustomBatch)

nodules_df = pd.read_excel('C:/Users/s120116/OneDrive - TU Eindhoven/TUE/Afstuderen/CSVFILES/all_info_averaged_observer.xlsx')
 
 
load_and_segment     = (Pipeline()
                            .load(fmt='blosc',components=['spacing', 'origin', 'images','masks'])
                            .loadMalignancy())

line=luna_dataset >> load_and_segment

def get_sample_from_batches(cbatch):

    cim=cbatch.unpack(component='images')
    clabel=cbatch.nodules.malignancy

    bin_clabel=np.where(clabel>=3, 1, 0)

    return cim, bin_clabel


batch=line.next_batch(batch_size=5)


ct=line.next_batch(batch_size=1)

ct=batch
center_pix =np.round(np.abs(batch.nodules.nodule_center -
                            batch.origin) /batch.spacing).astype(int)
nodule_size=np.round(ct.nodules.nodule_size)
for i in range(len(center_pix)):
    slice_nr=center_pix[i,0]
    fig, ax=plt.subplots(1)
   # plt.axis('off')
    ax.imshow(ct.images[slice_nr,:,:],'gray')
    size=nodule_size[i,1]
    
    rect=patches.Rectangle((center_pix[i,2]-size,center_pix[i,1]-size), size*2,size*2,edgecolor='red', facecolor='none')
    #plt.scatter(center_pix[i,2],center_pix[i,1])
    ax.add_patch(rect)


#save as png for powerpoint
im_array=ct.images
slices.multi_slice_viewer(im_array)
path='C:/Users/s120116/Pictures/CTslice/'
name='CTslice'
minim=np.amin(im_array)
maxim=np.amax(im_array)
for i in range(0,len(im_array),3):
    image=im_array[i,:,:]
    plt.imsave(path+name+str(i)+'.png',image,[minim,maxim],cmap='gray')
