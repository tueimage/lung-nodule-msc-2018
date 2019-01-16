# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 10:17:34 2018

@author: linde
"""
import matplotlib.pyplot as plt
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
import glob
import nistmodel as NIST

def seperate_cs_pe(im_60, im_190):

    keVin = [60., 190.] # input energies
    keVref = 70 # reference energy, take any value
    ray_pe = 1.0 # fraction of Rayleigh Scattering to include in PE component
    
    # calculate the energy scaling factors
    f_pe_ref = NIST._dPE('Water', keVref)+ray_pe*NIST._dRA('Water', keVref)
    f_cs_ref = NIST._dCS('Water', keVref)
    f_pe = [ NIST._dPE('Water', k)+ray_pe*NIST._dRA('Water', k) for k in keVin]
    f_cs = [ NIST._dCS('Water', k) for k in keVin]
    
    # only use relative factors
    f_pe = [ f/f_pe_ref for f in f_pe ]
    f_cs = [ f/f_cs_ref for f in f_cs ]
    
    # calculate mu_water for keVin
    mu_w = [ NIST._MU('Water', k) for k in keVin ]
    
    # solve pe, cs form Hi and Lo
    mulo = mu_w[0]*(im_60/1000. +1.)
    muhi = mu_w[1]*(im_190/1000. +1.)
    
    cs = (muhi-mulo*f_pe[1]/f_pe[0])/(f_cs[1]-f_cs[0]*f_pe[1]/f_pe[0])
    pe = (mulo-cs*f_cs[0])/f_pe[0]
    
    
    mu_water = NIST._MU('Water', 70)
    cs_hu = 1000.*(cs-mu_water)/mu_water
    
    pe_hu = 1000.*(pe-mu_water)/mu_water
    
    return cs_hu, pe_hu

def load_pipeline():
    pipeline=   (Pipeline()
                .load(fmt='dicom'))
    return  pipeline                      


fileList_060=[]
filelist_190=[]
for i in range(78,81): #from 1 to number of scans
        number=str(i).zfill(6) 
        path='D:/DATA20181008/' + number+'/'+ '060'
        if os.path.isdir(path)==True:
            fileList_060.append(path)#+ '/' + os.listdir(path)[0])
        path='D:/DATA20181008/' + number+'/'+ '190'
        if os.path.isdir(path)==True:
            filelist_190.append(path)#+ '/' + os.listdir(path)[0])    
            
                        
LUNA_pre='C:/Users/linde/Documents/CS_PE_seperatedtest' 

if not os.path.exists(LUNA_pre):
    os.makedirs(LUNA_pre)
    

#set up dataset structure

luna_index_low = FilesIndex(path=fileList_060,sort=True,dirs=True)      # preparing indexing structure
luna_dataset_low= ds.Dataset(index=luna_index_low, batch_class=CTICB) 


luna_index_high = FilesIndex(path=filelist_190,sort=True,dirs=True)      # preparing indexing structure
luna_dataset_high= ds.Dataset(index=luna_index_high, batch_class=CTICB) 



cancer_cropline=load_pipeline()

line_low=luna_dataset_low >> cancer_cropline
line_high= luna_dataset_high >> cancer_cropline


for i in range(len(luna_dataset_low)):
    if luna_dataset_high.index.indices[i] != luna_dataset_low.index.indices[i]:
        print('error!'+ ' high :'+ luna_dataset_high.index.indices[i] + ' low: '  + luna_dataset_low.index.indices[i] )

    low_im=line_low.next_batch(batch_size=1,shuffle=False)
    high_im=line_high.next_batch(batch_size=1, shuffle=False)
    
    im_60=low_im.images
    im_190=high_im.images
    
    
                             
    
    cs_hu, pe_hu =  seperate_cs_pe(im_60, im_190)
    
    
    
    
    #set low im batch to cs, and high im batch to pe
    low_im.images=cs_hu  
    high_im.images=pe_hu 
    
    low_im.unify_spacing(shape=(400,512,512), spacing=(2.0,1.0,1.0),padding='constant')
    high_im.unify_spacing(shape=(400,512,512), spacing=(2.0,1.0,1.0),padding='constant')
    
    #dump CS and PE in right folders
    low_im.dump(dst=LUNA_pre+'/CS/',components=['spacing', 'origin', 'images'])
    high_im.dump(dst=LUNA_pre+'/PE/',components=['spacing', 'origin', 'images'])
    
    print(i)





























#
#
#
#
#
##got now dicoom in HU, transform to mu
#im_60_mu=im_60*NIST._MU('Water', 60) / 1000 + NIST._MU('Water', 60)
#im_190_mu=im_60*NIST._MU('Water', 190) / 1000 + NIST._MU('Water', 190)
#
#
#
#
#Matrix=np.array([ [ NIST._MU('Water', 60),  NIST._MU('I', 60),NIST._MU('CaHa', 60), NIST._MU('Air', 60)],
#                  [ NIST._MU('Water', 190),  NIST._MU('I', 190),NIST._MU('Ca', 190),NIST._MU('Air', 190)],
#                  [1,1,1,1]]
#        
#                  )
#import scipy
#
#NMatrix=np.full([512,512, 3,3], Matrix)
#
#result=np.array([(im_60_mu).flatten(), (im_190_mu).flatten(), np.full([512*512],1)])
#resultN=np.swapaxes(result,0,1)
#
#new=np.zeros([512*512,3])
#for i in range(512*512):
#    new[i,:]=scipy.optimize.nnls(Matrix,resultN[i,:])[0]
#    print(i)
#    
#new_im=np.reshape(new,[512,512,3])
#
#
#new_im=np.swapaxes(new,0,2)
#
#new_im2=new_im[0]*NIST._MU('Water', 70)
#
#new_im_HU=1000.*(new_im2-NIST._MU('Water', 70)) / NIST._MU('Water', 70)
#plt.figure()
#plt.imshow(new_im[:,:,2])
