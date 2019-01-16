# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:43:10 2018

@author: s120116
This script can be used to get the heatmap from a CNN and a input
So basically it shows how the input pixels contribute to the final decision.
"""
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/s120116/OneDrive - TU Eindhoven/TUE/Afstuderen/lung-nodule-msc-2018')
import numpy as np
from CTImagesCustomBatch import CTImagesCustomBatch as CTICB
from radio import dataset as ds
from radio.dataset import Pipeline
from datetime import datetime
startTime = datetime.now()
import CTsliceViewer as slice
import seaborn as sns
import innvestigate.utils.visualizations as ivis
import innvestigate

#enter CNN load from name here
cnn = 'path'
path='C:/Users/s120116/Documents/Preprocessed_Images/Crops(16x32x32)CancerwithMalignancyAllSubsetsNew/subset*/training/cancer/cancer*/*'


def make_dataset(folder):
    index=ds.FilesIndex(path=folder,dirs=True)
    dataset=ds.Dataset(index=index,batch_class=CTICB)
    return dataset                
                        
pipeline_load= Pipeline().load(fmt='blosc', components=['spacing', 'origin', 'images','masks'])


cancer_trainset= make_dataset(path)
sample_cancer_train=(cancer_trainset >> pipeline_load)

cbatch=sample_cancer_train.next_batch(1, n_epochs=1, drop_last=True,shuffle=True)
cim=cbatch.unpack(component='images')

#load model
analyzer=innvestigate.create_analyzer("lrp.epsilon", cnn)
analysis=analyzer.analyze(cim)

analysis_viz=np.squeeze(analysis)

slice.multi_slice_viewer(cbatch.images)
#slice.multi_slice_viewer(analysis_viz)

X=ivis.gamma(analysis, minamp=0, gamma=0.5)
new=ivis.heatmap(X,cmap_type="seismic")


new2=sns.heatmap(X[0,8,:,:,0])
slice.multi_slice_viewer(new[0,:,:,:,:])
im=new[0,8,:,:,:]
plt.imshow(im,  vmin=0, vmax=1)

fig=plt.figure(frameon=False)
im1=plt.imshow(cbatch.images[8,:,:],cmap='gray')
im2=plt.imshow(im,alpha=0.6)
plt.show()
