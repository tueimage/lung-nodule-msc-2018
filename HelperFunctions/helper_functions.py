# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 11:35:28 2018

@author: s120116
"""
import numpy as np
import keras
from CTImagesCustomBatch import CTImagesCustomBatch as CTICB
from radio import dataset as ds
from radio.dataset import Pipeline

#function to create dataset from folder name
def make_dataset(folder):
    index=ds.FilesIndex(path=folder,dirs=True)
    dataset=ds.Dataset(index=index,batch_class=CTICB)
    return dataset      

def load_line():    
    return Pipeline().load(fmt='blosc', components=['spacing', 'origin', 'images'])

def load_line_folder(folder):
    dataset= make_dataset(folder)
    pipeline_load=load_line()
    sample_line=(dataset >> pipeline_load)
    return sample_line,dataset
