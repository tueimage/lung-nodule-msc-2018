# -*- coding: utf-8 -*-
"""
Created on Fri May  4 14:39:50 2018

@author: s120116

This script gives the functions for a 3D slice viewer for CT images.
The  Final viewer is the multi_slice_viewer(), and can be used to vizualize a 3D array and scroll through the slices
with j and k
"""


import matplotlib.pyplot as plt
import numpy as np

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)
                
def multi_slice_viewer(volume, colormap='viridis'):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index], vmin=np.amin(volume), vmax=np.amax(volume), cmap=colormap)
    plt.title(ax.index)
    fig.canvas.mpl_connect('key_press_event', process_key)

def masks_images_viewer(image,mask,indexnum):
    remove_keymap_conflicts({'j', 'k'})
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.volume=image
    ax2.volume=mask
    
    ax1.index=image.shape[0] // 2
    ax2.index=mask.shape[0] // 2
    
    ax1.imshow(image[ax1.index])
    ax2.imshow(mask[ax2.index])
   
    fig.suptitle('Scan='+ str(ax1.index)+ '   Index=' + str(indexnum[ax1.index] ))
    fig.canvas.mpl_connect('key_press_event', lambda event: process_key_doubleax(event,indexnum))

def process_key_doubleax(event,indexnum):
    fig = event.canvas.figure
    ax1 = fig.axes[0]
    ax2= fig.axes[1]
    if event.key == 'j':
        title=previous_slice_index(ax1,indexnum)
        previous_slice_index(ax2,indexnum)
    elif event.key == 'k':
        title=next_slice_index(ax1,indexnum)
        next_slice_index(ax2,indexnum)
    fig.suptitle( title)
    fig.canvas.draw()
    
def previous_slice_index(ax,indexnum):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
    titlestring='Slice='+ str(ax.index)+ '   Index=' + str(indexnum[ax.index] )
    return titlestring

def next_slice_index(ax,indexnum):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    titlestring='Slice='+ str(ax.index)+ '   Index=' + str(indexnum[ax.index] )
    return titlestring




def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
    plt.title(ax.index)

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    plt.title(ax.index)