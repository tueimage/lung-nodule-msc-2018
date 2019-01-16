# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 13:22:30 2018

@author: linde
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 14 10:53:16 2018

@author: s120116

In this file the lung is segmented from a CT image 
subsequently this volume is dilated to get border tissue as well
"""

import CTsliceViewer as slice
import matplotlib.pyplot as plt
import numpy as np # linear algebra
#import matplotlib.pyplot as plt
from skimage import measure
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2 
from scipy import ndimage




    

#does work but very slow, change of chrashing computer
def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
    


#get largest volume
def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]

    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None
    
    
    
    
#function to segment lung
 #segmented by treshold, and then closing the volume       
 

def new_seg(image):
    

    
    
    
    
    binary_image = np.array(image > -320, dtype=np.int8)#+1
    

    #
  
    kernel = np.ones((3,3), np.uint8) #was kernel(3,3)
    binary_image = cv2.morphologyEx(binary_image.astype(np.uint8),cv2.MORPH_CLOSE, kernel) #(this one before)
   
  #  binary_image=ndimage.binary_closing(binary_image.astype(np.uint8),kernel,iterations=1)
    
    labels = measure.label(binary_image,4) 
    slice.multi_slice_viewer(labels)  
    
    labels_bin=binary_image*labels
    tissue_label=largest_label_volume(labels_bin, bg=0)
    
    binary_image[labels!= tissue_label] = 0
    binary_image=binary_image+1
    
    
    
    labels = measure.label(binary_image,8) 
    
    #slice.multi_slice_viewer(labels)  
    
    bin_shape = binary_image.shape
    
    # create np.array of unique labels of corner pixels
    corner_labs = np.zeros(0)
    
    for z_ind in range(bin_shape[0]):
        corner_labs = np.append(corner_labs, labels[z_ind, 0, 0])
        corner_labs = np.append(
            corner_labs, labels[z_ind, bin_shape[1] - 1, 0])
        corner_labs = np.append(
            corner_labs, labels[z_ind, bin_shape[1] - 1, bin_shape[2] - 1])
        corner_labs = np.append(
            corner_labs, labels[z_ind, 0, bin_shape[2] - 1])
    
    #array with all possible corner values
    bk_labs = np.unique(corner_labs)
    
    # Fill the air around the person by making all groups with label equal to border pixels background
    for background_label in bk_labs:
        binary_image[background_label == labels] = 2
    #
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    
    
    
    # For every slice we determine the largest solid structure
    for i, axial_slice in enumerate(binary_image):
        axial_slice = axial_slice - 1
        labeling = measure.label(axial_slice)
        l_max = largest_label_volume(labeling, bg=0)
        
        if l_max is not None: #This slice contains some lung
            binary_image[i][labeling != l_max] = 1
    
    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
        
    return binary_image    





def total_segmentation(batch,dil_r=10):
    #make 4D array with masks and construct kernel
    kernel = np.ones((dil_r,dil_r,dil_r), np.uint8)   
  
    list_of_masks=[] #dilates segmentation
    list_of_segs=[] #segmentations
    
    #for each image make mask
    for i in range(len(batch)):
        #determine lung segmentation
        
        #this only for Utrecht data because here is head included
        numslice=35
        numslice2=319
        
        batch[i].images[:int(numslice),:,:]=np.zeros([int(numslice),512,512])
        batch[i].images[int(numslice2):,:,:]=np.zeros([batch.images.shape[0]-int(numslice2),512,512])
     
        #print('hoi')
        # ---------------------------
        
        seg_fill=new_seg(batch.get(i , 'images'))#, True) #get one by one images from batch and segment these
      
        #dilate this segmentation
        if np.count_nonzero(seg_fill) > 1000000: #check is segmentation is 'valid'
            im_dil=ndimage.morphology.binary_dilation(seg_fill,kernel).astype(np.uint8)
           
           # im_dil= seg_fill #(removed dilation for faster processing)
        else: #if not valid return array with ones --> no segmentation will be applied
           im_dil= np.ones_like(seg_fill)
          
           print('error in segmentation', batch.index.indices)
   
        list_of_masks.append(im_dil)
        list_of_segs.append(seg_fill)
            
    
    #concatenate array into 3D stacked array (corresponds to format of batch class)
    masks=np.concatenate(list_of_masks,axis=0)
    segs=np.concatenate(list_of_segs, axis=0)
    return masks, segs