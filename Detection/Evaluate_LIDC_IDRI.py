# -*- coding: utf-8 -*-
"""

@author: s120116

This file uses the trained detection network to evaluate a new image in a sliding window fashion.
The scan is cut in 3D patches and each patch is given to the network, the outcome from this is than combined into a complete prediction.

The prediction map from the image is than compared with the actual annotations to obtain sensitivity / FP rate at different thresholds.

"""
import sys
sys.path.append('/home/lshesse')
#sys.path.append('C:/Users/s120116/OneDrive - TU Eindhoven/TUE/Afstuderen/lung-nodule-msc-2018')
import matplotlib
matplotlib.use('Agg')
import numpy as np
import keras
from CTImagesCustomBatch import CTImagesCustomBatch as CTICB
from radio import dataset as ds
from radio.dataset import Pipeline
from datetime import datetime
startTime = datetime.now()
import pandas as pd
import time
from skimage import measure
import os
import tensorflow as tf
import fnmatch
import keras.backend.tensorflow_backend


#configure GPU
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.tensorflow_backend.set_session(session)
start_time = time.clock()
 

#determine closest bounding box of image
def bbox2_3D(img):

  r = np.any(img, axis=(1, 2))
  c = np.any(img, axis=(0, 2))
  z = np.any(img, axis=(0, 1))

  rmin, rmax = np.where(r)[0][[0, -1]]
  cmin, cmax = np.where(c)[0][[0, -1]]
  zmin, zmax = np.where(z)[0][[0, -1]]

  return rmin, rmax, cmin, cmax, zmin, zmax



#pad the image with a correct number of zeros for the sliding window approach
def pad_for_Prediction(im, mask, segm,crop_size, step_size):
    left_pad=((crop_size-step_size)/2).astype(int) #ensures left upper corner can have patch surrounding it
    right_pad=(left_pad+crop_size).astype(int) #ensures right corner can have patch, also if center part is partly outside image
    
    bounding_im_pad=np.pad(im,((left_pad[0],right_pad[0]),(left_pad[1],right_pad[1]),(left_pad[2],right_pad[2])),mode='constant', constant_values=(170))
    bounding_mask_pad=np.pad(mask,((left_pad[0],right_pad[0]),(left_pad[1],right_pad[1]),(left_pad[2],right_pad[2])),mode='constant', constant_values=(0))
    bounding_segm_pad=np.pad(segm,((left_pad[0],right_pad[0]),(left_pad[1],right_pad[1]),(left_pad[2],right_pad[2])),mode='constant', constant_values=(0))
    return bounding_im_pad, bounding_mask_pad, bounding_segm_pad

#function to make the prediction map
def get_prediction_map(cnn,bounding_im_pad, prediction_map, step_size, crop_size,batch_size):
    
        patch_list=[] #list for patches, enabling batch predictions
        patch_coords = []
        for z in range(prediction_map.shape[0]):
            print('next slice')
            for y in range(prediction_map.shape[1]):
                for x in range(prediction_map.shape[2]):
                    
                    #extract patch
                    patch=bounding_im_pad[z*step_size[0]:z*step_size[0]+crop_size[0], y*step_size[1]:y*step_size[1]+crop_size[1],x*step_size[2]:x*step_size[2]+crop_size[2]]
                    
                    #add patch to current batch
                    if np.unique(patch).size > 2: #if only background patch does not have to be evaluated
                        patch_list.append(patch)
                        patch_coords.append((z,y,x))
             
            
                        if len(patch_list) % batch_size == 0: #if batch size has been reached, evaluate patches
                            batch_data=np.expand_dims(np.array(patch_list),4)
                            pred=cnn.predict(batch_data,batch_size=batch_size)
                            
                            for i in range(len(pred)): #for each prediction
                                p_z=patch_coords[i][0]
                                p_y=patch_coords[i][1]
                                p_x=patch_coords[i][2]
                                prediction_map[p_z,p_y,p_x]= pred[i] #assign result to correct location in prediction map
                            
                            patch_list=[]
                            patch_coords=[]
                            
        return prediction_map
    
    
 #convert prediction map to prediction image of same size as initial image
def get_prediction_image(bounding_im,prediction_map,step_size) :      
    prediction_im=np.zeros(bounding_im.shape)
    #cast predictions back to images, each prediction is for a mini_box (stepsize)
    for z in range(prediction_map.shape[0]):
        for y in range(prediction_map.shape[1]):
            for x in range(prediction_map.shape[2]):
                 prediction_im[z*step_size[0] : z  * step_size[0]+ step_size[0],y*step_size[1] : y  * step_size[1]+ step_size[1],x*step_size[2] :x  * step_size[2]+ step_size[2] ]= prediction_map[z,y,x]
    
    return prediction_im


#compare prediction image to actual predictions (bounding mask))        
def verify_predictions(prediction_im,bounding_mask,bounding_segm, treshold,FP_correction=True):
     #binarize prediction with certain threshold
      prediction_bin=np.zeros_like(prediction_im)
      prediction_bin[prediction_im < treshold] = 0
      prediction_bin[prediction_im >= treshold] = 1
      correct_labels=[]
      TrueDetected=0
      MissedDetection=0
      label_prediction=measure.label(prediction_bin)
      label_masks=measure.label(bounding_mask)
      for i in np.unique(label_masks)[1:]: #dont take label 0, iterate over known nodules
          if any (prediction_bin[label_masks==i]  ==1): # if a nodule (in mask), check whether there is overlap with prediction
                lab=label_prediction[label_masks==i] #number of detection label
                for j in np.unique(lab):
                    if j>0:
                       correct_labels.append(j) 
                        
                TrueDetected = TrueDetected + 1
                
          else:
                MissedDetection= MissedDetection + 1
      number_detection=np.trim_zeros(np.unique(label_prediction))
      FalsePositive=[x for x in number_detection if x not in correct_labels]  
      FP_copy=FalsePositive
      if FP_correction==True:
          for i in FalsePositive: #iterate over each of false positve labels
              if any (bounding_segm[label_prediction==i]==1): #if there is overlap with an irrelevant finding
                  FP_copy.remove(i)
                  
      fp_num=len(FP_copy)            
          
      return TrueDetected, MissedDetection,fp_num
       

def calc_Sensitivity(TrueDetected,MissedDetection):
        
        if TrueDetected+MissedDetection == 0:
            Sensitivity=1
        else:
            Sensitivity=TrueDetected/ (TrueDetected + MissedDetection)
        return Sensitivity
      
def FPmining(detections, correct_labels, batch, zmin,ymin,xmin, FPminingList):
    #do FP mining
    for i in range(len(detections)):
        if (detections[i].label  not in correct_labels):
            coords=np.array(detections[i].centroid)+[zmin,ymin,xmin]
            FPminingList.append([''.join(batch.index.indices), coords[0],coords[1],coords[2]]) #realize coordinates with bounding box
    return FPminingList 


    
# function to combine everything    
def eval_on_images(path,cnn, nodules_df, nodules_eval, crop_size = np.array([16,32,32]), step_size = np.array([8,8,8]),FPminingNeed=False,saveImages=False):

    #get data
    luna_index_test = ds.FilesIndex(path=path, dirs=True)      # preparing indexing structure
    luna_dataset_test = ds.Dataset(index=luna_index_test, batch_class=CTICB)
        

    #replace negative diameters in irrelevant findings with small diamter of 3mm   
    nodules_eval['diameter_mm']=nodules_eval['diameter_mm'].replace(-1,3)
    
    #this pipeline does the preprocessing and gets the ground truth for the image
    preprocessing	       =     (Pipeline()
                                  .load(fmt='blosc', components=['spacing', 'origin', 'images','segmentation']) 
                                  .fetch_nodules_info(nodules_df)
                                  .create_mask()
                                  .fetch_nodules_info(nodules_eval ,update=True))
    
    preprocess_line=(luna_dataset_test >> preprocessing) 
    
    #possible thresholds
    treshold_list=[  0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97,0.99,0.995,0.998,0.999, 0.9995,0.9998, 0.9999,1]


    FalsePositiveList=np.zeros([len(luna_dataset_test, ), len(treshold_list)])
    SensitivityList=np.zeros([len(luna_dataset_test, ), len(treshold_list)])
    TrueDetectedList=np.zeros([len(luna_dataset_test, ), len(treshold_list)])
    MissedDetectedList=np.zeros([len(luna_dataset_test, ), len(treshold_list)])
    FPminingNeed=False
    FPminingList=[]
    
    #define batch size in which prediction should b emakde
    batch_size=20
    
    folder='Image_Data'
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    folder_files='Image_evaluation'   
    if not os.path.exists(folder_files):
        os.makedirs(folder_files)                 
    index_list=[]
    
    for k in range(len(luna_dataset_test)):
        start_time = time.clock()
        batch=preprocess_line.next_batch(batch_size=1)
        im_index=batch.indices
        index_list.append(str(im_index))
        
        #crop images to bounding box to not classify to much
        zmin,zmax,ymin,ymax,xmin,xmax=bbox2_3D(batch.segmentation)
        segmentation=batch.segmentation[zmin:zmax,ymin:ymax,xmin:xmax] #extract segmentation
    

        batch.create_mask_irrelevant() #when segmentation is no longer needed, put irrelevant findings into this component
        
        #cut all images to they bounding box of segmentation to reduce computational load
        bounding_im=batch.images[zmin:zmax,ymin:ymax,xmin:xmax]
        bounding_mask=batch.masks[zmin:zmax,ymin:ymax,xmin:xmax]
        bounding_segm=batch.segmentation[zmin:zmax,ymin:ymax,xmin:xmax]
        
        
        #padd the images to ensure correct patch extraction at boundaries
        bounding_im_pad,bounding_mask_pad,bounding_segm_pad=pad_for_Prediction(bounding_im, bounding_mask,bounding_segm, crop_size,step_size)
        
        #make empty array for prediction
        size=bounding_im.shape
        prediction_size=np.ceil(size/step_size).astype(int) #make sure all pixels got a mini-box
        prediction_map=np.zeros(prediction_size)
        
        start_pred_time=time.clock()
            
        #get prediction map of image
        prediction_map=get_prediction_map(cnn,bounding_im_pad, prediction_map, step_size, crop_size,batch_size)
        #cast prediction map to same size as prediction image
        prediction_im=get_prediction_image(bounding_im, prediction_map, step_size)
        
        prediction_im=prediction_im * segmentation #all predictions outside segmentation are not relevant
        #save predicted images
        if saveImages==True:
           np.save(folder+'/'+  'prediction_im'+str(k), prediction_im)
           np.save(folder+'/'+  'bounding_im'+str(k), bounding_im)
           np.save(folder+'/'+ 'bounding_mask'+str(k), bounding_mask)
           np.save(folder+'/'+  'bounding_irrel'+str(k), bounding_segm)
        
          #determine of predictions are correct
        for i in range(len(treshold_list)):
           treshold= treshold_list[i]
           TrueDetected, MissedDetected, FalsePositive=verify_predictions(prediction_im, bounding_mask,bounding_segm, treshold)
           Sensitivity=calc_Sensitivity(TrueDetected,MissedDetected)
           SensitivityList[k,i]=Sensitivity
           FalsePositiveList[k,i]=FalsePositive
           TrueDetectedList[k,i]=TrueDetected
           MissedDetectedList[k,i]=MissedDetected  
    
    
    

    
    #do false positive mining if wanted
    
#        if FPminingNeed==True:    
#           detections=measure.regionprops(label_prediction)
#           FPminingList=FPmining(detections, correct_labels, batch, zmin,ymin,xmin,FPminingList)


    
    
        
        print( (time.clock()-start_pred_time)/60)
        print("--- %s minutes ---" % ((time.clock() - start_time)/60))
     
    #determine number of detected / missed nodules
    cor_tresh=np.sum(TrueDetectedList,0)
    mis_tresh=np.sum(MissedDetectedList,0)
    sens_tresh=cor_tresh/(cor_tresh+mis_tresh)
    fp_tresh=np.mean(FalsePositiveList,0)   
    
    #save all files
    np.save(folder_files + '/FPlist',FalsePositiveList  )
    np.save(folder_files + '/SensList', SensitivityList)
    np.save(folder_files + '/TrueDetected',TrueDetectedList    )
    np.save(folder_files + '/MissedDetected', MissedDetectedList) 

    np.save(folder_files + '/sens_tresh',sens_tresh)
    np.save(folder_files + '/fp_tresh',fp_tresh)

    #writer seriesuid to excel file
    df=pd.DataFrame({'series':index_list})
    writer = pd.ExcelWriter(folder_files+'/seriesUID.xlsx', engine='xlsxwriter')
    df.to_excel(writer, index=False,header=True)

    #writer seriesuid to excel file
    df_2=pd.DataFrame({'treshold':treshold_list, 'sensitivity':sens_tresh, 'false positives': fp_tresh})
    writer = pd.ExcelWriter(folder_files+'/FROC.xlsx', engine='xlsxwriter')
    df_2.to_excel(writer, index=False,header=True)

    #calculate overall sensitivy and fp_rate
    total_correct_detected=np.sum(TrueDetectedList,0)
    total_nodules=total_correct_detected + np.sum(MissedDetectedList,0)
    Sensitivity=np.divide(total_correct_detected,total_nodules)
    fp_rate=np.mean(FalsePositiveList,0)
    
    
    #save sensitivity to txt file
    names  = np.array(['Sensitivity:','FalsePositives:'])
    floats = np.array([Sensitivity[8], fp_rate[8] ])
    
    ab = np.zeros(names.size, dtype=[('var1', 'U20'), ('var2', float)])
    ab['var1'] = names
    ab['var2'] = floats
    
    np.savetxt(folder_files + '/accuracy.txt', ab, fmt="%18s %10.3f")
    
    
    
    
    
    #write false positive list to file
    if FPminingNeed==True:
        a=pd.DataFrame(FPminingList)
        writer = pd.ExcelWriter(folder_files+'/FalsePositiveMiningList.xlsx', engine='xlsxwriter')
        a.columns = ["seriesuid", "coordX", "coordY", "coordZ"]
        a.to_excel(writer, index=False,header=True)

#do evaluation on the combination of large and small patch size, which have already been evaluated individually
def eval_2models(folder1, folder2): 

    numIm1= len(fnmatch.filter(os.listdir(folder1+ '/Image_Data'), 'bounding_im*.npy'))
    numIm2= len(fnmatch.filter(os.listdir(folder2+'/Image_Data'), 'bounding_im*.npy'))
    if numIm1 != numIm2:
        raise ValueError('number of analyzed images is not the same for both models')
     
  

    df_series1 = pd.read_excel(folder1+'/Image_evaluation/'+'seriesUID.xlsx', sheet_name=0) # can also index sheet by name or fetch all sheets
    seriesUID_1 = df_series1['series'].tolist()
    
    df_series2 = pd.read_excel(folder2+'/Image_evaluation/'+'seriesUID.xlsx', sheet_name=0) # can also index sheet by name or fetch all sheets
    seriesUID_2 = df_series2['series'].tolist()
    

       
    folder='Image_Data'
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    folder_files='Image_evaluation'   
    if not os.path.exists(folder_files):
        os.makedirs(folder_files) 
        
    treshold_list=[ 0.1,0.2,0.3,0.4,0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97,0.99,0.995,0.998,0.999,0.9995,1]

    FalsePositiveList=np.zeros([numIm1, len(treshold_list)])
    SensitivityList=np.zeros([numIm1 , len(treshold_list)])
    TrueDetectedList=np.zeros([numIm1, len(treshold_list)])
    MissedDetectedList=np.zeros([numIm1, len(treshold_list)])

    
    for k in range(numIm1):
        print(k)
        series_num=seriesUID_1[k]
        index2=seriesUID_2.index(series_num)
        
        prediction_im_mod1=np.load(folder1+ '/Image_Data'+'/prediction_im'+str(k)+'.npy')
        prediction_im_mod2=np.load(folder2+ '/Image_Data'+'/prediction_im'+str(index2)+'.npy')
        
        bounding_mask = np.load(folder1+'/Image_Data'+'/bounding_mask'+str(k)+'.npy')
        bounding_segm = np.load(folder1+'/Image_Data'+'/bounding_irrel'+str(k)+'.npy') #see whether this is _irr or_segm
        bounding_im=np.load(folder1+ '/Image_Data'+'/bounding_im'+str(k)+'.npy')
        
    
        prediction_im=(prediction_im_mod1+prediction_im_mod2 )/2 #average predictions
        
             #save predicted images
        np.save(folder+'/' + 'prediction_im'+str(k), prediction_im)
        np.save(folder+'/'+  'bounding_im'+str(k), bounding_im)
        np.save(folder+'/'+ 'bounding_mask'+str(k), bounding_mask)
        np.save(folder+'/'+ 'bounding_irr'+str(k), bounding_segm)
        
        for i in range(len(treshold_list)):
           treshold= treshold_list[i]
           TrueDetected, MissedDetected, FalsePositive=verify_predictions(prediction_im, bounding_mask,bounding_segm, treshold)
           Sensitivity=calc_Sensitivity(TrueDetected,MissedDetected)
           SensitivityList[k,i]=Sensitivity
           FalsePositiveList[k,i]=FalsePositive
           TrueDetectedList[k,i]=TrueDetected
           MissedDetectedList[k,i]=MissedDetected
     

 
     #save all files
    np.save(folder_files + '/FPlist',FalsePositiveList  )
    np.save(folder_files + '/SensList', SensitivityList)
    np.save(folder_files + '/TrueDetected',TrueDetectedList    )
    np.save(folder_files + '/MissedDetected', MissedDetectedList) 
 
    #np load image by image from folder
    #writer seriesuid to excel file
    writer = pd.ExcelWriter(folder_files+'/seriesUID.xlsx', engine='xlsxwriter')
    df_series1.to_excel(writer, index=False,header=True)
    
    cor_tresh=np.sum(TrueDetectedList,0)
    mis_tresh=np.sum(MissedDetectedList,0)
    sens_tresh=cor_tresh/(cor_tresh+mis_tresh)
    fp_tresh=np.mean(FalsePositiveList,0)   
    
    np.save(folder_files + '/sens_tresh',sens_tresh)
    np.save(folder_files + '/fp_tresh',fp_tresh)
    

def main():
    crop_size = np.array([16,32,32]) #crop size
    cnn = keras.models.load_model('/home/lshesse/evaluate18-07/Correct_Scripts/Correct_Crops[16,32,32]/TrainingDataneuralnet_final')
    path_images='/home/lshesse/Datasets/Preprocessed_Images/subset* - split/validate/*' 
    nodules_eval=pd.read_csv('/home/lshesse/annotations_excluded.csv') #irrelevant findings
    nodules_df = pd.read_csv('/home/lshesse/annotations.csv')
    eval_on_images(path_images ,cnn, nodules_df, nodules_eval,crop_size = crop_size, step_size = np.array([8,8,8]),FPminingNeed=False)



    #to evaluate on both scales
    # enter folders where evalauted images on individual scales are located
    path='/ '
    folder1=path+'Run1/evalTestImages'
    folder2=path+'Run32x64x64Run12/Train6000/EvalAndSave'

    eval_2models(folder1,folder2)
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
