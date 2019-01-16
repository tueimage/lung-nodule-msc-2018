# -*- coding: utf-8 -*-
"""
Created on Thu May 24 09:36:22 2018

@author: s120116
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  7 10:26:00 2018

@author: s120116
This script is used to train a CNN for the detection of lung nodules. As training input nodule and non-nodule crops are used.

"""
import sys
sys.path.append('/home/lshesse')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import keras
from CTImagesCustomBatch import CTImagesCustomBatch as CTICB
from radio import dataset as ds
from radio.dataset import Pipeline
from datetime import datetime
startTime = datetime.now()
import CNNfile_detection as model
#import CTsliceViewer as slice
import matplotlib.patches as mpatches
import os
import pandas as pd
import tensorflow as tf
import keras.backend.tensorflow_backend
import Evaluate_LIDC_IDRI as im_eval

#configure GPU
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.tensorflow_backend.set_session(session)


#input
val_images=True #define whether validation on whole test set should be performed after training
LUNA_test='/home/lshesse/Datasets/Preprocessed_Images/subset* - split/validate/*' 
nodules_df = pd.read_csv('/home/lshesse/annotations.csv') #nodule annotations
nodules_eval=pd.read_csv('/home/lshesse/annotations_excluded.csv') #irrelevant findings

# Define folders containing train and validation crops
path='/home/lshesse/Datasets/Crops(32x64x64)/'

#define whether augmentated samples should be used
possible_flips=['frontback', 'leftright', 'noflips', 'updown']
cancer_folder=[]
for i in possible_flips:
    path_train=path+'augmentedCropsFlip2/training/cancer/'+ i + '/*' 
    cancer_folder.append(path_train)


val_cancer_folder=path+'subset*/validate/cancer/*/*'
ncancer_folder=path+'subset*/training/noncancer/*/*'
val_ncancer_folder=path+'subset*/validate/noncancer/*/*'

#define the crop size
crop_size=np.array([32,64,64])

#load CNN
cnn=model.get_net_detection(input_shape=(32,64,64,1))


#%%------------------------

current_folder_path, testname = os.path.split(os.getcwd())

savepath='TrainingData'
#makes folder for all savings
if not os.path.exists(savepath):
    os.makedirs(savepath)
    

#function to create dataset from folder name
def make_dataset(folder):
    index=ds.FilesIndex(path=folder,dirs=True)
    dataset=ds.Dataset(index=index,batch_class=CTICB)
    return dataset                
                        
#
#def augmentdata(self): #for now, whole batch same spacing change, each scan 50% change to be flipped
#    
#    #scale whole batch with factor between 0.8 and 1.2
#    spacing=self.get(0,'spacing')
#    spacing_randomizer=lambda *args: tuple(random.uniform(0.8,1.2)* np.squeeze(spacing)) 
#    
#    self.unify_spacing_withmask(spacing=spacing_randomizer(), shape=(self.get(0,'images').shape))
#    
#    #flip left right for each of the scans in the batch seperately
#    for i in range(len(self)):
#        if np.random.choice(np.arange(2)) == 0: #50% change of flip batch
#            self[i].images=np.flip(self[i].images,2)
#            self[i].masks=np.flip(self[i].masks,2)
#         
#    return self
#



#function to make a sample for training from two half batches (positve and negative)
def get_sample_from_batches(cbatch,ncbatch):

    cim=cbatch.unpack(component='images')
    
    ncim=ncbatch.unpack(components='images')
    newIm=np.vstack((cim,ncim))

    clabel=cbatch.classification_targets(6)
    nclabel=ncbatch.classification_targets(6)
    newLab=np.vstack((clabel,nclabel)) #get array of ((#im, x,y,z,1))
    
    
    #shuffle them equally to get shuffled batch
    s=np.arange(newLab.shape[0])
    np.random.shuffle(s)
    Lab=newLab[s]
    Im=newIm[s]
    return Im, Lab 

def validate_on_samples(sample_cancer_train_eval,sample_noncancer_train_eval, eval_size, batchsize):
    eval_loss= np.empty([eval_size,1])
    for j in range(eval_size): #test on batch for number of batches defined in eval_size
        
        cbatch_train=sample_cancer_train_eval.next_batch(batchsize, n_epochs=None, drop_last=True,shuffle=True)
        ncbatch_train=sample_noncancer_train_eval.next_batch(batchsize,n_epochs=None, drop_last=True,shuffle=True)
     
        im_train,lab_train= get_sample_from_batches(cbatch_train,ncbatch_train)
        train_loss=cnn.test_on_batch(im_train,lab_train)
        eval_loss[j]=train_loss
    loss=np.mean(eval_loss)
    return loss

def save_train_test_loss(losslist,test_losslist, val_freq ):
    x_test=list(range(0,len(test_losslist)*val_freq,val_freq))
    x_train=list(range(0,len(losslist)*val_freq,val_freq))
    plt.figure()
    plt.plot(x_test, test_losslist, 'r-', x_train, losslist, 'b-' )
    red_patch = mpatches.Patch(color='red', label='Validation loss')
    blue_patch = mpatches.Patch(color='blue', label='Training loss')
    plt.legend(handles=[red_patch,blue_patch])
    plt.title(testname)
    plt.ylim((0,0.5))
    plt.savefig(savepath+'/Losses.png')

#create datasets voor cancer/noncancer and training/testing
cancer_testset= make_dataset(val_cancer_folder)
ncancer_testset= make_dataset(val_ncancer_folder)
cancer_trainset= make_dataset(cancer_folder)
ncancer_trainset= make_dataset(ncancer_folder)                        


#make lists for the losses
losslist = []
test_losslist=[]


# create pipeline to load images and give dataset structures to pipeline
pipeline_load= Pipeline().load(fmt='blosc', components=['spacing', 'origin', 'images','masks'])


#get training and testing pipelines with data
sample_cancer_train=(cancer_trainset >> pipeline_load)
sample_noncancer_train=(ncancer_trainset >> pipeline_load)

sample_cancer_test=(cancer_testset >> pipeline_load)
sample_noncancer_test=(ncancer_testset >> pipeline_load)

#use seperate pipeline for evaluation to make sure all images are used for training itself
sample_cancer_train_eval=(cancer_trainset >> pipeline_load)
sample_noncancer_train_eval=(ncancer_trainset >> pipeline_load)



#training parameters
n_epochs=2 #number of epochs for cancer training set, others continue untill this one has finished
cancer_batchsize = 10
ncancer_batchsize = 10 #total batch size is cancer + ncancer batchsize

epoch_iter=np.round(len(cancer_trainset)/cancer_batchsize) #number of iterations per epoch
save_freq_CNN=5 #save network every nth epoch
save_freq_plots=5 #save plots every nth epoch
val_freq=10 #validae every nth iteration
eval_size=5
#train network
i=0
Iterations=8000 #nr of iterations for one epoch for all scales
changed=False #become true if learning rate is increased


#start of actual training
for i in range(Iterations):#int(np.around(len(cancer_trainset)*n_epochs/cancer_batchsize))):
    try:
        #generate training samples
        cbatch=sample_cancer_train.next_batch(cancer_batchsize, n_epochs=None, drop_last=True,shuffle=True)
        ncbatch=sample_noncancer_train.next_batch(ncancer_batchsize,n_epochs=None, drop_last=True,shuffle=True)
       
    
        im_train,lab_train = get_sample_from_batches(cbatch,ncbatch)
      
        #train on training samples and append loss
        cnn.train_on_batch(im_train,lab_train)
      
        
        #validate every 10 iterations
        if i % val_freq == 0:
            
            #do evaluation on training set            
            train_loss= validate_on_samples(sample_cancer_train_eval,sample_noncancer_train_eval, eval_size, cancer_batchsize) #final eval size= eval_size*batchsize
            losslist.append(train_loss)
            
            #do evaluation on validation set
            val_loss=validate_on_samples(sample_cancer_test, sample_noncancer_test, eval_size, cancer_batchsize)
            test_losslist.append(val_loss)
       
            print('Batch: {}'.format(i))
            print('Loss: {}'.format(train_loss))
            print('TestLoss: {}'.format(val_loss)) 
        

        if i % (4000) == 0: #save CNN every nth epoch with different name

            cnn.save(savepath+'/neuralnetVGG_inter'+ str(i))
        
       # if i % (epoch_iter * save_freq_plots) == 0:   #save plots every nth iteration, overwrite preiouvs ones
        #    save_train_test_loss(losslist, test_losslist, val_freq)
        
        
        #after n epochs, decrease learning rate with factor 10
       # if n_epoch == 5 and changed==False:
        #    K.set_value(cnn.optimizer.lr, K.get_value(cnn.optimizer.lr) *0.1)
         #   changed=True 
        #if i% epoch_iter ==0:
         #   K.set_value(cnn.optimizer.lr, K.get_value(cnn.optimizer.lr)*0.5)
        i=i+1
        print(i)
          
    except StopIteration: #prevent stopping program when n_epchs has been reached
        print('End of training: Dataset has been iterated number of defined epochs')
        break
    
   

#save list of losses and trained network in same folder as file
np.savetxt(savepath+'/losslist_train.csv', losslist, delimiter=',')
np.savetxt(savepath+'/losslist_val.csv', test_losslist, delimiter= ',')
cnn.save(savepath+'/neuralnet_final')
#cnn.save_weights(savepath+ '/my_model_weights.h5')


#make and save figure of losses
save_train_test_loss(losslist, test_losslist, val_freq)

#make plot from radio tutorial code
loss_history_val=pd.Series(test_losslist).rolling(60).mean()
loss_history_train=pd.Series(losslist).rolling(60).mean()
plt.figure()
loss_history_val.plot(grid=True, color='red')
loss_history_train.plot(color='blue')
red_patch = mpatches.Patch(color='red', label='Validation loss')
blue_patch = mpatches.Patch(color='blue', label='Training loss')
plt.ylim((0,0.5))
plt.legend(handles=[red_patch,blue_patch])
plt.savefig(savepath+'/AveragedLosses.png')




#
#----- determine accuracy of train set by labeling all images in train and validation set

def get_predictions(sample_cancer_train, cancer_trainset, cancer_batchsize):
    predictlist=[]
    labellist=[]
    for i in range(int(np.ceil(len(cancer_trainset)/cancer_batchsize))):
            batch=sample_cancer_train.next_batch(cancer_batchsize, n_epochs=1, drop_last=False,shuffle=False)
            im_train=batch.unpack(components='images')
            
            lab_know=batch.classification_targets(6) #check known label
            lab_predict= cnn.predict(im_train) #predict label
                 
            predictlist.append(lab_predict[:,0].tolist()) #append labels to list
            labellist.append(lab_know[:,0].tolist())
    return predictlist, labellist

def get_accuracy(predictlist, labellist):
#flatten out lists to be able to compare them
    predict_list_flat = [item for sublist in predictlist for item in sublist]
    label_list_flat = [item for sublist in labellist for item in sublist]
    predict_bin=np.around(np.array(predict_list_flat))

    mistake=predict_bin-np.array(label_list_flat)
    accuracy=1-((np.count_nonzero(mistake))  / len(predict_bin))
    return accuracy



#use seperate pipeline for evaluation to make sure all images are used for training itself
sample_cancer_train_eval=(cancer_trainset >> pipeline_load)
sample_noncancer_train_eval=(ncancer_trainset >> pipeline_load)


#construct all dataset pipelines again
sample_cancer_train=(cancer_trainset >> pipeline_load)
sample_noncancer_train=(ncancer_trainset >> pipeline_load)

sample_cancer_val=(cancer_testset >> pipeline_load)
sample_noncancer_val=(ncancer_testset >> pipeline_load)

#label all cancer/noncancer crops from the train set
predictlist_pos_train, labellist_pos_train=get_predictions(sample_cancer_train, cancer_trainset, cancer_batchsize)
predictlist_neg_train, labellist_neg_train= get_predictions(sample_noncancer_train, cancer_trainset, cancer_batchsize)
      
#calculate accuracies for trainset
accuracy_train_pos=get_accuracy(predictlist_pos_train,labellist_pos_train)
accuracy_train_neg=get_accuracy(predictlist_neg_train,labellist_neg_train)

#label all cancer/noncancer crops from the validation set
predictlist_pos_val, labellist_pos_val=get_predictions(sample_cancer_val, cancer_testset, cancer_batchsize)
predictlist_neg_val, labellist_neg_val= get_predictions(sample_noncancer_val, cancer_testset, cancer_batchsize)

#calculate accuracies for validation set
accuracy_val_pos=get_accuracy(predictlist_pos_val,labellist_pos_val)
accuracy_val_neg=get_accuracy(predictlist_neg_val,labellist_neg_val)

        
    
#save accuracies to .txt -------------------------------------------------------------------------------
#names  = np.array(['Train Accuracy_pos:','Train_Accuracy_neg:','Val_Accuracy_pos:',  'Val_Accuracy_neg:'])
#floats = np.array([ accuracy_train_pos, accuracy_train_neg , accuracy_val_pos, accuracy_val_neg ])

#ab = np.zeros(names.size, dtype=[('var1', 'U20'), ('var2', float)])
#ab['var1'] = names
#ab['var2'] = floats

#np.savetxt(savepath+'/accuracy.txt', ab, fmt="%18s %10.3f")
np.save(savepath+'/trainloss', losslist)
np.save(savepath+'/testloss', test_losslist)



# ---------------------------------------------------------------------------------------------
#validate on whole images
if val_images== True:
   im_eval.eval_on_images(LUNA_test,cnn, nodules_df,nodules_eval,crop_size=crop_size,saveImages=False)





