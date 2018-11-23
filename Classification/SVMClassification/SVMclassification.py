# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 10:28:32 2018

@author: linde
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:56:17 2018

@author: s120116
"""

from sklearn import preprocessing
import pandas as pd
from radio import dataset as ds

import numpy as np
import pickle
import os
#from mpl_toolkits.mplot3d import Axes3D
from HelperFileClassification import load_features, optimize_and_cv, get_predictions_cv
from sklearn import svm
# enter path to diagnosis file 
# diagnosis file should hold number of scan (should correspond to numbers of folder names, label: 'benigne', 'lung', 'metastases'), and a patientnumber ( for if two scans 
# are from one patient, if this is not the case they can have same number as scan number (every scan needs to have different number))

#Input -----------
diagnosis_path = 'C:/Users/linde/OneDrive - TU Eindhoven/TUE/Afstuderen/CSVFILES/batchLablesLinde2.csv'#excel file with diagnosis

# -----------------------------------------

SVMpath1= '../Models/SVM_fit_bm.pickle'
SVMpath2= '../models/SVM_fit_blungmet.pickle'

diagnosis_df = pd.read_csv(diagnosis_path)
diagnosis_df=diagnosis_df.set_index('outfolder')

def bin_labels(labels):
    labels_bin=np.zeros(len(labels))
    for i in range(len(labels)):
        if labels[i] == 'benige' or  labels[i] == 'benigne multinodair' :
            labels_bin[i]=0
        if labels[i] == 'long'   :
            labels_bin[i]=1 
        if labels[i] == 'crc'   or  labels[i] == 'melanoom'  :
            labels_bin[i]=2      
        
    return labels_bin

#load all data
savepath = '../Final_Results'  
if not os.path.exists(savepath):
    os.makedirs(savepath)
    
    
features_folder='../../../NoduleFeatures/*'   
#features_folder='../../../ResultingData/NoduleFeatures/*'
index_train=ds.FilesIndex(path=[features_folder],dirs=True,sort=True)




#load the featurers, labels, groups and indices in memory
# it assumes now that the folder name of each feature vector consists of a number with trailing zeros (untill 6 elements), plus nodule number, so it takes the first 6
# elements and converts this to a number. If the naming is differently please adapt this in the load_features function in HelperFileClassificaiton
features_orig, labels_orig ,groups_orig, indices_orig =load_features(index_train,diagnosis_df)

#do preprocessing of feature vectors
features_norm=preprocessing.normalize(features_orig) 

#binarize labels
labels_bin=bin_labels(labels_orig)

#dataset 1: Benign / Malignant, 
labels_bm=np.where(labels_bin>0, 1,0)

#dataset2: all data, benign / metastasen / lung
labels_bmetlung=labels_bin



#determine C
Clist={'C':[0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000]}


#perform SVM classification for both class divisions, benign vs malignant and benign vs lung vs metastases
# first for new fit

#do both tests for single nodules =-----------------------------------------------------------------------------
print('Start Single Nodule Scoring')
print('Start test1')
f1micro, f1macro, Cnum, d_fin, pvalue= optimize_and_cv(features_norm, labels_bm, groups_orig, Clist,permut=False)
Test1Dict={}
Test1Dict['f1_micro_cv']=d_fin['test_accuracy']
Test1Dict['f1_macro_cv']=macroCV=d_fin['test_f1macro']
Test1Dict['pvalue']=pvalue
Test1Dict['Cnum']=Cnum
Test1Dict['f1micro']=f1micro
Test1Dict['f1macro']=f1macro


print('Start test2')
f1micro, f1macro, Cnum, d_fin, pvalue= optimize_and_cv(features_norm, labels_bmetlung, groups_orig, Clist,permut=False)
Test2Dict={}
Test2Dict['f1_micro_cv']=d_fin['test_accuracy']
Test2Dict['f1_macro_cv']=macroCV=d_fin['test_f1macro']
Test2Dict['pvalue']=pvalue
Test2Dict['Cnum']=Cnum
Test2Dict['f1micro']=f1micro
Test2Dict['f1macro']=f1macro

Test1_single=pd.DataFrame.from_dict(Test1Dict)
Test2_single=pd.DataFrame.from_dict(Test2Dict)

#save results from crossvalidation 
with open(os.path.join(savepath, 'BenignMalignantMetr.pickle'), 'wb') as handle:
    pickle.dump(Test1_single, handle)

with open(os.path.join(savepath,'BenignLungMetastMetr.pickle'), 'wb') as handle:
    pickle.dump(Test2_single, handle)

# --------------------------------------------------------------------------------------------------------------------------

#get prediction results to construct later confusion matrices with
predictions_test1=get_predictions_cv(features_norm,labels_bm, groups_orig,Test1Dict['Cnum'])
predictions_test2=get_predictions_cv(features_norm,labels_bmetlung, groups_orig,Test2Dict['Cnum'])
        
#save results from predictions
with open(os.path.join(savepath,'BenignMalignantPred.pickle'), 'wb') as handle:
    pickle.dump(predictions_test1, handle)

with open(os.path.join(savepath,'BenignLungMetastPred.pickle'), 'wb') as handle:
    pickle.dump(predictions_test2, handle)


# Now load and apply prefitted SVM models on new data -------------------------------------------------------
    
SVM_bm=pickle.load((open(SVMpath1, 'rb')))
prediction_bm= SVM_bm.predict(features_norm)


SVM_blungmet=pickle.load((open(SVMpath2, 'rb')))
prediction_blungmet=SVM_blungmet.predict(features_norm)

if not os.path.exists(os.path.join(savepath,'prefitted')):
    os.makedirs((os.path.join(savepath,'prefitted')))

with open(os.path.join(savepath,'prefitted','Prediction_BenignMal.pickle'), 'wb') as handle:
    pickle.dump(prediction_bm, handle)

with open(os.path.join(savepath,'prefitted', 'Prediciton_BenignLungMet.pickle'), 'wb') as handle:
    pickle.dump(prediction_blungmet, handle)


with open(os.path.join(savepath,'prefitted','Labels_BenignMal.pickle'), 'wb') as handle:
    pickle.dump(labels_bm, handle)

with open(os.path.join(savepath,'prefitted', 'Labels_BenignLungMet.pickle'), 'wb') as handle:
    pickle.dump(labels_bmetlung, handle)


# ---------------------------
    
