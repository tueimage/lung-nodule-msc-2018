# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:56:24 2018

@author: s120116
"""

from sklearn import preprocessing
import pandas as pd
from radio import dataset as ds
from radio.dataset import Pipeline
from CTImagesCustomBatch import CTImagesCustomBatch as CTICB
import keras
import helper_functions as helper
import sklearn.decomposition as decom
import numpy as np
from sklearn.model_selection import GroupKFold  , GroupShuffleSplit  , KFold
import matplotlib.pyplot as plt
import pickle
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import os
from keras.layers import MaxPooling3D
from sklearn.metrics.pairwise import euclidean_distances
import skimage.measure as measure
from scipy.ndimage.filters import gaussian_filter
#cnn = keras.models.load_model('C:/Users/s120116/Documents/FinalModels/neuralnet_final(32x32x64)')
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import  make_scorer, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
import sklearn
from sklearn.model_selection import cross_validate
from scipy.spatial.distance import directed_hausdorff
from sklearn.model_selection import permutation_test_score

def load_features(index, diagnosis,nodule_info):
    # use this function if pca is already applied before
    features=np.zeros([len(index), 4096])
    labels=[]
    groups=np.zeros([len(index),1])
    indices=np.zeros([len(index),1])
    diameters=np.zeros([len(index),1])
   # malprediction=np.zeros([len(index),1])
    for i in range(len(index)):
        path=index.get_fullpath(index.indices[i])
    #transform features and save
        feature=np.load(path+'/features.npy')
        #malpred=np.load(path+'/malignancyprediction.npy')
        #add feature to matrix
        features[i,:]=feature
        #malprediction[i]=malpred
        
        name=os.path.basename(path)
        index_num=int(name[:6])
        
        nodule_num=int(float(name[15:18]))
        
        patient_sizelist=nodule_info.loc[nodule_info['indices']==index_num, 'diameters']
        string_sizes=patient_sizelist.values[0][1:-1].split(',')
        diameter= float(string_sizes[nodule_num-1])
        diameters[i]=diameter
        
        indices[i]=(index_num)
        labels.append(diagnosis.loc[index_num,'label'])
        groups[i]=diagnosis.loc[index_num,'patuid']
      
    return features, labels  ,groups , indices, diameters

def reduce_labels(features_train, labels_train_bin, groups_train,indices_train, keepnum1=1, keepnum2=0,keepnum3=0,keepnum4=0):
    reduced_labels=[]
    reduced_features=[]
    reduced_groups=[]
    reduced_indices=[]
    for i in range(len(features_train)):
        if labels_train_bin[i]==keepnum1 or labels_train_bin[i]==keepnum2 or labels_train_bin[i]==keepnum3 or labels_train_bin[i]==keepnum4:
            reduced_labels.append(labels_train_bin[i])
            reduced_features.append(features_train[i,:])
            reduced_groups.append(groups_train[i])
            reduced_indices.append(indices_train[i])
    return np.array(reduced_features), np.array(reduced_labels), np.array(reduced_groups), np.array(reduced_indices)

def groups_scans(features_train, labels_train, groups_train, indices_train,function='max'):
    unique_index, unique_counts=np.unique(indices_train, return_counts=True)
    group_scans=np.zeros([len(unique_index)])
    labels_scans=[None] *len(unique_index)


    features_scans=np.zeros([len(unique_index),4096])

    for i in range(len(features_train)):
    #features=features_train[i,:]
        scan_nr=indices_train[i]
        num= int(np.where(unique_index == scan_nr)[0])
    
        if function == 'max':
            features_scans[num]=np.maximum(features_scans[num], features_train[i])
        
        if function == 'min':
            features_scans[num]=np.minimum(features_scans[num], features_train[i]) 
            
        if function == 'mean': 
            features_scans[num]=features_scans[num] + features_train[i]
            
    
        labels_scans[num]=labels_train[i]
        group_scans[num]=groups_train[i]
     
    if function == 'mean': 
        for i in range(len(features_scans)):
            features_scans[i,:]=features_scans[i,:]/unique_counts[i]   
    return features_scans, labels_scans, group_scans


def groups_scans_spect(features_train, labels_train, groups_train, indices_train,function='max'):
    unique_index, unique_counts=np.unique(indices_train, return_counts=True)
    group_scans=np.zeros([len(unique_index)])
    labels_scans=[None] *len(unique_index)


    features_scans=np.zeros([len(unique_index),3*4096])

    for i in range(len(features_train)):
    #features=features_train[i,:]
        scan_nr=indices_train[i]
        num= int(np.where(unique_index == scan_nr)[0])
    
        if function == 'max':
            features_scans[num]=np.maximum(features_scans[num], features_train[i])
        
        if function == 'min':
            features_scans[num]=np.minimum(features_scans[num], features_train[i]) 
            
        if function == 'mean': 
            features_scans[num]=features_scans[num] + features_train[i]
            
    
        labels_scans[num]=labels_train[i]
        group_scans[num]=groups_train[i]
     
    if function == 'mean': 
        for i in range(len(features_scans)):
            features_scans[i,:]=features_scans[i,:]/unique_counts[i]   
    return features_scans, labels_scans, group_scans



def reduce_diameters(features_train, labels_train_bin, groups_train,indices_train,diameters):
    reduced_labels=[]
    reduced_features=[]
    reduced_groups=[]
    reduced_indices=[]
    reduced_diameters=[]
    for i in range(len(features_train)):
        if 3 <=  diameters[i]:
            reduced_labels.append(labels_train_bin[i])
            reduced_features.append(features_train[i,:])
            reduced_groups.append(groups_train[i])
            reduced_indices.append(indices_train[i])
            reduced_diameters.append(diameters[i])
    return np.array(reduced_features), np.array(reduced_labels), np.array(reduced_groups), np.array(reduced_indices), np.array(reduced_diameters)


def optimize_and_cv(features_orig_norm, labels_orig_bin, groups_orig, Clist,permut=True):
    print('GridSearchCV')
    classifier= svm.LinearSVC( loss='hinge', max_iter=20000, class_weight='balanced')
    gfk=GroupKFold(n_splits=10)
    clf=GridSearchCV(classifier, Clist, cv=gfk, scoring=['f1_macro', 'f1_micro'],refit=False, return_train_score=False)
    clf.fit(features_orig_norm, np.ravel(labels_orig_bin),np.ravel(groups_orig))
    
    GridResults=pd.DataFrame(clf.cv_results_)
    Cdict=GridResults.loc[GridResults['rank_test_f1_macro']== 1]['params']
    Cnum=Cdict.iloc[0].get('C')
    
    
    clf = make_pipeline(svm.LinearSVC(C=Cnum , max_iter=50000,loss='hinge', class_weight='balanced'))
    gfk=GroupKFold(n_splits=10)
    
    scoring = {'f1macro': 'f1_macro',
              
                'accuracy': 'accuracy'}
    print('Crossvalidating starts')
    
    scores=cross_validate(clf, features_orig_norm,np.ravel(labels_orig_bin),np.ravel(groups_orig) ,cv=gfk, scoring=scoring, return_train_score=True)
    if permut==True:
        print('Permutaiton starts')
        score,permuation_scores,pvalue =permutation_test_score(classifier, features_orig_norm, labels_orig_bin,scoring='f1_macro', cv=10, n_permutations=100)
    else:
        pvalue=0
    
    d_fin=pd.DataFrame(scores)
    
    
        
    final_results=np.mean(d_fin)
    
    print(final_results)
    test_f1micro=final_results.loc['test_accuracy']
    test_f1macro=final_results.loc['test_f1macro']
    return test_f1micro, test_f1macro, Cnum ,d_fin, pvalue

def load_features_spect(index1, index2, index3, diagnosis,nodule_info):
    # use this function if pca is already applied before
    features=np.zeros([len(index1), 4096 *3 ])
    labels=[]
    groups=np.zeros([len(index1),1])
    indices=np.zeros([len(index1),1])
    diameters=np.zeros([len(index1),1])
    for i in range(len(index1)):
        path1=index1.get_fullpath(index1.indices[i])
        path2=index2.get_fullpath(index2.indices[i])
        path3=index3.get_fullpath(index3.indices[i])
    #transform features and save
        feature1=np.load(path1+'/features.npy')
        feature2=np.load(path2+'/features.npy')
        feature3=np.load(path3+'/features.npy')



        #add feature to matrix
        combined_features=np.concatenate((feature1, feature2, feature3), axis=None)
        features[i,:]=combined_features

        
        name=os.path.basename(path1)
        index_num=int(name[:6])
        
        nodule_num=int(float(name[15:18]))
        patient_sizelist=nodule_info.loc[nodule_info['indices']==index_num, 'diameters']
        string_sizes=patient_sizelist.values[0][1:-1].split(',')
        diameter= float(string_sizes[nodule_num-1])
        diameters[i]=diameter
        
        indices[i]=(index_num)
        labels.append(diagnosis.loc[index_num,'label'])
        groups[i]=diagnosis.loc[index_num,'patuid']
          
    return features, labels  ,groups , indices, diameters