# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:48:28 2018

@author: linde
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:08:42 2018

@author: linde
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:39:51 2018

@author: linde
"""

from keras import backend as K
from sklearn import preprocessing
import pandas as pd
from radio import dataset as ds
from radio.dataset import Pipeline
from CTImagesCustomBatch import CTImagesCustomBatch as CTICB
import keras
import helper_functions as helper
import sklearn.decomposition as decom
import numpy as np
from sklearn.model_selection import GroupKFold  , GroupShuffleSplit  
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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import balanced_accuracy_score, make_scorer, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
import sklearn
from sklearn.model_selection import cross_validate
from scipy.spatial.distance import directed_hausdorff
import scipy
def load_features(index, diagnosis,nodule_info):
    # use this function if pca is already applied before
    features=np.zeros([len(index), 4096])
    labels=[]
    groups=np.zeros([len(index),1])
    indices=np.zeros([len(index),1])
    diameters=np.zeros([len(index),1])
    malprediction=np.zeros([len(index),1])
    for i in range(len(index)):
        path=index.get_fullpath(index.indices[i])
    #transform features and save
        feature=np.load(path+'/features.npy')
        malpred=np.load(path+'/malignancyprediction.npy')
        #add feature to matrix
        features[i,:]=feature
        malprediction[i]=malpred
        
        name=os.path.basename(path)
        index_num=int(name[:6])
        
        nodule_num=int(float(name[15:18]))
        
        patient_sizelist=nodule_info.loc[nodule_info['indices']==index_num, 'diameters']
        string_sizes=patient_sizelist.values[0][1:-1].split(',')
        diameter= float(string_sizes[nodule_num-1])
        diameters[i]=diameter
        
        indices[i]=(index_num)
        labels.append(diagnosis_df.loc[index_num,'label'])
        groups[i]=diagnosis_df.loc[index_num,'patuid']
       
      
        print(i)
    return features, labels  ,groups , indices, malprediction,diameters



def reduce_labels(features_train, labels_train_bin, groups_train,indices_train, keepnum1=0, keepnum2=0,keepnum3=0,keepnum4=0):
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
            
            
  
    return features_scans, labels_scans, group_scans, unique_counts, unique_index


def groups_scans_Haussdorf(features_train, labels_train, groups_train, indices_train, method='max'):
    unique_index=np.unique(indices_train)
    group_scans=np.zeros([len(unique_index)])
    labels_scans=[None] *len(unique_index)

    distance_metric=np.zeros([len(unique_index),len(unique_index)])
 
    features_scans=[ [] for _ in range( len(unique_index))]
    for i in range(len(features_train)):
    #features=features_train[i,:]
        scan_nr=indices_train[i]
        
        num= int(np.where(unique_index == scan_nr)[0])
    
        features_scans[num].append(features_train[i])
    
        labels_scans[num]=labels_train[i]
        group_scans[num]=groups_train[i]
    
    for i in range(len(features_scans)):
        for j in range(len(features_scans)):
            set1=np.array(features_scans[i])
            set2=np.array(features_scans[j])
            
            Y=scipy.spatial.distance.cdist(set1,set2, 'euclidean')
            mindist=np.sort(Y)[:,0]    
                
                
            if method== 'max':   
                distance=np.max(mindist)
            if method== 'min': 
                distance=np.min(mindist)
            if method== 'mean': 
                distance=np.mean(mindist)    
            distance=np.mean(np.array([directed_hausdorff(set1,set2)[0],directed_hausdorff(set2,set1)[0]]))
            distance_metric[i,j]=distance

        
    #i_lower = np.tril_indices(len(unique_index), -1)
    #distance_metric[i_lower] = distance_metric.T[i_lower]
    return distance_metric, labels_scans, group_scans


def optimize_and_cv(features_orig_norm, labels_orig_bin, groups_orig, Clist):
    print('GridSearchCV')
    classifier= svm.LinearSVC( loss='hinge', max_iter=20000, class_weight='balanced')
    gfk=GroupKFold(n_splits=5, method='balance')
    clf=GridSearchCV(classifier, Clist, cv=gfk, scoring=['f1_macro', 'f1_micro'],refit=False, return_train_score=False)
    clf.fit(features_orig_norm, np.ravel(labels_orig_bin),np.ravel(groups_orig))
    
    GridResults=pd.DataFrame(clf.cv_results_)
    Cdict=GridResults.loc[GridResults['rank_test_f1_macro']== 1]['params']
    Cnum=Cdict.iloc[0].get('C')
    
    
    clf = make_pipeline(svm.LinearSVC(C=Cnum , max_iter=50000,loss='hinge', class_weight='balanced'))
    gfk=GroupKFold(method='balance',n_splits=5)
    scoring = {'f1macro': 'f1_macro',
              
                'accuracy': 'accuracy'}
    print('Crossvalidating starts')
    scores=cross_validate(clf, features_orig_norm,np.ravel(labels_orig_bin),np.ravel(groups_orig) ,cv=gfk, scoring=scoring, return_train_score=True)
    
    d_fin=pd.DataFrame(scores)
    
    final_results=np.mean(d_fin)
    
    print(final_results)
    test_f1micro=final_results.loc['test_accuracy']
    test_f1macro=final_results.loc['test_f1macro']
    return test_f1micro, test_f1macro, Cnum


#define paths
cancer_folder='C:/Users/linde/Documents/Features_Classification/*/*conv'

diagnosis_df = pd.read_csv('C:/Users/linde/OneDrive - TU Eindhoven/TUE/Afstuderen/CSVFILES/batchLablesLinde2.csv')
diagnosis_df = pd.read_excel('C:/Users/linde/OneDrive - TU Eindhoven/TUE/Afstuderen/CSVFILES/AnnotatiesPim/diagnosis_patient_nodulesisze_contrastl.csv')

#load everything
index_train=ds.FilesIndex(path=[cancer_folder],dirs=True,sort=True)
features_orig, labels_orig ,groups_orig, indices_orig,_,diameters =load_features(index_train,diagnosis_df,nodule_info) 

#reduce diameters
features_orig, labels_orig, groups_orig, indices_orig,diameters= reduce_diameters(features_orig, labels_orig, groups_orig, indices_orig, diameters)




#normalize features
features_orig_norm = preprocessing.normalize(features_orig)  

#transform features to number labels
labels_orig_bin=np.zeros(len(labels_orig))
for i in range(len(labels_orig)):
    if labels_orig[i] == 'benigne multinodulair':
        labels_orig_bin[i]=0
    if labels_orig[i] == 'benige'   :
        labels_orig_bin[i]=1
    if labels_orig[i]== 'crc':
        labels_orig_bin[i]=2 
    if labels_orig[i] == 'long'   :
        labels_orig_bin[i]=3 
    if labels_orig[i] == 'melanoom'   :
        labels_orig_bin[i]=4      
        
        

#determine C
Clist={'C':[0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000]}


#make different datasets / labels

#dataset 1: Benign / Malignant, All data
labels_bm_all=np.where(labels_orig_bin>1, 1,0)

#dataset2: Benign / Malignant, reduced dataset
features_less, labels_less, groups_less, indices_less= reduce_labels(features_orig_norm, labels_orig_bin, groups_orig, indices_orig, keepnum1=1, keepnum2=2,keepnum3=3,keepnum4=4)
labels_bm_less=np.where(labels_less>1, 1,0)


#dataset3: all data, benign / metastasen / lung
labels_bmetlung_all=np.where(labels_orig_bin>1, labels_orig_bin,1  )
labels_bmetlung_all = np.where(labels_bmetlung_all >3, 3,labels_bmetlung_all)

#dataset4: reduced data, benign / metastasen/ lung
labels_bmetlung_less=np.where(labels_less>1, labels_less,1  )
labels_bmetlung_less = np.where(labels_bmetlung_less >3, 3,labels_bmetlung_less)

#dataset 5: reduced data, benign / lung / crc/ melanoma
labels_blcm_less=np.where(labels_less>1, labels_less,1  )



def group_and_analyze(features, labels, groups,indices, Clist, function='max'):
    features_scans, labels_scans, group_scan=groups_scans(features, labels, groups, indices,function=function)
    f1micro, f1macro, Cnum= optimize_and_cv(features_scans, labels_scans, group_scan, Clist)
    return f1micro, f1macro, Cnum
    

    
    
#do experiements with dataset / labels
final_results=np.zeros([3,5,2])
CnumList_fin=np.zeros([3,5,1])
i=0
#Dataset1
for methods in ['max' ]:
    
    results_ar=np.zeros([5,2])
    CnumList=np.zeros([5,1])

    print('Start Dataset1')

    f1micro, f1macro,Cnum=group_and_analyze(features_orig_norm, labels_bm_all, groups_orig, indices_orig, Clist, function=methods)
    results_ar[0,:]=[f1micro, f1macro]
    CnumList[0]=Cnum
    #Dataset2
    print('Start Dataset2')
    f1micro, f1macro, Cnum = group_and_analyze(features_less, labels_bm_less, groups_less,indices_less, Clist,function=methods)
    results_ar[1,:]=[f1micro, f1macro]
    CnumList[1]=Cnum
    #dataset3
    print('Start Dataset3')
    f1micro, f1macro , Cnum= group_and_analyze(features_orig_norm, labels_bmetlung_all, groups_orig, indices_orig,Clist,function=methods)
    results_ar[2,:]=[f1micro, f1macro]
    CnumList[2]=Cnum
    #dataset4
    print('Start Dataset4')
    f1micro, f1macro , Cnum= group_and_analyze(features_less, labels_bmetlung_less, groups_less, indices_less, Clist,function=methods)
    results_ar[3,:]=[f1micro, f1macro]
    CnumList[3]=Cnum
    #dataset5
    print('Start Dataset5')
    f1micro, f1macro, Cnum = group_and_analyze(features_less, labels_blcm_less, groups_less,indices_less, Clist,function=methods)
    results_ar[4,:]=[f1micro, f1macro]
    CnumList[4]=Cnum

    final_results[i,:,:]=results_ar
    CnumList_fin[i,:,:]=CnumList
    i=i+1
    
    
np.save('Classification_results_groups_Hauss', final_results)
np.save('C_results_groups_Hauss', CnumList_fin)

#
##small test with predictions of max operators
##f1micro, f1macro, Cnum = group_and_analyze(features_less, labels_blcm_less, groups_less,indices_less, Clist,function=methods)
features_scans, labels_scans, group_scan, counts, index=groups_scans(features_orig_norm, labels_orig, groups_orig, indices_orig, function='max')
df= {'indices': index, 'counts' : counts, 'labels': labels_scans}
dataframe=pd.DataFrame(data=df)
dataframe.groupby('labels')['counts'].hist(bins=15)
dataframe.groupby('labels')['counts'].std()
plt.legend(['benign','benigne multi', 'crc','lung', 'melanoma'])


nodules_df=pd.read_csv('nodule_data.csv')
nodules_df['diagnosis']=0 #add column for diagnosis

nodules_df.dropna()



    
unique_index=np.unique(indices_orig)
selected_df=diagnosis_df.loc[unique_index]

nodules_df['scannum']=int([e[6:] for e in nodules_df['StudyInstanceUID']])-1

dataframe['diameters']=size_list_total
size_list_total=[]
for index, row in dataframe.iterrows():
    num=row['indices']
    numstring='1.2.3.'+ str(int(num)+1)
    
    nodules_rows=nodules_df[nodules_df['StudyInstanceUID'] == numstring]
    size_list=nodules_rows['Diameter [mm]'].tolist()
    print(size_list)
    size_list_total.append(size_list)

    print(numstring)
    
for name, group in dataframe.groupby('labels'):
    sizes=(group['diameters'].tolist())
    sizes_flat=[item for sublist in sizes for item in sublist]
    
    print(name)
    print(np.mean(np.array(sizes_flat)))
    print(np.std(np.array(sizes_flat)))
    
    
sizes_flat=[item for sublist in size_list_total for item in sublist]    
    
dataframe['diameters']=size_list_total    
    
#classifier= svm.LinearSVC( loss='hinge', max_iter=20000, class_weight='balanced', C=100)
#gfk=GroupKFold(n_splits=5, method='balance')
#scores=cross_val_predict(classifier, features_scans,np.ravel(labels_scans),np.ravel(group_scan) ,cv=gfk)
#mat=sklearn.metrics.confusion_matrix(labels_scans,scores)    
