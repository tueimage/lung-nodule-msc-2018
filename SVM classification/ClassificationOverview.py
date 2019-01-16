# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:56:17 2018

@author: s120116
In this file the features and labels of the nodule crops are loaded and SVM classification is performed.
Classificaiton is performed both with and without the spectral data, and with two types of labels: 2 and 3 class
"""


from sklearn import preprocessing
import pandas as pd
from radio import dataset as ds
import numpy as np
from sklearn.model_selection import GroupKFold 
from sklearn import svm
from sklearn.manifold import TSNE
import sklearn



from HelperFileClassification import load_features, groups_scans, reduce_diameters,optimize_and_cv, groups_scans_spect, load_features_spect

def bin_labels(labels):
    labels_bin=np.zeros(len(labels))
    for i in range(len(labels)):
        if labels[i] == 'benigne multinodulair':
            labels_bin[i]=0
        if labels[i] == 'benige'   :
            labels_bin[i]=1
        if labels[i]== 'crc':
            labels_bin[i]=3 
        if labels[i] == 'long'   :
            labels_bin[i]=2 
        if labels[i] == 'melanoom'   :
            labels_bin[i]=4       
    return labels_bin


#load all data
cancer_folder='C:/Users/linde/Documents/Features_Classification/*/*conv'
cancer_folder_060='C:/Users/linde/Documents/Features_Classification/*/*060'
cancer_folder_190='C:/Users/linde/Documents/Features_Classification/*/*190'


diagnosis_df = pd.read_csv('C:/Users/linde/OneDrive - TU Eindhoven/TUE/Afstuderen/CSVFILES/batchLablesLinde2.csv')
nodule_info = pd.read_excel('C:/Users/linde/OneDrive - TU Eindhoven/TUE/Afstuderen/CSVFILES/AnnotatiesPim/diagnosis_patient_nodulesize_contrastl.xlsx')

#load everything
index_train=ds.FilesIndex(path=[cancer_folder],dirs=True,sort=True)
index060=ds.FilesIndex(path=[cancer_folder_060],dirs=True,sort=True)
index190=ds.FilesIndex(path=[cancer_folder_190],dirs=True,sort=True)

features_orig, labels_orig ,groups_orig, indices_orig,diameters =load_features(index_train,diagnosis_df,nodule_info)
features_orig_spect, labels_orig_spect ,groups_orig_spect, indices_orig_spect,diameters_spect =load_features_spect(index_train,index060, index190, diagnosis_df,nodule_info)



#do preprocessing of feature vectors
features_norm_spect=preprocessing.normalize(features_orig_spect)  
features_norm=preprocessing.normalize(features_orig) 

#exclude nodules with diameter <3mm
reduced_features, reduced_labels, reduced_groups, reduced_indices, _ = reduce_diameters(features_norm, labels_orig, groups_orig,indices_orig,diameters)
reduced_features_spect, reduced_labels_spect, reduced_groups_spect, reduced_indices_spect, _ = reduce_diameters(features_norm_spect, labels_orig_spect, groups_orig_spect,indices_orig_spect,diameters_spect)




#binarize labels
reduced_labels_bin=bin_labels(reduced_labels)
reduced_labels_bin_spect=bin_labels(reduced_labels_spect)



#determine C
Clist={'C':[0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000]}


#dataset 1: Benign / Malignant, All data
labels_bm_all=np.where(reduced_labels_bin>1, 1,0)
labels_bm_spect=np.where(reduced_labels_bin_spect>1, 1,0)

#dataset3: all data, benign / metastasen / lung
labels_bmetlung_all=np.where(reduced_labels_bin>1, reduced_labels_bin,1  )
labels_bmetlung_all = np.where(labels_bmetlung_all >3, 3,labels_bmetlung_all)

labels_bmetlung_spect=np.where(reduced_labels_bin_spect>1, reduced_labels_bin_spect,1  )
labels_bmetlung_spect = np.where(labels_bmetlung_spect>3, 3,labels_bmetlung_spect)


def group_and_analyze(features, labels, groups,indices, Clist, function='max',permut=True):
    features_scans, labels_scans, group_scan=groups_scans(features, labels, groups, indices,function)
    f1micro, f1macro, Cnum, d_fin, pvalue= optimize_and_cv(features_scans, labels_scans, group_scan, Clist,permut=permut)
    return f1micro, f1macro, Cnum, d_fin, pvalue


def group_and_analyze_spect(features, labels, groups,indices, Clist, function='max',permut=True):
    features_scans, labels_scans, group_scan=groups_scans(features, labels, groups, indices,function)
    f1micro, f1macro, Cnum, d_fin, pvalue= optimize_and_cv(features_scans, labels_scans, group_scan, Clist,permut=permut)
    return f1micro, f1macro, Cnum, d_fin, pvalue


##run both test for normal data
print('Start Dataset1')
Clist300={'C':[300]}
f1micro, f1macro,Cnum,d_fin,pvalue=group_and_analyze(reduced_features, reduced_labels_bin, reduced_groups, reduced_indices, Clist, function='max',permut=False)
Test1Dict={}
Test1Dict['f1_micro_cv']=d_fin['test_accuracy']
Test1Dict['f1_macro_cv']=macroCV=d_fin['test_f1macro']
Test1Dict['pvalue']=pvalue
Test1Dict['Cnum']=Cnum
Test1Dict['f1micro']=f1micro
Test1Dict['f1macro']=f1macro


dataset2
print('Start Dataset2')
Clist30={'C':[30]}
f1micro, f1macro , Cnum,d_fin, pvalue= group_and_analyze(reduced_features, reduced_labels_bin, reduced_groups,reduced_indices,Clist,function='max',permut=True)
#fill dict with all information
Test2Dict={}
Test2Dict['f1_micro_cv']=d_fin['test_accuracy']
Test2Dict['f1_macro_cv']=d_fin['test_f1macro']
Test2Dict['pvalue']=pvalue
Test2Dict['Cnum']=Cnum
Test2Dict['f1micro']=f1micro
Test2Dict['f1macro']=f1macro



#make dataframes from result
Test1=pd.DataFrame.from_dict(Test1Dict)
Test2=pd.DataFrame.from_dict(Test2Dict)

df=pd.DataFrame({'Benign/Malignant': Test1.f1_micro_cv, 'Benign/Lung/Metastases':Test2.f1_micro_cv,'Mode': 'Conventional','group':'Per Scan'})
#
##plot results from crossvalidation in boxplot
#
##run both tests for contatenated spectral data ---------------------------------------------------------------------------------------------
print('Start Spectral Data')
print('Start Dataset1')
Clist100={'C':[100]}
f1micro, f1macro,Cnum,d_fin,pvalue=group_and_analyze_spect(reduced_features_spect, labels_bm_spect, reduced_groups_spect, reduced_indices_spect, Clist, function='max',permut=False)
Test1Dict={}
Test1Dict['f1_micro_cv']=d_fin['test_accuracy']
Test1Dict['f1_macro_cv']=macroCV=d_fin['test_f1macro']
Test1Dict['pvalue']=pvalue
Test1Dict['Cnum']=Cnum
Test1Dict['f1micro']=f1micro
Test1Dict['f1macro']=f1macro


#dataset2
print('Start Dataset2')
f1micro, f1macro , Cnum,d_fin, pvalue= group_and_analyze_spect(reduced_features_spect, labels_bmetlung_spect, reduced_groups_spect,reduced_indices_spect,Clist,function='max',permut=False)
#fill dict with all information
Test2Dict={}
Test2Dict['f1_micro_cv']=d_fin['test_accuracy']
Test2Dict['f1_macro_cv']=d_fin['test_f1macro']
Test2Dict['pvalue']=pvalue
Test2Dict['Cnum']=Cnum
Test2Dict['f1micro']=f1micro
Test2Dict['f1macro']=f1macro

#
#make dataframes from result
Test1_spect=pd.DataFrame.from_dict(Test1Dict)
Test2_spect=pd.DataFrame.from_dict(Test2Dict)

df_spect=pd.DataFrame({'Benign/Malignant': Test1_spect.f1_micro_cv, 'Benign/Lung/Metastases':Test2_spect.f1_micro_cv,'Mode':'Spectral' ,'group':'Per Scan'})

#




#do both tests for single nodules =-----------------------------------------------------------------------------
print('Start Single Nodule Scoring')
print('Start test1')
f1micro, f1macro, Cnum, d_fin, pvalue= optimize_and_cv(reduced_features,reduced_labels_bin, reduced_groups, Clist,permut=False)
Test1Dict={}
Test1Dict['f1_micro_cv']=d_fin['test_accuracy']
Test1Dict['f1_macro_cv']=macroCV=d_fin['test_f1macro']
Test1Dict['pvalue']=pvalue
Test1Dict['Cnum']=Cnum
Test1Dict['f1micro']=f1micro
Test1Dict['f1macro']=f1macro

Clist30={'C':[30]}
print('Start test2')
f1micro, f1macro, Cnum, d_fin, pvalue= optimize_and_cv(reduced_features, reduced_labels_bin, reduced_groups, Clist,permut=True)
Test2Dict={}
Test2Dict['f1_micro_cv']=d_fin['test_accuracy']
Test2Dict['f1_macro_cv']=macroCV=d_fin['test_f1macro']
Test2Dict['pvalue']=pvalue
Test2Dict['Cnum']=Cnum
Test2Dict['f1micro']=f1micro
Test2Dict['f1macro']=f1macro

Test1_single=pd.DataFrame.from_dict(Test1Dict)
Test2_single=pd.DataFrame.from_dict(Test2Dict)

df_single=pd.DataFrame({'Benign/Malignant': Test1_single.f1_micro_cv, 'Benign/Lung/Metastases':Test2_single.f1_micro_cv,'Mode':'Conventional' , 'group':'Per Nodule'})



#do both tests for single nodule in spect =-----------------------------------------------------------------------------
print('Start Single Nodule Scoring')
print('Start test1')
f1micro, f1macro, Cnum, d_fin, pvalue= optimize_and_cv(reduced_features_spect, labels_bm_spect, reduced_groups_spect, Clist,permut=False)
Test1Dict={}
Test1Dict['f1_micro_cv']=d_fin['test_accuracy']
Test1Dict['f1_macro_cv']=macroCV=d_fin['test_f1macro']
Test1Dict['pvalue']=pvalue
Test1Dict['Cnum']=Cnum
Test1Dict['f1micro']=f1micro
Test1Dict['f1macro']=f1macro


print('Start test2')
f1micro, f1macro, Cnum, d_fin, pvalue= optimize_and_cv(reduced_features_spect, labels_bmetlung_spect, reduced_groups_spect, Clist, permut=True)
Test2Dict={}
Test2Dict['f1_micro_cv']=d_fin['test_accuracy']
Test2Dict['f1_macro_cv']=macroCV=d_fin['test_f1macro']
Test2Dict['pvalue']=pvalue
Test2Dict['Cnum']=Cnum
Test2Dict['f1micro']=f1micro
Test2Dict['f1macro']=f1macro

Test1_single_spect=pd.DataFrame.from_dict(Test1Dict)
Test2_single_spect=pd.DataFrame.from_dict(Test2Dict)

df_single_spect=pd.DataFrame({'Benign/Malignant': Test1_single_spect.f1_micro_cv, 'Benign/Lung/Metastases':Test2_single_spect.f1_micro_cv,'Mode':'Spectral' , 'group':'Per Nodule'})


















##combine datafrmaes in right way for plotting
df_all=pd.concat([df, df_spect,df_single, df_single_spect])
plot_pd=pd.melt(df_all, id_vars=['Mode', 'group'], var_name=['Classification groups'],  value_name='F1_micro')
#
##plot results from crossvalidation in boxplot
#import seaborn as sns
sns.color_palette("pastel")
plt.figure()

g=sns.catplot(x='group', y='F1_micro', hue='Mode', col='Classification groups',data=plot_pd, kind='box',palette='pastel',legend=False )
#(g.set_axis_labels("", "Survival Rate")
#g.set_xticklabels(["Men", "Women", "Children"])
g.set_titles("{col_name}")
g.set_axis_labels("", 'Accuracy')
g.add_legend( title= None )


plt.plot(1, 2, c='black')





#from matplotlib2tikz import save as tikz_save
tikz_save("AccuracyBoxplots.tikz", figureheight='\\figureheight', figurewidth='\\figurewidth',strict=False)
ax=sns.swarmplot(x='variable', y='value', hue='Mode' ,data=plot_pd,color=".25")
#run both tests for late fusion spectral data    

-------------------------------
import scipy
tstats1, pvalue1=scipy.stats.ttest_rel(Test1_spect.f1_macro_cv, Test1.f1_macro_cv)
tstats2, pvalue2=scipy.stats.ttest_rel(Test2_spect.f1_macro_cv, Test2.f1_macro_cv)

tstats1, pvalue1_mic=scipy.stats.ttest_rel(Test1_spect.f1_micro_cv, Test1.f1_micro_cv) #only this one is statisticl significant
tstats2, pvalue2_mic=scipy.stats.ttest_rel(Test2_spect.f1_micro_cv, Test2.f1_micro_cv)


tstats1, pvalue1_nod=scipy.stats.ttest_rel(Test1_single_spect.f1_macro_cv, Test1_single.f1_macro_cv)
tstats2, pvalue2_nod=scipy.stats.ttest_rel(Test2_single_spect.f1_macro_cv, Test2_single.f1_macro_cv)

tstats1, pvalue1_mic_nod=scipy.stats.ttest_rel(Test1_single_spect.f1_micro_cv, Test1_single.f1_micro_cv) #only this one is statisticl significant
tstats2, pvalue2_mic_nod=scipy.stats.ttest_rel(Test2_single_spect.f1_micro_cv, Test2_single.f1_micro_cv)














# ----------------------------------------------------  Get all fonsusion tables
def get_predictions_cv(features,labels, groups, Cnum):
    classifier= svm.LinearSVC( C=Cnum, loss='hinge', max_iter=20000, class_weight='balanced')
    folds=GroupKFold(n_splits=10)
    total_perf=[]
    for train_index, test_index in folds.split(features,labels,groups): #gaat er twee keer doorheen , waarby train/test flippe
        X_train, X_test = features[train_index], features[test_index]
        Y_train, Y_test = labels[train_index], labels[test_index]
        
       
        svm_fit=classifier.fit(X_train,Y_train)
        labels_pred=svm_fit.predict(X_test)
        perform_ar=np.vstack((Y_test,labels_pred))
        
        total_perf.append(perform_ar)
            
    return total_perf
        
#scan-level 
#predictions_test1=get_predictions_cv(reduced_features,labels_bm_all, reduced_groups,Test1['Cnum'].tolist()[0])


features_scans, labels_scans, group_scan=groups_scans(reduced_features, reduced_labels_bin, reduced_groups,reduced_indices,'max')
#features_scans_spect, labels_scans, group_scan=groups_scans_spect(reduced_features_spect, labels_bmetlung_all, reduced_groups,reduced_indices,'max')

#nodule-level
predictions_conv=get_predictions_cv(reduced_features,reduced_labels_bin, reduced_groups,Test1_single.Cnum[0])
#predictions_spect=get_predictions_cv(reduced_features_spect,labels_bmetlung_all, reduced_groups,30)

a=np.hstack(predictions_conv)
matrix_nodule=sklearn.metrics.confusion_matrix(a[0,:], a[1,:], labels=None, sample_weight=None)

#a_spect=np.hstack(predictions_spect)
#matrix_spect=sklearn.metrics.confusion_matrix(a_spect[0,:], a_spect[1,:], labels=None, sample_weight=None)





#scan-level
predictions_conv_scans=get_predictions_cv(features_scans,np.array(labels_scans), np.array(group_scan),Test1.Cnum[0])
#predictions_spect_scans=get_predictions_cv(features_scans_spect,np.array(labels_scans), np.array(group_scan),100)




a=np.hstack(predictions_conv_scans)
matrix_scan=sklearn.metrics.confusion_matrix(a[0,:], a[1,:], labels=None, sample_weight=None)

#a_spect=np.hstack(predictions_spect_scans)
#matrix_spect=sklearn.metrics.confusion_matrix(a_spect[0,:], a_spect[1,:], labels=None, sample_weight=None)
##
#
#plt.figure()
#plot_confusion_matrix(matrix, classes=['Benign', 'Lung', 'Metastases'],
#                      title='')
##save workspace -----------------------------------------------------------------
#import shelve
#
#
#filename='../ClassificationOverview.out'
#my_shelf = shelve.open(filename,'n') # 'n' for new
#
#for key in dir():
#    try:
#        my_shelf[key] = globals()[key]
#    except TypeError:
#        #
#        # __builtins__, my_shelf, and imported modules can not be shelved.
#        #
#        print('ERROR shelving: {0}'.format(key))
#my_shelf.close()
#
#
##for opening worksape
#my_shelf = shelve.open(filename)
#for key in my_shelf:
#    globals()[key]=my_shelf[key]
#my_shelf.close()
#
