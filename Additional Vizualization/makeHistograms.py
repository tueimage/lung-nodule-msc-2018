# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:52:27 2018

@author: s120116
In this script the intensities of CT scans are compared with histograms
The histograms are plotted devided in different groups
"""

import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/s120116/OneDrive - TU Eindhoven/TUE/Afstuderen/lung-nodule-msc-2018')

from datetime import datetime
startTime = datetime.now()
import numpy as np
import pandas as pd

#load all necesary files
hist_count=np.load('intensitie_counts.npy')
hist_count_utrecht=np.load('intensitie_counts_utrecht.npy')
xls = pd.read_excel('LIDC-IDRI_MetaData.xls')
df = xls.parse(xls.sheet_names[0])


def get_bins(hist_count):

    """
    This function determines the bin location to get equal bins inside the lung
    """
    
    #flatten out lists and determine bin size in interesting region 
    value_lists= [valuelist for _ ,valuelist , _  in hist_count]
    flat_value_list = [item for sublist in value_lists for item in sublist]
    
    existing_values=np.unique(np.array(flat_value_list)) #flat list with a possible pixel values
    total_counts=np.zeros(len(existing_values))
    
    for i in range(len(hist_count)):
        pixel_sum=sum(hist_count[i][2])
        for j in range(len(hist_count[i][1])):
            value=hist_count[i][1][j]
            count_rel=hist_count[i][2][j] / pixel_sum
            index=np.where(existing_values==value)
            total_counts[index]=total_counts[index]+count_rel
            
    #determine interesting region
    indices=np.where(np.logical_and(existing_values>=-950,existing_values<=-600))
    indexmin950=min(indices[0])
    indexmin600=max(indices[0])
    
    #clip value lists to only interesting part of lung
    values_lung=existing_values[int(indexmin950):int(indexmin600)+1]
    counts_lung=total_counts[int(indexmin950):int(indexmin600)+1]
    
    #get relative count
    total_counts_lung=np.sum(counts_lung) 
    counts_lung_rel=counts_lung/ total_counts_lung  
    
    #determine bin locations
    bin_list=[]
    bin_list.append(min(values_lung))
    summed=0
    k=1
    for i in range(len(counts_lung_rel)):
        summed=summed+counts_lung_rel[i] #add all values along relative counts
        if (summed >= k*0.05): #if next bin has been reached (0.1), add value
            k=k+1 
            bin_list.append(values_lung[i+1]) #because of boundstake i+1
    
    bin_list.append(max(values_lung))        
    
    
    a=plt.hist(values_lung,weights=counts_lung_rel, bins=bin_list)  
    
    #determine centers of bin for plot
    centers=np.zeros(len(a[0]))
    for i in range(1,len(a[1])):
        centers[i-1]=(a[1][i-1]+a[1][i])/2
        

    return centers, bin_list

centers,bin_list=get_bins(hist_count)

plt.axis([-950, -600, 0, 1])


#plot all scans seperately (big mess)
for i in range(700):
    values=hist_count[i][1]
    counts=hist_count[i][2]
    indices=np.where(np.logical_and(values>=-950,values<=-600))
    indexmin950=min(indices[0])
    indexmin600=max(indices[0])
 
    values_lung=values[int(indexmin950):int(indexmin600)+1]
    counts_lung=counts[int(indexmin950):int(indexmin600)+1]
    
    total_counts_lung=np.sum(counts_lung) 
    counts_lung_rel=counts_lung/ total_counts_lung  
    
    a=np.histogram(values_lung,weights=counts_lung_rel, bins=bin_list)  
    
    plt.plot(centers,a[0])



#--------------------------------
    #make lists of manufacturers


def sort_categories(hist_count, categorie='Manufacturer'):
    
    categories=[]
    categories_numbers=[]
    for i in range(len(hist_count)):
    
    
        series_uid=hist_count[i][0]
        categorie_name=(df[(df['Series UID']==''.join(series_uid ))])[categorie]
        categorie_string=categorie_name.to_string(index=False)
        if not categorie_string in categories:
            categories.append(categorie_string)
            categories_numbers.append([]) #create list in list
        
        num_cat=categories.index(categorie_string)
        categories_numbers[num_cat].append(i)   

    return categories, categories_numbers
    
    



def returnhist(hist_count,index_numbers=np.zeros(len(hist_count)),bins=bin_list):
    """" 
    Function returns the bin height for a given histogram count matrix, index numbers and the list of 
    bin locations
    """
    hist_count_cat=np.take(hist_count, index_numbers,0)

    #get list of all existing values and make a list of equal length with zeros
    value_lists= [valuelist for _ ,valuelist , _  in hist_count_cat]                    
    flat_list = [item for sublist in value_lists for item in sublist]
    existing_values=np.unique(np.array(flat_list)) #unique list of values in GEscans
    
    
    #determine interesting values               
    indices=np.where(np.logical_and(existing_values>=-950,existing_values<=-600))
    indexmin950=min(indices[0])
    indexmin600=max(indices[0])
 
    #clip value lists to only interesting part of lung
    values_lung=existing_values[int(indexmin950):int(indexmin600)+1]
    total_counts=np.zeros(len(values_lung))
    
    
    
    #counts_lung=total_counts[int(indexmin950):int(indexmin600)+1]
    
    
    
    #fill count list with right numbers
    for i in range(len(hist_count_cat)):
        
        indices=np.where(np.logical_and(hist_count_cat[i][1]>=-950,hist_count_cat[i][1]<=-600))
        indexmin950=min(indices[0])
        indexmin600=max(indices[0])
        
        values_range=hist_count_cat[i][1][int(indexmin950):int(indexmin600)+1]    
        counts_range=hist_count_cat[i][2][int(indexmin950):int(indexmin600)+1]   
        
        pixel_sum=sum(counts_range)
        for j in range(len(counts_range)):
            value=values_range[j]
            count_rel=counts_range[j]  / pixel_sum
            index=np.where(values_lung==value)
            total_counts[index]=total_counts[index]+count_rel
     
    
    
    
  

    total_counts_lung=np.sum(total_counts) 
    counts_lung_rel=total_counts/ total_counts_lung          
      
    GeHist=np.histogram(values_lung,weights=counts_lung_rel, bins=bin_list)    
    return GeHist[0]




def plot_categories(hist_count, categories_numbers, categories,centers, excluded=None):

    probList=[] 
    label=[]
    for i in range(len(categories)):  
        print(i)
        if i != excluded:
            GeHist=returnhist(hist_count, index_numbers=categories_numbers[i]) 
            probList.append(GeHist)
            label.append(categories[i])

   
    
    plt.figure()
    for i in range(len(probList)):
        plt.plot(centers,probList[i]) 
    plt.legend(label)

    plt.xticks(np.ceil(centers))
    plt.xlabel('HU')
    




def plot_1_categorie(hist_count,class_nr,categories_numbers,categories,centers):
    
    index=categories_numbers[class_nr]
    print(index)
    hist_count_cat=np.take(hist_count,index,0)
    plt.figure()
    for i in range(len(hist_count_cat)):
        values=hist_count_cat[i][1]
        counts=hist_count_cat[i][2]
        indices=np.where(np.logical_and(values>=-950,values<=-600))
        indexmin950=min(indices[0])
        indexmin600=max(indices[0])
     
        values_lung=values[int(indexmin950):int(indexmin600)+1]
        counts_lung=counts[int(indexmin950):int(indexmin600)+1]
        
        total_counts_lung=np.sum(counts_lung) 
        counts_lung_rel=counts_lung/ total_counts_lung  
        
        a=np.histogram(values_lung,weights=counts_lung_rel, bins=bin_list)  
        
        plt.plot(centers,a[0],alpha=0.5)
    
    plt.title(categories[class_nr])

    plt.xticks(np.ceil(centers))
    plt.xlabel('HU')



#do final plotting and categorizing

categories, categories_numbers=sort_categories(hist_count,categorie='Manufacturer')
plot_1_categorie(hist_count, 0,categories_numbers,categories,centers)

plot_categories(hist_count, categories_numbers, categories,centers, excluded=3)




plt.figure()
GeHist_utrecht=returnhist(hist_count_utrecht,index_numbers=[0,1])
GeHist_utrecht2=returnhist(hist_count_utrecht,index_numbers=[0])

plt.plot(centers,GeHist_utrecht)
plt.plot(centers,GeHist_utrecht2)



plt.title('Utrecht')


plt.xticks(np.ceil(centers))
plt.xlabel('HU')
plt.legend([categories[0],  categories[1] , categories[2] , categories[4], 'Utrecht'])
#bin_list.append(max(total_values)) 