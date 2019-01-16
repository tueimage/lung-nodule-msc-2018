# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:45:06 2018

@author: s120116
"""
import matplotlib.pyplot as plt
import numpy as np
import CTsliceViewer as slice
import matplotlib.pyplot as plt

path='D:/Data_Linde_thesis/FinalEvaluation/Testing/'


path32=path+'Scale=3.2cm/Image_Data/'
path64=path+'Scale=6.4cm/Image_Data/'


ImNum=str(0)


#small scale
im_pred=np.load(path32+'prediction_im'+ ImNum +'.npy')
im=np.load(path32+'bounding_im'+ ImNum +'.npy')
im_ir=np.load(path32+'bounding_irrel'+ ImNum +'.npy')
im_mask=np.load(path32+'bounding_mask'+ ImNum +'.npy')

slice.multi_slice_viewer(im_pred)
slice.multi_slice_viewer(im)
slice.multi_slice_viewer(im_mask)
slice.multi_slice_viewer(im_ir)



#largescale
im_pred=np.load(path64+'prediction_im'+ ImNum +'.npy')
im=np.load(path64+'bounding_im'+ ImNum +'.npy')
im_ir=np.load(path64+'bounding_irrel'+ ImNum +'.npy')
im_mask=np.load(path64+'bounding_mask'+ ImNum +'.npy')

slice.multi_slice_viewer(im_pred)
slice.multi_slice_viewer(im)
slice.multi_slice_viewer(im_mask)
slice.multi_slice_viewer(im_ir)





path32_eval=path+'Scale=3.2cm/Image_evaluation/'
path64_eval=path+'Scale=6.4cm/Image_evaluation/'


misdet_32=np.load(path32_eval+'MissedDetected.npy')
cordet_32=np.load(path32_eval+'TrueDetected.npy')
misdet_64=np.load(path64_eval+'MissedDetected.npy')
cordet_64=np.load(path64_eval+'TrueDetected.npy')


senslist=np.load('SensList.npy')
fplist=np.load('FPlist.npy')
fplist_cor=np.delete(fplist,38,0)y')


cor_tres=np.sum(cordet,0)
mis_tresh=np.sum(misdet,0)
sens_tresh=cor_tres/(cor_tres+mis_tresh)

fp_tres=np.sum(fplistr,0)/(len(fplist_cor)-3)
fp_tres=np.mean(fplist,0)

sensitivity=sens_tresh
falsepositives=fp_tres




labels=np.load('labellist_val.npy')
predictions=np.load('predictionlist_val.npy')

plt.figure()
plt.scatter(labels, predictions)
plt.xlabel('labels')
plt.ylabel('predictions')

#code to import sensitivity and fp rate, add it to a list and plot it
FROC_test=[]

#this part for each folder
fplist=np.load('FPlist.npy')
fp_tres=np.mean(fplist,0)
falsepositives=fp_tres



FROC_test=FROClist
sensitivity=np.load('sens_tresh.npy')
falsepositives=np.load('fp_tresh.npy')
name='6.4 cm'
FROC_test.append([name,falsepositives,sensitivity])

FROC_test.append([name,falsepositives,sensitivity])
FROC_test[1]=[name,falsepositives,sensitivity]
#FROC_test=FROClist

plt.figure()

    
#for i in range(len(FROC_val)):
 #   handle=plt.plot(FROC_val[i][1],FROC_val[i][2], '-*',color='C'+ str(i+2) ,linewidth=2.0, label=FROC_val[i][0], alpha=1)   
    
for i in range(len(FROC_test)):
    handle=plt.plot(FROC_test[i][1],FROC_test[i][2], '-*', color='C'+str(i),  linewidth=2.0, label=FROC_test[i][0])   
plt.legend()#fontsize='20', loc='lower left')
plt.xlabel('average FP/scan')#,fontsize='20')
plt.ylabel('sensitivity')#,fontsize='20')
plt.xlim((0,25))
plt.ylim((0,1))

plt.tick_params(axis='both')#, labelsize='16')
plt.xticks([0.125,2,4,8,12,16],[0.125,2,4,8,12,16])
#fig = plt.gcf()
#fig.set_size_inches(8, 6)
#fig.savefig('FROCcombi_5.pgf')


from matplotlib2tikz import save as tikz_save
tikz_save("detectionSpectralData.tikz", figureheight='\\figureheight', figurewidth='\\figurewidth',strict=False)


plt.savefig('FROCvalData.png', transparant=True)
import pickle

with open("FROCval.txt", "wb") as fp:   #Pickling
    pickle.dump(FROC_val, fp)

plt.savefig('FROCAllSmall.png', transparant=True)


#load list again
with open("FROCval.txt", "rb") as fp:   # Unpickling
    FROCval = pickle.load(fp)
    
with open("FROCtest.txt", "rb") as fp:   # Unpickling
    FROCtest = pickle.load(fp)    

#save plots and np array FROClist

FROClist2=b+FROClist

