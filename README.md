# Lung Nodules Classification

In this Github the code I developed during my master thesis is given. The purpose of this code is to detect nodules in a CT scan and subsequently to classify them as being benign, malignant or metastases. At the moment only the classification code is completely finished for use, the detection part will follow soon. 

The classification approach I used in my thesis is shown in the figure below. In the top part a neural net is trained using the [LIDC-IDRI database](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI), resulting in malignancy scores for lung nodules. This trained network can subsequently be used as feature extractor for a new dataset (bottom row), and these features can then be classified with a SVM. The code in this github is to apply the pretrained network to a new dataset, thus the bottom row of the figure. 

![alt text](https://github.com/tueimage/lung-nodule-msc-2018/blob/master/ClassificationOverview.png)

The trained neural network (3D conv net) can be downloaded from figshare, and should be put in the folder Models, in order for everything to work:

````
https://figshare.com/articles/Trained_Neural_Network_on_Lung_Nodules/7376360
````

## Data preparation
The code for data preparation is found in the folder named this way.  The data first has to be preprocessed (_Preprocessing.py_), then crops around the nodules have to be made (_CreateNodulesCrops.py_) and at last feature extraction takes place (_FeaturesExtraction.py_). The three scripts are combined in one as: _DataPreparationCombined_, however for troubleshooting the individual files are available as well.

During development of the code I used the package Radio, which is a package specifically for using CT scans & annotations for detection algorithms, and I added my own code to this package in the file _CTImagesCustomBatch.py_. 

````
Khudorozhkov R., Emelyanov K., Koryagin A. RadIO library for data science research of CT images. 2017.
https://github.com/analysiscenter/radio
````

There are a few points which should be noticed when using the code, dependent on the data:

1. At the moment the script is made for DICOM files, it is also possible to load mhd files. For this see the documentation of Radio, and adapt the load function. The DICOM files of the individual slices should be saved per scan in a folder, which are all together in the main folder. Deeper data structures can give problems as the iterator over the data takes the lowest folder level as index name, this should thus not be equal for multiple scans. I used the structure below, which worked fine for all code:

DICOMfolder
  - 00001     -> Containing individual slices for this scan
  - 00002
  - 00003

2. The annotations should be presented in world coordinates in an excel file with the following column headers:
'PatientID', 'CoordZ',  'CoordY', 'CoordX', 'Diameter [mm]', 'LesionID' (lesion id is the number of the nodule in the scan, can be always 1 when there is just one nodule per scan). The order of the columns is not important. There is a folder with an example annotation file available in this git. 
If the names are different this can be changed in the function fetch_nodules_info_generalized from CTImagesCustomBatch. It is also important the the entries of the PatientID column correspond to the foldernames of the dicoms. If this is not the case the same function should be adopted. 
To test the annotations / loading of data _NoduleTest.py_ can be used, which gets one scan through the batch and shows the crops it made, if the nodules are in the center of each box (boxes are shown after each other, so every 16 slices are one crop), everything is correct. Else have a look at 3. 

3. During loading of the DICOMS, I had to adapt the order in which the slices were loaded (descending / ascending) to get correct z-coordinates of the annotations. I am not sure whether this can differ for other sets, but this could be tried when the z-coordinate for the annotations is not correct. This parameters can be changed in _load_dicom_ in the _CTImagesCustomBatch_ in the following line:

````
list_of_dicoms.sort(key=lambda x: int(x.ImagePositionPatient[2]), reverse=False) #change reverse parameter
````

To summarize, the following scripts can run after each other for the data preparation:
````
NoduleTest.py --> To check the annotations / loading of data
TotalDataPreperation --> If this gives errors, check the separate scripts (Preprocessing, CreateNoduleCrops and FeatureExtraction)
````

# SVM classification
Next, the feature vectors can be classified with SVM. The script _SVMclassification.py_ (in folder SVMClassification) can be used for this. For the classification an excel file with diagnosis is necessary, with the columns 'scannum', 'labels', 'patuid'. Also from this file an example is available. The 'patuid' parameters should have a unique number for each patient, if all scans are from different patients, this number can be the same as the scannum. The labels of the groups should be one of: 'benign', 'metastases', 'lung'. Other labels are possible but this then needs to be adapted in the main script _SVMclassification.py_, in the function _bin_labels()_.

The features are loaded and coupled to the patient diagnosis in the function _load_features.py_. This function now assumes that each folder name consists of a number with trailing zeros (as in the folder structure example above), together with the nodule number. To get the diagnosis it thus takes the first 6 characters and converts this to a number. If the folder structure is different, adaptions have to be made to this function. It can be found in the file _HelperFileClassification.py_.

In this script SVM is applied on two group divisions: benign / malignant and benign / lung / malignant. The script results in dataframes with the metrices from the crossvalidation, as well as predictions from the crossvalidations (to make confusion matrices). These are saved in the folder 'Final_Results'. A prefitted SVM model is also applied to the data, which results in predictions for each sample. These are also saved in the folder 'prefitted'.  

If you have any questions regarding the code or want to run it on your own database, I am happy to help with any problems. I would also be very interested in how the method performs on other datasets.
