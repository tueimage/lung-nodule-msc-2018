# -*- coding: utf-8 -*-
"""
Created on Mon May 28 20:35:37 2018

@author: s120116
"""
from radio import CTImagesMaskedBatch
from radio.preprocessing.mask import make_rect_mask_numba
from radio.preprocessing.resize import resize_scipy, resize_pil 
from radio.dataset import  action, inbatch_parallel, DatasetIndex, SkipBatchException
import os
import numpy as np
try:
    from tqdm import tqdm_notebook
except ImportError:
    tqdm_notebook = lambda x: x
import logging
logger = logging.getLogger(__name__) 

#import segmentUtrecht as lung
import PIL
from numba import njit
import pandas as pd
import pydicom as dicom

#check dit
AIR_HU = -2000
DARK_HU = -2000


@njit(nogil=True)
def get_nodules_numba(data, positions, size):
    size = size.astype(np.int64)
 
    out_arr = np.zeros((np.int(positions.shape[0]), size[0], size[1], size[2]))

    n_positions = positions.shape[0]
    for i in range(n_positions):
        out_arr[i, :, :, :] = data[positions[i, 0]: positions[i, 0] + size[0],
                                   positions[i, 1]: positions[i, 1] + size[1],
                                   positions[i, 2]: positions[i, 2] + size[2]]

    return out_arr.reshape(n_positions * size[0], size[1], size[2])
  
def bbox2_3D(img):

  r = np.any(img, axis=(1, 2))
  c = np.any(img, axis=(0, 2))
  z = np.any(img, axis=(0, 1))

  rmin, rmax = np.where(r)[0][[0, -1]]
  cmin, cmax = np.where(c)[0][[0, -1]]
  zmin, zmax = np.where(z)[0][[0, -1]]

  return rmin, rmax, cmin, cmax, zmin, zmax
  
class CTImagesCustomBatch(CTImagesMaskedBatch):
    
    nodules_dtype = np.dtype([('patient_pos', np.int, 1),
                              ('offset', np.int, (3,)),
                              ('img_size', np.int, (3,)),
                              ('nodule_center', np.float, (3,)),
                              ('nodule_size', np.float, (3,)),
                              ('spacing', np.float, (3,)),
                                ('origin', np.float, (3,)),
                            
                              ('malignancy',np.float ,1) ,
                              ('sphericity', np.float, 1),
                              ('margin', np.float, 1),
                              ('spiculation', np.float ,1),
                              ('texture', np.float,1),
                              ('calcification', np.float, 1),
                              ('internal_structure', np.float, 1),
                              ('lobulation',np.float,1),
                              ('subtlety', np.float,1 )]   )      #state, 0=excluded, 1=included  
            
    candidates_dtype = np.dtype([('patient_pos', np.int, 1),
                              ('offset', np.int, (3,)),
                              ('img_size', np.int, (3,)),
                              ('candidate_center', np.float, (3,)),
                              ('spacing', np.float, (3,)),
                              ('origin', np.float, (3,)),
                              ('candidate_label', np.int,1)])
            

  
    components = "images", "masks", "segmentation", "spacing", "origin","nodules"
  
    def __init__(self, index, *args, **kwargs):
        """ Execute Batch construction and init of basic attributes
    
        Parameters
        ----------
        index : Dataset.Index class.
            Required indexing of objects (files).
        """
        super().__init__(index, *args, **kwargs)
        self.segmentation=None
        self.candidates=None
        self.nodulesEval=None
        


        
    def _init_load_blosc(self, **kwargs):
        """ Init-func for load from blosc.
    
        Fills images/masks-components with zeroes if the components are to be updated.
    
        Parameters
        ----------
        **kwargs
                components : str, list or tuple
                    iterable of components names that need to be loaded
        Returns
        -------
        list
            list of ids of batch-items, i.e. series ids or patient ids.
        """
        # fill 'images', 'masks'-comps with zeroes if needed
        skysc_components = {'images', 'masks', 'segmentation'} & set(kwargs['components'])
        self._prealloc_skyscraper_components(skysc_components)
    
        return self.indices
      
    
    def get_pos(self, data, component, index,dst=None):
        """ Return a positon of an item for a given index in data
        or in self.`component`.
        
        Fetch correct position inside batch for an item, looks for it
        in `data`, if provided, or in `component` in self.
        
        Parameters
        ----------
        data : None or ndarray
            data from which subsetting is done.
            If None, retrieve position from `component` of batch,
            if ndarray, returns index.
        component : str
            name of a component, f.ex. 'images'.
            if component provided, data should be None.
        index : str or int
            index of an item to be looked for.
            may be key from dataset (str)
            or index inside batch (int).
        
        Returns
        -------
        int
            Position of item
        
        Notes
        -----
        This is an overload of get_pos from base Batch-class,
        see corresponding docstring for detailed explanation.
        """
        if data is None:
            ind_pos = self._get_verified_pos(index)
            if component in ['images', 'masks','segmentation']:
                return slice(self.lower_bounds[ind_pos], self.upper_bounds[ind_pos])
            else:
                return slice(ind_pos, ind_pos + 1)
        else:
            return index  
    
    
    
    @inbatch_parallel(init='indices', post='_post_default', target='threads')
    def _load_dicom(self, patient_id, **kwargs):
        """ Read dicom file, load 3d-array and convert to Hounsfield Units (HU).

        Notes
        -----
        Conversion to hounsfield unit scale using meta from dicom-scans is performed.
        """
        # put 2d-scans for each patient in a list
        patient_pos = self.index.get_pos(patient_id)
    
        patient_folder = (self.index.get_fullpath(patient_id)) #.replace(str(patient_id),"")

        
       # patient_folder=patient_folder.replace(str(patient_id),"") #added for utrecht data
        
        list_of_dicoms = [dicom.read_file(os.path.join(patient_folder, s))
                          for s in os.listdir(patient_folder)]
        

        list_of_dicoms.sort(key=lambda x: int(x.ImagePositionPatient[2]), reverse=False) #adapted from True

        dicom_slice = list_of_dicoms[0]
        intercept_pat = dicom_slice.RescaleIntercept
        slope_pat = dicom_slice.RescaleSlope
        zspacing=np.abs(list_of_dicoms[10].ImagePositionPatient[2] - list_of_dicoms[11].ImagePositionPatient[2]) #adapted!!!!
        self.spacing[patient_pos, ...] = np.asarray([zspacing,
                                                     float(dicom_slice.PixelSpacing[0]),
                                                     float(dicom_slice.PixelSpacing[1])], dtype=np.float)

        self.origin[patient_pos, ...] = np.asarray([float(dicom_slice.ImagePositionPatient[2]),
                                                    float(dicom_slice.ImagePositionPatient[1]),
                                                    float(dicom_slice.ImagePositionPatient[0])], dtype=np.float)

        patient_data = np.stack([s.pixel_array for s in list_of_dicoms]).astype(np.int16)

        patient_data[patient_data == AIR_HU] = 0 #adapted for Spectral Reconstructions

        # perform conversion to HU
        if slope_pat != 1:
            patient_data = slope_pat * patient_data.astype(np.float64)
            patient_data = patient_data.astype(np.int16)

        patient_data += np.int16(intercept_pat)
        return patient_data
    
  
    def sample_negatives(self, num_nodules, nodule_size, histo=None):
            """ Sample random nodules positions in CTImagesBatchMasked.
    
            Samples random nodules positions in ndarray. Each nodule have shape
            defined by `nodule_size`. If size of patients' data along z-axis
            is not the same for different patients, NotImplementedError will be raised.
    
            Parameters
            ----------
            num_nodules : int
                number of nodules to sample from dataset.
            nodule_size : ndarray(3, )
                crop shape along (z,y,x).
            histo : tuple
                np.histogram()'s output.
                3d-histogram, represented by tuple (bins, edges).
    
            Returns
            -------
            ndarray
                ndarray(num_nodules, 3). 1st array's dim is an index of sampled
                nodules, 2nd points out start positions (integers) of nodules
                in batch `skyscraper`.
            """
            all_indices = np.arange(len(self))
           
            sampled_indices = np.random.choice(
                all_indices, num_nodules, replace=True)
            
            #sort indices  to loop over scans one by one
            sampled_indices=np.sort(sampled_indices)
            
            offset = np.zeros((num_nodules, 3))
            offset[:, 0] = self.lower_bounds[sampled_indices] #offset for z axis is what has to be added to get coordinate in skyscraper
          
    
       
            #sampler: get random samples between 0 and 1
            sampler = lambda size: np.random.rand(size, 3)
        
            sample_list=[]
            
            #loop over each scan
            for scan_nr in range(len(self)):
                #count the number of needed samples for this scan
                num_scan=list(sampled_indices).count(scan_nr) 
                
                zmin,zmax,ymin,ymax,xmin,xmax=bbox2_3D(self[scan_nr].segmentation)
                (z,y,x)=self[scan_nr].images.shape
                red_box=(zmin+(z-zmax),ymin+(y-ymax),xmin+(x-xmax))
                data_shape = self.images_shape[scan_nr, :]
              
                offset_box=(zmin,ymin,xmin)
                #while this number is not reached
               
                num_samples=0
                max_trial=num_scan*200
                i=0
                while num_samples < num_scan:
                #calculate bounding box coordinates
                   # print(num_samples)
                    sample=sampler(size=1) * (data_shape-red_box)+offset_box
                    sample=sample[0,:]
    
                    
                    if self[scan_nr].segmentation[int(sample[0]),int(sample[1]),int(sample[2])] == 1: #als sample binnen long zit
                     
                        
                        start_pix = (np.rint(sample) - np.rint(nodule_size / 2))

            
                        end_pix = start_pix + nodule_size
                
                
        
                        zslice=slice(int(start_pix[0]),int(end_pix[0]))
                        yslice=slice(int(start_pix[1]),int( end_pix[1]))
                        xslice=slice(int(start_pix[2]),int(end_pix[2]))
             
                        
                        patch=self[scan_nr].masks[zslice,yslice,xslice]
                        
                        if np.count_nonzero(patch)==0: #if no nodule is present in scan
                        
                            #sample=(sample-np.ceil(nodule_size-1/2)).astype(int) #get upper left corner of sample
                            sample_list.append(start_pix)
                            num_samples=num_samples+1
                    i=i+i
                    if i > max_trial:
                        print('Maximum iterations reached, not enough negative samples possible for this scan')
                      #  print(self.indices[scan_nr])
                 
               

              
                #eerst midden coordinaten bepalen, deze shiften om links boven hoek te krijgen
                #offset box: shift of  coordinates to get in box
              
            
         
            #shift 
            #until here looping over scans
            
           #shift all  to get left upper corner
           # slices.multi_slice_viewer(self.images)  
            sample_list_array=np.asarray(sample_list)
            #plt.scatter(sample_list_array[:,2],sample_list_array[:,1]) 
            #add z dimension offset for stack of scans and remove outer dimension
            sample_list_final=np.squeeze(np.asarray(sample_list_array+ offset, dtype=np.int))
          
            
            #delete segmentations to  reduce memory
          
 
            return sample_list_final, sampled_indices
#
    @action
    def fetch_nodules_info_general(self, nodules=None, nodules_records=None, update=False, images_loaded=True):
        """Extract nodules' info from nodules into attribute self.nodules.      

        """
        if self.nodules is not None and not update:
            logger.warning("Nodules have already been extracted. " +
                           "Put update argument as True for refreshing")
            return self

        if nodules_records is not None:
            # load from record-array
            self.nodules = nodules_records

        else:
            #if necessary, adapt here the name of the patient / iterating
            #entries for PatientID should correspond to folder names of dicoms / preprocessed images
            nodules_df = nodules.set_index("PatientID").astype(str)


            unique_indices = nodules_df.index.unique()

            inter_index = np.intersect1d(unique_indices,self.indices)
  
            #adapt if other headers necessary
            nodules_df = nodules_df.loc[inter_index,
                                        ["CoordZ", "CoordY",
                                         "CoordX",  "Diameter [mm]", "LesionID"]]
            
            if np.any((nodules_df.isnull()==True)):
                print('No Nodules in patient')
                return self
                
            num_nodules = nodules_df.shape[0]
            self.nodules = np.rec.array(np.zeros(num_nodules,
                                                 dtype=self.nodules_dtype))
          
            counter = 0
            for (pat_id,coordz, coordy, coordx, diameter, lesionid) in nodules_df.itertuples():
        

                pat_pos = self.index.get_pos(pat_id)
                self.nodules.patient_pos[counter] = pat_pos
                self.nodules.nodule_center[counter, :] = np.array([coordz,
                                                                   coordy,
                                                                   coordx])
                self.nodules.nodule_size[counter, :] = np.array([diameter, diameter, diameter])
                self.nodules.malignancy[counter]=lesionid
                counter += 1

        self._refresh_nodules_info(images_loaded)
     
        return self

    
    @action
    def fetch_nodules_info_malignancy(self, nodules=None, nodules_records=None, update=False, images_loaded=True):
        """Extract nodules' info from nodules into attribute self.nodules.      

        """
        if self.nodules is not None and not update:
            logger.warning("Nodules have already been extracted. " +
                           "Put update argument as True for refreshing")
            return self

        if nodules_records is not None:
            # load from record-array
            self.nodules = nodules_records

        else:
            # assume that nodules is supplied and load from it
            required_columns = np.array(['seriesuid', 
                                         'coord_x', 'coord_y', 'coord_z', 'diameter','malscore',
                                         'sphericiy', 'margin', 'spiculation', 'texture',
                                         'calcification', 'internal_structure', 'lobulation',
                                         'subtlety'])

            if not (isinstance(nodules, pd.DataFrame) and np.all(np.in1d(required_columns, nodules.columns))):
                raise ValueError(("Argument 'nodules' must be pandas DataFrame"
                                  + " with {} columns. Make sure that data provided"
                                  + " in correct format.").format(required_columns.tolist()))
            nodules[['seriesuid']]=nodules[['seriesuid']].astype(str)
            nodules_df = nodules.set_index('seriesuid')
            
            
            unique_indices = nodules_df.index.unique()
            inter_index = np.intersect1d(unique_indices, self.indices)
            
            nodules_df = nodules_df.loc[inter_index,
                                        ["coord_z", "coord_y",
                                         "coord_x",  "diameter","malscore",
                                         "sphericiy", "margin", "spiculation", "texture",
                                         "calcification", "internal_structure", "lobulation",
                                         "subtlety"]]
          
            num_nodules = nodules_df.shape[0]
            self.nodules = np.rec.array(np.zeros(num_nodules,
                                                 dtype=self.nodules_dtype))
          
            counter = 0
            for (pat_id, coordz, coordy, coordx, diam, malscore, 
                 sphericity, margin ,spiculatio, texture, calcification,
            internal_structure, lobulation, subtlety) in nodules_df.itertuples():
                pat_pos = self.index.get_pos(pat_id)
                self.nodules.patient_pos[counter] = pat_pos
                self.nodules.nodule_center[counter, :] = np.array([coordz,
                                                                   coordy,
                                                                   coordx])
                self.nodules.nodule_size[counter, :] = np.array([diam, diam, diam])
                self.nodules.malignancy[counter]= malscore
                self.nodules.sphericity[counter]=sphericity
                self.nodules.margin[counter]=margin
                self.nodules.spiculation[counter]=spiculatio
                self.nodules.texture[counter]=texture
                self.nodules.calcification[counter]=calcification
                self.nodules.internal_structure[counter]=internal_structure
                self.nodules.lobulation[counter]=lobulation
                self.nodules.subtlety[counter]=subtlety
                counter += 1

        self._refresh_nodules_info(images_loaded)
     
        return self
    
 

    @action
    def sample_nodules(self, batch_size, nodule_size=(32, 64, 64), share=0.8, variance=None,        # pylint: disable=too-many-locals, too-many-statements
                       mask_shape=None, histo=None, data='nonUtrecht'):
        """ Sample random crops of `images` and `masks` from batch.

        Create random crops, both with and without nodules in it, from input batch.

        Parameters
        ----------
        batch_size : int
            number of nodules in the output batch. Required,
            if share=0.0. If None, resulting batch will include all
            cancerous nodules.
        nodule_size : tuple, list or ndarray of int
            crop shape along (z,y,x).
        share : float
            share of cancer crops in the batch.
            if input CTImagesBatch contains less cancer
            nodules than needed random nodules will be taken.
        variance : tuple, list or ndarray of float
            variances of normally distributed random shifts of
            nodules' start positions.
        mask_shape : tuple, list or ndarray of int
            size of `masks` crop in (z,y,x)-order. If not None,
            crops with masks would be of mask_shape.
            If None, mask crop shape would be equal to crop_size.
        histo : tuple
            np.histogram()'s output.
            Used for sampling non-cancerous crops.

        Returns
        -------
        Batch
            batch with cancerous and non-cancerous crops in a proportion defined by
            `share` with total `batch_size` nodules. If `share` == 1.0, `batch_size`
            is None, resulting batch consists of all cancerous crops stored in batch.
        """
        # make sure that nodules' info is fetched and args are OK
        
        
        if self.nodules is None:
            #if data=='Utrecht':
             #   raise SkipBatchException('Batch without nodules cannot be passed further through the workflow')
        
            raise AttributeError("Info about nodules location must " +
                                 "be loaded before calling this method")
        if variance is not None:
            variance = np.asarray(variance, dtype=np.int)
            variance = variance.flatten()
            if len(variance) != 3:
                logger.warning('Argument variance be np.array-like' +
                               'and has shape (3,). ' +
                               'Would be used no-scale-shift.')
                variance = None

        if share == 0.0 and batch_size is None:
            raise ValueError('Either supply batch_size or set share to positive number')

        # pos of batch-items that correspond to crops
        crops_indices = np.zeros(0, dtype=np.int16)

        # infer the number of cancerous nodules and the size of batch
        batch_size = batch_size if batch_size is not None else 1.0 / share * self.num_nodules
        cancer_n = int(share * batch_size)
        batch_size = int(batch_size)
        cancer_n = self.num_nodules if cancer_n > self.num_nodules else cancer_n

        if batch_size == 0:
            raise SkipBatchException('Batch of zero size cannot be passed further through the workflow')

        # choose cancerous nodules' starting positions
        nodule_size = np.asarray(nodule_size, dtype=np.int)
        if self.num_nodules == 0:
            cancer_nodules = np.zeros((0, 3))
        else:
            # adjust cancer nodules' starting positions s.t. nodules fit into
            # scan-boxes
            
            
                                              
           
            cancer_nodules = self._fit_into_bounds(
                        nodule_size, variance=variance)
            if data== 'Utrecht':
                sample_indices = np.arange(self.num_nodules)
                                                              
            else:    
                sample_indices = np.random.choice(np.arange(self.num_nodules),
                                              size=cancer_n, replace=False)
            # randomly select needed number of cancer nodules (their starting
            # positions)
            

            cancer_nodules = cancer_nodules[sample_indices, :]

            # store scans-indices for chosen crops
            cancerous_indices = self.nodules.patient_pos[sample_indices].reshape(-1)
            crops_indices = np.concatenate([crops_indices, cancerous_indices])

        nodules_st_pos = cancer_nodules

        # if non-cancerous nodules are needed, add random starting pos
        if batch_size - cancer_n > 0:
            # sample starting positions for (most-likely) non-cancerous crops
            random_nodules, random_indices = self.sample_negatives(batch_size - cancer_n,
                                                                        nodule_size, histo=histo)
         
            # concat non-cancerous and cancerous crops' starting positions
            nodules_st_pos = np.vstack([nodules_st_pos, random_nodules]).astype(
                np.int)  # pylint: disable=no-member
            
            # store scan-indices for randomly chose crops
            crops_indices = np.concatenate([crops_indices, random_indices])

        # obtain nodules' scans by cropping from self.images
        images = get_nodules_numba(self.images, nodules_st_pos, nodule_size)

        # if mask_shape not None, compute scaled mask for the whole batch
        # scale also nodules' starting positions and nodules' shapes
        if mask_shape is not None:
            scale_factor = np.asarray(mask_shape) / np.asarray(nodule_size)
            batch_mask_shape = np.rint(
                scale_factor * self.images_shape[0, :]).astype(np.int)
            batch_mask = self.fetch_mask(batch_mask_shape)
            nodules_st_pos = np.rint(
                scale_factor * nodules_st_pos).astype(np.int)
        else:
            batch_mask = self.masks
            mask_shape = nodule_size

        # crop nodules' masks
        masks = get_nodules_numba(batch_mask, nodules_st_pos, mask_shape)

        # build nodules' batch
        bounds = np.arange(batch_size + 1) * nodule_size[0]
        crops_spacing = self.spacing[crops_indices]
        offset = np.zeros((batch_size, 3))
        offset[:, 0] = self.lower_bounds[crops_indices]
        
        #for lidc-idri data migth need a '+'
        crops_origin = self.origin[crops_indices] - crops_spacing * (nodules_st_pos - offset)

        if data == 'Utrecht':
#             crops_origin_z = self.origin[crops_indices] - crops_spacing * (nodules_st_pos - offset)
#             for i, sub in enumerate(crops_origin):
#                 sub[0]=crops_origin_z[i][0]
             names_gen = zip(self.indices[crops_indices], [str(e) for e in self.nodules.malignancy])
        else:       
            names_gen = zip(self.indices[crops_indices], self.make_indices(batch_size))
      #  ix_batch = ['_'.join([prefix[:-6], random_str]) for prefix, random_str in names_gen] for own data
        ix_batch = ['_'.join([prefix, random_str]) for prefix, random_str in names_gen]

        nodules_batch = type(self)(DatasetIndex(ix_batch))
        nodules_batch._init_data(images=images, bounds=bounds, spacing=crops_spacing, origin=crops_origin, masks=masks)  # pylint: disable=protected-access

        # set nodules info in nodules' batch
       
        nodules_records = [self.nodules[self.nodules.patient_pos == crop_pos] for crop_pos in crops_indices]
      
        new_patient_pos = []
        for i, records in enumerate(nodules_records):
            new_patient_pos += [i] * len(records)
        new_patient_pos = np.array(new_patient_pos)
        nodules_records = np.concatenate(nodules_records)
        nodules_records = nodules_records.view(np.recarray)
        nodules_records.patient_pos = new_patient_pos
      
  
       
  
        # leave out nodules with zero-intersection with crops' boxes
        nodules_batch.fetch_nodules_info_general(nodules_records=nodules_records)
        
        nodules_batch._filter_nodules_info()                                                     # pylint: disable=protected-access

        return nodules_batch
#    
#    def _filter_nodules_info_utrecht(self): #does not work yet, probablity new origin of Z-direcito not working
#        """ Filter record-array self.nodules s.t. only records about cancerous nodules
#        that have non-zero intersection with scan-boxes be present.
#    
#        Notes
#        -----
#        can be called only after execution of fetch_nodules_info and _refresh_nodules_info
#        """
#        # nodules start and trailing pixel-coords
#        print(self.nodules.nodule_center)
#        print(self.nodules.origin)
#        print(self.nodules.spacing)
#        center_pix = np.abs(self.nodules.nodule_center - self.nodules.origin) / self.nodules.spacing
#        for i in range(len(center_pix)):
#            center_pix[i][0]=self.images.shape[0]-self.nodules.nodule_center[i][0]
#       
#        start_pix = center_pix - np.rint(self.nodules.nodule_size / self.nodules.spacing / 2)
#        
#        
#        start_pix = np.rint(start_pix).astype(np.int)
#        end_pix = start_pix + np.rint(self.nodules.nodule_size / self.nodules.spacing)
#    
#        # find nodules with no intersection with scan-boxes
#        nods_images_shape = self.images_shape[self.nodules.patient_pos]
#        start_mask = np.any(start_pix >= nods_images_shape, axis=1)
#        end_mask = np.any(end_pix <= 0, axis=1)
#        zero_mask = start_mask | end_mask
#    
#        # filter out such nodules
#        self.nodules = self.nodules[~zero_mask]

    @action
    def sample_candidates(self, batch_size, nodule_size=(32, 64, 64), type_cand='FPred'):       # pylint: disable=too-many-locals, too-many-statements):
        """ Sample random crops of `images` and `masks` from batch.

        Create random crops, both with and without nodules in it, from input batch.

        Parameters
        ----------
        batch_size : int
            number of nodules in the output batch. Required,
            if share=0.0. If None, resulting batch will include all
            cancerous nodules.
        nodule_size : tuple, list or ndarray of int
            crop shape along (z,y,x).
        share : float
            share of cancer crops in the batch.
            if input CTImagesBatch contains less cancer
            nodules than needed random nodules will be taken.
        variance : tuple, list or ndarray of float
            variances of normally distributed random shifts of
            nodules' start positions.
        mask_shape : tuple, list or ndarray of int
            size of `masks` crop in (z,y,x)-order. If not None,
            crops with masks would be of mask_shape.
            If None, mask crop shape would be equal to crop_size.
        histo : tuple
            np.histogram()'s output.
            Used for sampling non-cancerous crops.

        Returns
        -------
        Batch
            batch with cancerous and non-cancerous crops in a proportion defined by
            `share` with total `batch_size` nodules. If `share` == 1.0, `batch_size`
            is None, resulting batch consists of all cancerous crops stored in batch.
        """
       
        num_candidates=len(self.candidates.candidate_center)
        # make sure that nodules' info is fetched and args are OK
        if self.candidates is None:
            raise AttributeError("Info about nodules location must " +
                                 "be loaded before calling this method")
  

        # pos of batch-items that correspond to crops
        #crops_indices = np.zeros(0, dtype=np.int16)

        # infer the number of cancerous nodules and the size of batch
        batch_size = batch_size if batch_size is not None else   num_candidates
        batch_size = int(batch_size)


       

        # choose cancerous nodules' starting positions
        nodule_size = np.asarray(nodule_size, dtype=np.int)
        
        if num_candidates == 0:
            candidates = np.zeros((0, 3))
            print('No candidates/ FP in this batch')
            raise SkipBatchException('Batch of zero size cannot be passed further through the workflow')
            
        else:
            # adjust cancer nodules' starting positions s.t. nodules fit into
            # scan-boxes
            candidates = self._fit_candidates_into_bounds(
                nodule_size, type_cand = type_cand)
            
            if len(candidates) == 0:
                print('No candidates/ FP in this batch')
                raise SkipBatchException('Batch of zero size cannot be passed further through the workflow')
            
            # randomly select needed number of cancer nodules (their starting
            # positions)
            
            batch_size=min(batch_size,len(candidates))
            
            sample_indices = np.random.choice(np.arange(len(candidates)),
                                              batch_size, replace=False)
            candidates = candidates[sample_indices, :]

            # store scans-indices for chosen crops
           # cancerous_indices = self.nodules.patient_pos[0].reshape(-1)
          #  crops_indices = np.concatenate([crops_indices, cancerous_indices])
      #  slices.multi_slice_viewer(self.images)
       
    
            
        nodules_st_pos = candidates

        # obtain nodules' scans by cropping from self.images
        
        images = get_nodules_numba(self.images, nodules_st_pos, nodule_size)

    
        # crop nodules' masks
        masks = get_nodules_numba(self.masks, nodules_st_pos, nodule_size)

       
        
        crops_indices=np.zeros(batch_size).astype(int)
        
        # build nodules' batch
        bounds = np.arange(batch_size + 1) * nodule_size[0]
        crops_spacing = self.spacing[crops_indices]
        offset = np.zeros((batch_size, 3))
        offset[:, 0] = self.lower_bounds[crops_indices]
       
        crops_origin = self.origin[crops_indices] + crops_spacing * (nodules_st_pos - offset)
       
    
        names_gen = zip(self.indices[crops_indices], self.make_indices(batch_size))
        ix_batch = ['_'.join([prefix, random_str]) for prefix, random_str in names_gen]
    
        nodules_batch = type(self)(DatasetIndex(ix_batch))
        nodules_batch._init_data(images=images, bounds=bounds, spacing=crops_spacing, origin=crops_origin, masks=masks)  # pylint: disable=protected-access

            # pylint: disable=protected-access
        # set nodules info in nodules' batch
        candidate_records = [self.candidates[self.candidates.patient_pos == 0] ]
        new_patient_pos = []
        for i, records in enumerate(candidate_records):
            new_patient_pos += [i] * len(records)
        new_patient_pos = np.array(new_patient_pos)
        candidate_records = np.concatenate(candidate_records)
        candidate_records = candidate_records.view(np.recarray)
        candidate_records.patient_pos = new_patient_pos
        nodules_batch.fetch_candidate_info(candidate_records=candidate_records)

        # leave out nodules with zero-intersection with crops' boxes
       # nodules_batch._filter_nodules_info()         
        return nodules_batch
    
    @action
    def fetch_candidate_info(self, candidates=None, candidate_records=None, update=False, images_loaded=True):
  
        if self.candidates is not None and not update:
            logger.warning("Nodules have already been extracted. " +
                           "Put update argument as True for refreshing")
            return self
        

        if candidate_records is not None:
            # load from record-array
            self.candidates = candidate_records

        else:
            # assume that nodules is supplied and load from it
            required_columns = np.array(['seriesuid', 
                                         'coordZ', 'coordY', 'coordX','class'])
          
            if not (isinstance(candidates, pd.DataFrame) and np.all(np.in1d(required_columns, candidates.columns))):
                raise ValueError(("Argument 'nodules' must be pandas DataFrame"
                                  + " with {} columns. Make sure that data provided"
                                  + " in correct format.").format(required_columns.tolist()))

            candidates_df = candidates.set_index('seriesuid')

            unique_indices = candidates_df.index.unique()
            inter_index = np.intersect1d(unique_indices, self.indices) #welke indices zijn nu nodig
          
            candidates_df = candidates_df.loc[inter_index, #voor die index, pak alles er bij
                                        ["coordZ", "coordY",
                                         "coordX", "class"]]

            num_candidates = candidates_df.shape[0]
         
            self.candidates = np.rec.array(np.zeros(num_candidates,
                                                 dtype=self.candidates_dtype))
            counter = 0
            for pat_id, coordz, coordy, coordx, label in candidates_df.itertuples():
                pat_pos = self.index.get_pos(pat_id)
                self.candidates.patient_pos[counter] = pat_pos
                self.candidates.candidate_center[counter, :] = np.array([coordz,
                                                                   coordy,
                                                                   coordx])
                self.candidates.candidate_label[counter] = label
                counter += 1

        self._refresh_candidates_info(images_loaded)
        return self
     #this funciton should be everywhere where refresh candidates is no, but for now just dont change these
    #values after calling candidates info
   
    def _refresh_candidates_info(self, images_loaded=True):
        """ Refresh self.nodules attributes [spacing, origin, img_size, bias].

        This method is called to update [spacing, origin, img_size, bias]
        attributes of self.nodules because batch's inner data has changed,
        e.g. after resize.

        Parameters
        ----------
        images_loaded : bool
            if True, assumes that `_bounds` attribute is computed,
            i.e. either `masks` and/or `images` are loaded.
        """
        if images_loaded:
            self.candidates.offset[:, 0] = self.lower_bounds[
                self.candidates.patient_pos]
            self.candidates.img_size = self.images_shape[
                self.candidates.patient_pos, :]

        self.candidates.spacing = self.spacing[self.candidates.patient_pos, :]
        self.candidates.origin = self.origin[self.candidates.patient_pos, :]
       
#    @action
#    def create_mask_utrecht(self):
#        """ Create `masks` component from `nodules` component.
#
#        Notes
#        -----
#        `nodules` must be not None before calling this method.
#        see :func:`~radio.preprocessing.ct_masked_batch.CTImagesMaskedBatch.fetch_nodules_info`
#        for more details.
#        
#        X and Y component are calculated using spacing & origin, Z component with slice 
#        """
#        self.masks = np.zeros_like(self.images)
#        if self.nodules is None:
#            logger.warning("Info about nodules location must " +
#                           "be loaded before calling this method. " +
#                           "Nothing happened.")
#            return self
#        
#
#        center_pix = np.abs(self.nodules.nodule_center -
#                            self.nodules.origin) / self.nodules.spacing
#                            
#        print(center_pix) 
#        for i in range(len(center_pix))     :
#            center_pix[i][0]=self.images.shape[0]-self.nodules.nodule_center[i][0]
#        print(center_pix)    
#        start_pix = (center_pix - np.rint(self.nodules.nodule_size /
#                                          self.nodules.spacing / 2))
#        start_pix = np.rint(start_pix).astype(np.int)
#        make_mask_numba(self.masks, self.nodules.offset,
#                        self.nodules.img_size + self.nodules.offset, start_pix,
#                        np.rint(self.nodules.nodule_size / self.nodules.spacing))
#
#        return self    
    @action #creates mask of nodules but puts it in segmentation component
    def create_mask_irrelevant(self):
        """ Create `masks` component from `nodules` component.

        Notes
        -----
        `nodules` must be not None before calling this method.
        see :func:`~radio.preprocessing.ct_masked_batch.CTImagesMaskedBatch.fetch_nodules_info`
        for more details.
        """
        if self.nodules is None:
            logger.warning("Info about nodules location must " +
                           "be loaded before calling this method. " +
                           "Nothing happened.")
        self.segmentation = np.zeros_like(self.images)

        center_pix = np.abs(self.nodules.nodule_center -
                            self.nodules.origin) / self.nodules.spacing
        start_pix = (center_pix - np.rint(self.nodules.nodule_size /
                                          self.nodules.spacing / 2))
        start_pix = np.rint(start_pix).astype(np.int)
        make_rect_mask_numba(self.segmentation, self.nodules.offset,
                        self.nodules.img_size + self.nodules.offset, start_pix,
                        np.rint(self.nodules.nodule_size / self.nodules.spacing))

        return self
    @action
    def fetch_nodulesEval_info(self, nodules=None, nodules_records=None, update=False, images_loaded=True):
        """Extract nodules' info from nodules into attribute self.nodules.

        Parameters
        ----------
        nodules : pd.DataFrame
            contains:
             - 'seriesuid': index of patient or series.
             - 'coordZ','coordY','coordX': coordinates of nodules center.
             - 'diameter_mm': diameter, in mm.
        nodules_records : np.recarray
            if not None, should
            contain the same fields as describe in Note.
        update : bool
            if False, warning appears to remind that nodules info
            will be earased and recomputed.
        images_loaded : bool
            if True, i.e. `images` component is loaded,
            and image_size is used to compute
            correct nodules location inside `skyscraper`.
            If False, it doesn't update info of location
            inside `skyscraper`.

        Returns
        -------
        batch

        Notes
        -----
        Run this action only after  :func:`~radio.CTImagesBatch.load`.
        The method fills in record array self.nodules that contains the following information about nodules:
                               - self.nodules.nodule_center -- ndarray(num_nodules, 3) centers of
                                 nodules in world coords;
                               - self.nodules.nodule_size -- ndarray(num_nodules, 3) sizes of
                                 nodules along z, y, x in world coord;
                               - self.nodules.img_size -- ndarray(num_nodules, 3) sizes of images of
                                 patient data corresponding to nodules;
                               - self.nodules.offset -- ndarray(num_nodules, 3) of biases of
                                 patients which correspond to nodules;
                               - self.nodules.spacing -- ndarray(num_nodules, 3) of spacinf attribute
                                 of patients which correspond to nodules;
                               - self.nodules.origin -- ndarray(num_nodules, 3) of origin attribute
                                 of patients which correspond to nodules.
                               - self.nodules.patient_pos -- ndarray(num_nodules, 1) refers to
                                 positions of patients which correspond to stored nodules.

        """
        if self.nodules is not None and not update:
            logger.warning("Nodules have already been extracted. " +
                           "Put update argument as True for refreshing")
            return self

        if nodules_records is not None:
            # load from record-array
            self.nodules = nodules_records

        else:
            # assume that nodules is supplied and load from it
            required_columns = np.array(['seriesuid', 'diameter_mm',
                                         'coordZ', 'coordY', 'coordX', 'state'])

            if not (isinstance(nodules, pd.DataFrame) and np.all(np.in1d(required_columns, nodules.columns))):
                raise ValueError(("Argument 'nodules' must be pandas DataFrame"
                                  + " with {} columns. Make sure that data provided"
                                  + " in correct format.").format(required_columns.tolist()))

            nodules_df = nodules.set_index('seriesuid')

            unique_indices = nodules_df.index.unique()
            inter_index = np.intersect1d(unique_indices, self.indices)
            nodules_df = nodules_df.loc[inter_index,
                                        ["coordZ", "coordY",
                                         "coordX", "diameter_mm", "state"]]

            num_nodules = nodules_df.shape[0]
            self.nodules = np.rec.array(np.zeros(num_nodules,
                                                 dtype=self.nodules_dtype))
            counter = 0
            for pat_id, coordz, coordy, coordx, diam, state in nodules_df.itertuples():
                pat_pos = self.index.get_pos(pat_id)
                self.nodules.patient_pos[counter] = pat_pos
                self.nodules.nodule_center[counter, :] = np.array([coordz,
                                                                   coordy,
                                                                   coordx])
                self.nodules.nodule_size[counter, :] = np.array([diam, diam, diam])
                self.nodules.state[counter]=state
                counter += 1

        self._refresh_nodules_info(images_loaded)
        return self
   
        
        
    def _fit_into_bounds(self, size, variance=None):
        """ Fetch start voxel coordinates of all nodules.

        Get start voxel coordinates of all nodules in batch.
        Note that all nodules are considered to have
        fixed same size defined by argument size: if nodule is out of
        patient's 3d image bounds than it's center is shifted to border.

        Parameters
        ----------
        size : list or tuple of ndarrays
            ndarray(3, ) with diameters of nodules in (z,y,x).
        variance : ndarray(3, )
            diagonal elements of multivariate normal distribution,
            for sampling random shifts along (z,y,x) correspondingly.

        Returns
        -------
        ndarray
            start coordinates (z,y,x) of all nodules in batch.
        """
        size = np.array(size, dtype=np.int)

        center_pix = np.abs(self.nodules.nodule_center -
                            self.nodules.origin) / self.nodules.spacing
        start_pix = (np.rint(center_pix) - np.rint(size / 2))
     
        if variance is not None:
           # start_pix += np.random.multivariate_normal(np.zeros(3),
            #                                           np.diag(variance),
             #                                          self.nodules.patient_pos.shape[0])
            
            max_var=np.array(variance /2 )
           
            min_var=np.negative(max_var)
      
        
           
            var = np.random.uniform((min_var),(max_var),start_pix.shape)
            
            final_var=np.round(var)
         
            start_pix+=final_var
           # variation=random_variation*(np.self.nodules.nodule.size)
            
        end_pix = start_pix + size

        bias_upper = np.maximum(end_pix - self.nodules.img_size, 0)
        start_pix -= bias_upper
        end_pix -= bias_upper

        bias_lower = np.maximum(-start_pix, 0)
        start_pix += bias_lower
        end_pix += bias_lower
        

        
        return (start_pix + self.nodules.offset).astype(np.int)
    

#    
    def _fit_candidates_into_bounds(self, size, type_cand):
        """ Fetch start voxel coordinates of all nodules.

        Get start voxel coordinates of all nodules in batch.
        Note that all nodules are considered to have
        fixed same size defined by argument size: if nodule is out of
        patient's 3d image bounds than it's center is shifted to border.

        Parameters
        ----------
        size : list or tuple of ndarrays
            ndarray(3, ) with diameters of nodules in (z,y,x).
        variance : ndarray(3, )
            diagonal elements of multivariate normal distribution,
            for sampling random shifts along (z,y,x) correspondingly.

        Returns
        -------
        ndarray
            start coordinates (z,y,x) of all nodules in batch.
        """
        size = np.array(size, dtype=np.int)


       # center_pix = np.abs(self.candidates.candidate_center -
        #                    self.candidates.origin) / self.candidates.spacing
         
        start_pix_list=[]
     

     
        for i in range(len(self.candidates.candidate_center)):
            if self.candidates.candidate_label[i] == 0:
                if type_cand=='FPred':
                    center_pix=self.candidates.candidate_center[i]
                elif type_cand== 'LunCand':   
                    center_pix = np.abs((self.candidates.candidate_center[i] -
                            self.candidates.origin[i]) / self.candidates.spacing[i])
             
                else:
                    raise ValueError("Argument type_cand must have one of values: "
                             + "'FPred', r 'LunCand")
                    
                start_pix = (np.rint(center_pix) - np.rint(size / 2))

            
                end_pix = start_pix + size
                
                img_size_array=np.array(self.candidates.img_size[i])
                bias_upper = np.maximum(end_pix - img_size_array, 0)
                start_pix -= bias_upper
                end_pix -= bias_upper

                bias_lower = np.maximum(-start_pix, 0)
                start_pix += bias_lower
                end_pix += bias_lower
        
                zslice=slice(int(start_pix[0]),int(end_pix[0]))
                yslice=slice(int(start_pix[1]),int( end_pix[1]))
                xslice=slice(int(start_pix[2]),int(end_pix[2]))
             
                
                patch=self.masks[zslice,yslice,xslice]
                if np.count_nonzero(patch)==0: #if no nodule is present in scan
                    start_pix_list.append(start_pix)
                    
            
     
  
                
           
        
        start_pix=np.abs(start_pix_list)
        

        return (start_pix).astype(np.int)
    
    @action
    def get_middle_slices(self):
         CenterSliceIm=np.zeros([len(self),self.images.shape[1], self.images.shape[2]])   
         CenterSliceMask=np.zeros([len(self),self.images.shape[1], self.images.shape[2]])   



         for i in range(0, len(self)):
             im=self.get(i,'images')
             mask=self.get(i,'segmentation')
    
             CenterSliceZ=int(len(im)/2)
             CenterIm=im[CenterSliceZ,:,:]
             CenterMask=mask[CenterSliceZ,:,:]
    
             CenterSliceIm[i, :,:]=CenterIm
             CenterSliceMask[i,:,:]=CenterMask
        
         return CenterSliceIm, CenterSliceMask

    @action
    def get_lung_mask(self,rad=10):
        mask, segmentation=lung.total_segmentation(self, rad)
        self.masks=mask #dilated masks
        self.segmentation= segmentation #segmentation
        return self
    
    @action
    def apply_lung_mask(self, padding=170):  
        self.images = self.images * self.masks + padding * (1-self.masks)
        self.images[(self.masks-self.segmentation)*self.images >= 210] = 170
      #  self.segmentation=self.masks
     #   self.masks=None
        return self
      

  
    
    
    def _init_rebuildwithmask(self, **kwargs):
      """ Fetch args for `images` rebuilding using inbatch_parallel.


      Args-fetcher for parallelization using inbatch_parallel

      Parameters
      ----------
      **kwargs
              shape : tuple, list or ndarray of int
                  (z,y,x)-shape of every image in image component after action is performed.
              spacing : tuple, list or ndarray of float
                  (z,y,x)-spacing for each image. If supplied, assume that
                  unify_spacing is performed.

      Returns
      -------
      list
          list of arg-dicts for different workers
      """
      if 'shape' in kwargs:
          num_slices, y, x = kwargs['shape']
          new_bounds = num_slices * np.arange(len(self) + 1)
          new_data = np.zeros((num_slices * len(self), y, x))
          new_mask = np.zeros((num_slices * len(self), y, x))
          new_segm=np.zeros((num_slices * len(self), y, x))
      else:
          new_bounds = self._bounds
          new_data = np.zeros_like(self.images)
          new_mask = np.zeros_like(self.masks)
          new_segm=np.zeros_like(self.segmentation)
          
      all_args = []
      for i in range(len(self)):
          out_patient = new_data[new_bounds[i]: new_bounds[i + 1], :, :]
          out_mask = new_mask[new_bounds[i]: new_bounds[i + 1], :, :]
          out_segm= new_segm[new_bounds[i]: new_bounds[i + 1], :, :]
          
          item_args = {'patient': self.get(i, 'images'),
                       'out_patient': out_patient,
                       'res': new_data,
                       'patient_mask': self.get(i, 'masks'),
                       'out_mask': out_mask,
                       'res2': new_mask,
                       'patient_segm': self.get(i, 'segmentation'),
                       'out_segm': out_segm,
                       'res3': new_segm,
                       
                       
                       }
      
          

          # for unify_spacing
          if 'spacing' in kwargs:
              shape_after_resize = (self.images_shape * self.spacing
                                    / np.asarray(kwargs['spacing']))
              shape_after_resize = np.rint(shape_after_resize).astype(np.int)
              item_args['factor'] = self.spacing[i, :] / np.array(kwargs['spacing'])
              item_args['shape_resize'] = shape_after_resize[i, :]
              

          all_args += [item_args]

      return all_args
 
  
    def _post_rebuildwithmask(self, all_outputs, new_batch=False, **kwargs):
      """ Gather outputs of different workers for actions, rebuild `images` component.

      Parameters
      ----------
      all_outputs : list
          list of outputs. Each item is given by tuple
      new_batch : bool
          if True, returns new batch with data agregated
          from all_ouputs. if False, changes self.
      **kwargs
              shape : list, tuple or ndarray of int
                  (z,y,x)-shape of every image in image component after action is performed.
              spacing : tuple, list or ndarray of float
                  (z,y,x)-spacing for each image. If supplied, assume that
                  unify_spacing is performed.
      """
   
      self._reraise_worker_exceptions(all_outputs)
     
      new_data, a, new_masks, b , new_segm,c = all_outputs[0]
    
      new_bounds = np.cumsum([patient_shape[0] for _, patient_shape, _, _,_,_
                              in [[0, (0, ), 0, (0, ), 0 , (0,)]] + all_outputs])
      # each worker returns the same ref to the whole res array
     


      # recalculate new_attrs of a batch

      # for resize/unify_spacing: if shape is supplied, assume post
      # is for resize or unify_spacing
      if 'shape' in kwargs:
          new_spacing = self.rescale(kwargs['shape'])
      else:
          new_spacing = self.spacing

      # for unify_spacing: if spacing is supplied, assume post
      # is for unify_spacing
      if 'spacing' in kwargs:
          # recalculate origin, spacing
          shape_after_resize = np.rint(self.images_shape * self.spacing
                                       / np.asarray(kwargs['spacing']))
          overshoot = shape_after_resize - np.asarray(kwargs['shape'])
          new_spacing = self.rescale(new_shape=shape_after_resize)
          new_origin = self.origin + new_spacing * (overshoot // 2)
      else:
          new_origin = self.origin

      # build/update batch with new data and attrs
     #params = dict(images=new_data, bounds=new_bounds,
      #              origin=new_origin, spacing=new_spacing)
      
      params = dict(images=new_data, bounds=new_bounds,
                   origin=new_origin, spacing=new_spacing)
      if new_batch:
          batch_res = type(self)(self.index)
          batch_res.load(fmt='ndarray', **params)
          return batch_res
      else:
          self._init_data(**params)
          self.masks=new_masks
          self.segmentation=new_segm
          return self

    def _post_rebuildwithmask_noresize(self, all_outputs, new_batch=False, **kwargs):
      """ Gather outputs of different workers for actions, rebuild `images` component.

      Parameters
      ----------
      all_outputs : list
          list of outputs. Each item is given by tuple
      new_batch : bool
          if True, returns new batch with data agregated
          from all_ouputs. if False, changes self.
      **kwargs
              shape : list, tuple or ndarray of int
                  (z,y,x)-shape of every image in image component after action is performed.
              spacing : tuple, list or ndarray of float
                  (z,y,x)-spacing for each image. If supplied, assume that
                  unify_spacing is performed.
      """
   
      self._reraise_worker_exceptions(all_outputs)
     
      new_data, a, new_masks, b , new_segm, c= all_outputs[0]
    
      new_bounds = np.cumsum([patient_shape[0] for _, patient_shape, _, _
                              in [[0, (0, ), 0, (0, ), 0, (0,)]] + all_outputs])
      # each worker returns the same ref to the whole res array
     


      # recalculate new_attrs of a batch

      # for resize/unify_spacing: if shape is supplied, assume post
      # is for resize or unify_spacing
      if 'shape' in kwargs:
          new_spacing = self.rescale(kwargs['shape'])
      else:
          new_spacing = self.spacing

      # for unify_spacing: if spacing is supplied, assume post
      # is for unify_spacing
      if 'spacing' in kwargs:
          # recalculate origin, spacing
          shape_after_resize = np.rint(self.images_shape * self.spacing
                                       / np.asarray(kwargs['spacing']))
          
          overshoot = shape_after_resize - np.asarray(kwargs['shape'])
          new_spacing = self.rescale(new_shape=shape_after_resize)
          new_origin = self.origin + new_spacing * (overshoot // 2)
      else:
          new_origin = self.origin

      # build/update batch with new data and attrs
     #params = dict(images=new_data, bounds=new_bounds,
      #              origin=new_origin, spacing=new_spacing)
      
      params = dict(images=new_data, bounds=new_bounds,
                   origin=new_origin, spacing=new_spacing)
      if new_batch:
          batch_res = type(self)(self.index)
          batch_res.load(fmt='ndarray', **params)
          return batch_res
      else:
          self._init_data(**params)
          self.masks=new_masks
          self.segmentation=new_segm
          return self

    @action
    @inbatch_parallel(init='_init_rebuildwithmask', post='_post_rebuildwithmask', target='threads')
    def unify_spacing_withmask_noresize(self, patient, out_patient, res, patient2, out_patient2, res2, factor,
                    shape_resize, spacing=(1, 1, 1), shape=(128, 256, 256),
                    method='pil-simd', order=3, padding='edge', axes_pairs=None,
                    resample=None, *args, **kwargs):
      """ Unify spacing of all patients.

      Resize all patients to meet `spacing`, then crop/pad resized array to meet `shape`.

      Parameters
      ----------
      spacing : tuple, list or ndarray of float
          (z,y,x)-spacing after resize.
          Should be passed as key-argument.
      shape : tuple, list or ndarray of int
          (z,y,x)-shape after crop/pad.
          Should be passed as key-argument.
      method : str
          interpolation method ('pil-simd' or 'resize').
          Should be passed as key-argument.
          See CTImagesBatch.resize for more information.
      order : None or int
          order of scipy-interpolation (<=5), if used.
          Should be passed as key-argument.
      padding : str
          mode of padding, any supported by np.pad.
          Should be passed as key-argument.
      axes_pairs : tuple, list of tuples with pairs
          pairs of axes that will be used consequentially
          for performing pil-simd resize.
          Should be passed as key-argument.
      resample : None or str
          filter of pil-simd resize.
          Should be passed as key-argument
      patient : str
          index of patient, that worker is handling.
          Note: this argument is passed by inbatch_parallel
      out_patient : ndarray
          result of individual worker after action.
          Note: this argument is passed by inbatch_parallel
      res : ndarray
          New `images` to replace data inside `images` component.
          Note: this argument is passed by inbatch_parallel
      factor : tuple
          (float), factor to make resize by.
          Note: this argument is passed by inbatch_parallel
      shape_resize : tuple
          It is possible to provide `shape_resize` argument (shape after resize)
          instead of spacing. Then array with `shape_resize`
          will be cropped/padded for shape to = `shape` arg.
          Note that this argument is passed by inbatch_parallel

      Notes
      -----
      see CTImagesBatch.resize for more info about methods' params.

      Examples
      --------
      >>> shape = (128, 256, 256)
      >>> batch = batch.unify_spacing(shape=shape, spacing=(1.0, 1.0, 1.0),
                                      order=2, method='scipy', padding='reflect')
      >>> batch = batch.unify_spacing(shape=shape, spacing=(1.0, 1.0, 1.0),
                                      resample=PIL.Image.BILINEAR)
      """
     
      if method == 'scipy':
       
          args_resize = dict(patient=patient, out_patient=out_patient,
                             res=res, order=order, factor=factor, padding=padding)
          
          array1,array2= resize_scipy(**args_resize) #images, size
          args_resize2= dict(patient=patient2, out_patient=out_patient2,
                             res=res2, order=order, factor=factor, padding=padding)
          array3,array4= resize_scipy(**args_resize2) #masks, size
          
        
          return array1, array2, array3, array4 
      elif method == 'pil-simd':
          args_resize = dict(input_array=patient, output_array=out_patient,
                             res=res, axes_pairs=axes_pairs, resample=resample,
                             shape_resize=shape_resize, padding=padding)
          
          array1,array2 =  resize_pil(**args_resize) #images, size
          
          args_resize2 = dict(input_array=patient2, output_array=out_patient2,
                             res=res2, axes_pairs=axes_pairs, resample=PIL.Image.NEAREST,
                             shape_resize=shape_resize, padding=padding)
          array3, array4 = resize_pil(**args_resize2) #masks, size
          return array1, array2, array3, array4 
  

                       
    @action
    @inbatch_parallel(init='_init_rebuildwithmask', post='_post_rebuildwithmask', target='threads')
    def unify_spacing_withmask(self, patient, out_patient, res, patient_mask, out_mask, res2,patient_segm,out_segm,
                               res3, factor,
                    shape_resize, spacing=(1, 1, 1), shape=(128, 256, 256),
                    method='pil-simd', order=3, padding='edge', axes_pairs=None,
                    resample=None, *args, **kwargs):
      """ Unify spacing of all patients.

      Resize all patients to meet `spacing`, then crop/pad resized array to meet `shape`.

      Parameters
      ----------
      spacing : tuple, list or ndarray of float
          (z,y,x)-spacing after resize.
          Should be passed as key-argument.
      shape : tuple, list or ndarray of int
          (z,y,x)-shape after crop/pad.
          Should be passed as key-argument.
      method : str
          interpolation method ('pil-simd' or 'resize').
          Should be passed as key-argument.
          See CTImagesBatch.resize for more information.
      order : None or int
          order of scipy-interpolation (<=5), if used.
          Should be passed as key-argument.
      padding : str
          mode of padding, any supported by np.pad.
          Should be passed as key-argument.
      axes_pairs : tuple, list of tuples with pairs
          pairs of axes that will be used consequentially
          for performing pil-simd resize.
          Should be passed as key-argument.
      resample : None or str
          filter of pil-simd resize.
          Should be passed as key-argument
      patient : str
          index of patient, that worker is handling.
          Note: this argument is passed by inbatch_parallel
      out_patient : ndarray
          result of individual worker after action.
          Note: this argument is passed by inbatch_parallel
      res : ndarray
          New `images` to replace data inside `images` component.
          Note: this argument is passed by inbatch_parallel
      factor : tuple
          (float), factor to make resize by.
          Note: this argument is passed by inbatch_parallel
      shape_resize : tuple
          It is possible to provide `shape_resize` argument (shape after resize)
          instead of spacing. Then array with `shape_resize`
          will be cropped/padded for shape to = `shape` arg.
          Note that this argument is passed by inbatch_parallel

      Notes
      -----
      see CTImagesBatch.resize for more info about methods' params.

      Examples
      --------
      >>> shape = (128, 256, 256)
      >>> batch = batch.unify_spacing(shape=shape, spacing=(1.0, 1.0, 1.0),
                                      order=2, method='scipy', padding='reflect')
      >>> batch = batch.unify_spacing(shape=shape, spacing=(1.0, 1.0, 1.0),
                                      resample=PIL.Image.BILINEAR)
      """
      if method == 'scipy':
       
          args_resize = dict(patient=patient, out_patient=out_patient,
                             res=res, order=order, factor=factor, padding=padding)
          array1,array2= resize_scipy(**args_resize) #images, size
          
          args_resize2= dict(patient=patient_mask, out_patient=out_mask,
                             res=res2, order=order, factor=factor, padding=padding)
          array3,array4= resize_scipy(**args_resize2) #masks, size
          
          args_resize3= dict(patient=patient_segm, out_patient=out_segm,
                             res=res3, order=order, factor=factor, padding=padding)
          array5,array6= resize_scipy(**args_resize3) #masks, size
        
          return array1, array2, array3, array4 , array5, array6
      elif method == 'pil-simd':
          args_resize = dict(input_array=patient, output_array=out_patient,
                             res=res, axes_pairs=axes_pairs, resample=resample,
                             shape_resize=shape_resize, padding=padding)
          
          array1,array2 =  resize_pil(**args_resize) #images, size
          
          args_resize2 = dict(input_array=patient_mask, output_array=out_mask,
                             res=res2, axes_pairs=axes_pairs, resample=PIL.Image.NEAREST,
                             shape_resize=shape_resize, padding=padding)
          array3, array4 = resize_pil(**args_resize2) #masks, size
          
          
          args_resize2 = dict(input_array=patient_segm, output_array=out_segm,
                             res=res3, axes_pairs=axes_pairs, resample=PIL.Image.NEAREST,
                             shape_resize=shape_resize, padding=padding)
          array5, array6 = resize_pil(**args_resize2) #masks, size
          
          
          
          
          return array1, array2, array3, array4 , array5, array6
       
    @action    
    def predict_on_scans(self, model_name, strides=(16, 32, 32), crop_shape=(32, 64, 64),
                  batch_size=4, targets_mode='segmentation', data_format='channels_last',
                  show_progress=True, model_type='tf'):
      """ Get predictions of the model on data contained in batch.

      Transforms scan data into patches of shape CROP_SHAPE and then feed
      this patches sequentially into model with name specified by
      argument 'model_name'; after that loads predicted masks or probabilities
      into 'masks' component of the current batch and returns it.

      Parameters
      ----------
      model_name : str
          name of model that will be used for predictions.
      strides : tuple, list or ndarray of int
          (z,y,x)-strides for patching operation.
      crop_shape : tuple, list or ndarray of int
          (z,y,x)-shape of crops.
      batch_size : int
          number of patches to feed in model in one iteration.
      targets_mode: str
          type of targets 'segmentation', 'regression' or 'classification'.
      data_format: str
          format of neural network input data,
          can be 'channels_first' or 'channels_last'.
      model_type : str
          represents type of model that will be used for prediction.
          Possible values are 'keras' or 'tf'.

      Returns
      -------
      CTImagesMaskedBatch.
      """
      
      
      _model = model_name
      crop_shape = np.asarray(crop_shape).reshape(-1)
      strides = np.asarray(strides).reshape(-1)

      patches_arr = self.get_patches(patch_shape=crop_shape,
                                     stride=strides,
                                     padding='reflect')
      if data_format == 'channels_first':
          patches_arr = patches_arr[:, np.newaxis, ...]
      elif data_format == 'channels_last':
          patches_arr = patches_arr[..., np.newaxis]

      predictions = []
      iterations = range(0, patches_arr.shape[0], batch_size)
      if show_progress:
          iterations = tqdm_notebook(iterations)  # pylint: disable=redefined-variable-type
      for i in iterations:

          if model_type == 'tf':
              _prediction = _model.predict(feed_dict={'images': patches_arr[i: i + batch_size, ...]})
          else:
              _prediction = _model.predict(patches_arr[i: i + batch_size, ...])

          current_prediction = np.asarray(_prediction)
          if targets_mode == 'classification':
              current_prediction = np.stack([np.ones(shape=(crop_shape)) * prob
                                             for prob in current_prediction.ravel()])

        #  if targets_mode == 'regression':
            #  current_prediction = create_mask_reg(current_prediction[:, :3],
#                                                     current_prediction[:, 3:6],
#                                                     current_prediction[:, 6],
#                                                     crop_shape, 0.01)

          predictions.append(current_prediction)

      patches_mask = np.concatenate(predictions, axis=0)
      patches_mask = np.squeeze(patches_mask)
      self.load_from_patches(patches_mask, stride=strides,
                             scan_shape=tuple(self.images_shape[0, :]),
                             data_attr='masks')
      return self, patches_mask
    
    @action
    def loadMalignancy(self):
        num_nodules=len(self)
        self.nodules = np.rec.array(np.zeros(num_nodules,
                                                 dtype=self.nodules_dtype))
        for i in range(len(self)):
            print(i)
            path=self.index.get_fullpath(self.indices[i])
            print(path)
            nodules=np.load(path+'/nodules.npy')
            print(len(nodules))
            if len(nodules)>1:
                print('more than one nodule present')
                
            firstNodule=nodules[0]
            firstNodule[0]=i
            self.nodules[i]=firstNodule

        return self
   
    @action
    def dumpMalignancy(self, dst):
        for index in range(len(self.indices)):
            np.save(dst +'/'+self.indices[index]+'/nodules.npy', self[index].nodules)
            
        return self     
            