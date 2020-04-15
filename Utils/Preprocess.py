
import datetime
from zipfile import ZipFile
import math
import time
from skimage import morphology

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/vahadaneabhi01/datalab/training-assets/R_medical/atheeth/DAPI/nucleiseg_dapi')
from extractor_class.patch_extractor import PatchExtractor
from skimage.filters import threshold_otsu

import skimage
from skimage.io import imread,imsave
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import os, shutil


def generate_train_mask(patient_id,path2patient):
    path=os.path.join(path2patient,patient_id)+'/nuc_mask'
    if not os.path.exists(path):
        os.mkdir(path)
    path_bound=os.path.join(path2patient,patient_id)+'/bound_mask'
    if not os.path.exists(path_bound):
        os.mkdir(path_bound)
    
    
    n_array_list=[x for x in os.listdir(os.path.join(path2patient,patient_id)+'/NucSeg') if '.tif' in x]
    #print(n_array_list)
    for n_array in tqdm(n_array_list):
        img=imread(path2patient+'/'+patient_id+'/NucSeg/'+n_array)
       
        final_img=np.zeros(img.shape,dtype=np.uint8)
        for i,nuclei in enumerate(range(1,np.amax(img))):
            img_cancer=np.zeros(img.shape,dtype=np.uint8)
            img_cancer[np.where(img==nuclei)]=255

            contours=skimage.measure.find_contours(img_cancer,0)[0]

            img_temp=np.zeros(img.shape,dtype=np.uint8)
            for countour in contours:
                x,y=countour
                
                img_temp[int(x),int(y)]=255
            
            final_img+=img_temp
                
        final_img=morphology.dilation(final_img, selem=None)
        final_img=final_img.astype(np.uint8)
        imsave(path_bound+'/'+n_array.split('_')[-1],final_img)
        
        
                
                
            
        img[np.where(img!=0)]=255
        img=img.astype(np.uint8)
        imsave(path+'/'+n_array.split('_')[-1],img)
        
        
        
def fuse_roi(patient_id,path2patient):
    rois=os.listdir(path2patient+'/'+patient_id+'/DAPI')
    rois=[x for x in rois if '.DS_Store' not in x]
    #print(rois)
    main_dir=os.path.join(path2patient,patient_id)
    path=os.path.join(main_dir,'ROI')
    if not os.path.exists(path):
        os.mkdir(path)
    for roi in tqdm(rois):
        images_list=[x for x in os.listdir(main_dir+'/DAPI/'+roi) if '.tif' in x]
#         print(images_list)
        
        img_final=imread(os.path.join(os.path.join(os.path.join(main_dir,'DAPI'),roi),images_list[0]))
        

        for img_name in images_list[1:]:
            img_temp=imread(main_dir+'/DAPI/'+roi+'/'+img_name)
            try:
                
                img_final=np.maximum(img_temp,img_final)
            except:
                print(main_dir+'/DAPI/'+roi+'/'+img_name,'\n',images_list[0])
                
                print(img_temp.shape)
                

        imsave(path+'/'+roi+'.tif',img_final)