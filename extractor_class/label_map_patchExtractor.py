

import numpy as np
import glob
import os
import sys
import scipy

import matplotlib.pyplot as plt
#import cv2
import imutils
import re


import zipfile

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.feature_extraction import image


import skimage
from skimage.util import pad
from skimage import draw
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.filters import threshold_otsu

import time
import math

import random
from tqdm import tqdm_notebook as tqdm 

class LabelPatchExtractor():
    def __init__(self,patient_path_list,biomarker,img_patch_dir\
    ,mask_patch_dir,save_type='tif',overlap=False,progress_bar=True):
        assert biomarker in ['KI67','FOXP3'],'Biomarker not found'
        
        self.patient_path_list=patient_path_list
        
        self.img_patch_dir=img_patch_dir
        self.mask_patch_dir=mask_patch_dir
        self.biomarker_dir_list=[os.path.join(x,'Biomarkers/{}'.format(biomarker)) for x in patient_path_list]
        self.label_map_dir_list=[os.path.join(x,'LabelMaps/{}'.format(biomarker)) for x in patient_path_list]
            
        
        self.overlap=overlap
        self.progress_bar=progress_bar
        

    def extract_patches(self):
        if self.overlap:
            step=256
        else:
            step=512
        NO_PATCHES=0
        
#       for tag in ['train','test']:
        img_patch_dir=self.img_patch_dir
        mask_patch_dir=self.mask_patch_dir
        

        if not os.path.exists(img_patch_dir):
            os.mkdir(img_patch_dir)
            if self.progress_bar:
                print("image patch directory made")
        if not os.path.exists(mask_patch_dir):
            os.mkdir(mask_patch_dir)
            if self.progress_bar:
                print("mask patch directory made")
      
        exception_list=[]
        
       

        for patient in self.patient_path_list:
            biomarker_dir=[x for x in self.biomarker_dir_list if patient in x][-1]
            label_map_dir=[x for x in self.label_map_dir_list if patient in x][-1]
            image_list=os.listdir(biomarker_dir)
            image_list.sort()
            if self.progress_bar:
                loop=tqdm(image_list)
            else:
                loop=image_list
                
            
            for image in loop:
                try:
                
                    biomarker_path=os.path.join(biomarker_dir,image)
                    loop.set_description('Slide ID : {}, ROI : {}'.format(image.split('_')[1], image.split('_')[-2]))
                    #print(image,self.nuclei_mask_dir)

                    mask=[x for x in os.listdir(label_map_dir) if (image.split('_')[1] in x and image.split('_')[-2] in x)][-1]
                    mask_path=os.path.join(label_map_dir,mask)



                    biomarker_image=imread(biomarker_path)
                    mask_image_16_bit=imread(mask_path)

                    mask_image=np.zeros(mask_image_16_bit.shape,dtype=np.uint8)
                    mask_image[np.where(mask_image_16_bit!=0)]=255
                    del mask_image_16_bit



                    r,c=biomarker_image.shape#4663,3881

                    new_r_count=(math.ceil((r-512)/512)+1)#10
                    new_c_count=(math.ceil((c-512)/512)+1)#8


                    pad_r1=((new_r_count-1)*512-r+512)//2 #228
                    pad_r2=((new_r_count-1)*512-r+512)-pad_r1 #229
                    pad_c1=((new_c_count-1)*512-c+512)//2 #107
                    pad_c2=((new_c_count-1)*512-c+512)-pad_c1#108

                    biomarker_padded=np.pad(biomarker_image, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant', constant_values=0)#/np.amax(input_image)
                    mask_padded=np.pad(mask_image, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant', constant_values=0)

                    window_shape=(512,512)

                    biomarker_patches=skimage.util.view_as_windows(biomarker_padded, window_shape, step=step)
                    biomarker_patches=biomarker_patches.reshape((-1,512,512))

                    mask_patches=skimage.util.view_as_windows(mask_padded, window_shape, step=step)
                    mask_patches=mask_patches.reshape((-1,512,512))



                    for i,(biomarker_patch,mask_patch) in enumerate(zip(biomarker_patches,mask_patches)):
                        NO_PATCHES+=1



                        imsave(os.path.join(img_patch_dir,'Biomarker_'+image.split('_')[1]+'_'\
                                            +image.split('_')[-2]+'_{}_.tif'.format(i+1)),biomarker_patch)
                        imsave(os.path.join(mask_patch_dir,'LabelMap_'+image.split('_')[1]+'_'\
                                            +image.split('_')[-2]+'_{}_.tif'.format(i+1)),mask_patch)

                except:
                    exception_list.append(patient.split('/')[-1]+'/'+ image.split('.')[0])
        if self.progress_bar:
            print("NUMBER OF PATCHES ARE {}".format(NO_PATCHES),'\nException\n',*exception_list,sep='\n')