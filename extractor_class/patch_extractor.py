

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
from skimage.io import imread
from skimage.transform import resize
from skimage.filters import threshold_otsu

import time
import math

import random
from tqdm import tqdm_notebook as tqdm 

class PatchExtractor():
    def __init__(self,patient_path_list,img_patch_dir\
    ,nuclei_mask_patch_dir,bound_mask_patch_dir,roi_list=None,overlap=False,progress_bar=True):
        
        self.patient_path_list=patient_path_list
        
        self.img_patch_dir=img_patch_dir
        self.nuclei_mask_patch_dir=nuclei_mask_patch_dir
        self.bound_mask_patch_dir=bound_mask_patch_dir
        self.roi_list=roi_list
        self.overlap=overlap
        self.progress_bar=progress_bar
        

    def extract_patches(self):
        if self.overlap:
            step=256
        else:
            step=512
        NO_PATCHES=0
        
#       for tag in ['train','test']:
        img_dir_temp=self.img_patch_dir
        nuclei_mask_patch_dir_temp=self.nuclei_mask_patch_dir
        bound_mask_patch_dir_temp=self.bound_mask_patch_dir

        if not os.path.exists(img_dir_temp):
            os.mkdir(img_dir_temp)
            if self.progress_bar:
            
                print("image patch directory made")
        if not os.path.exists(nuclei_mask_patch_dir_temp):
            os.mkdir(nuclei_mask_patch_dir_temp)
            if self.progress_bar:
                print("mask patch directory made")
        if not os.path.exists(bound_mask_patch_dir_temp):
            os.mkdir(bound_mask_patch_dir_temp)
            if self.progress_bar:
                print("mask patch directory made")
        exception_list=[]
        
       

        for patient in self.patient_path_list:
            if self.roi_list:
                image_list=self.roi_list
            else:
                image_list=os.listdir(os.path.join(patient,'ROI'))
            image_list.sort()
            if self.progress_bar:
                loop=tqdm(image_list)
            else:
                loop=image_list
                
            
            for image in loop:
                try:
                
                    image_path=os.path.join(patient+'/ROI',image)
                    loop.set_description('Slide ID : {}, ROI : {}'.format(patient.split('/')[-1], image.split('.')[0]))
                    #print(image,self.nuclei_mask_dir)

                    nuclei_mask=[x for x in os.listdir(patient+'/nuc_mask') if image.split('.')[0] in x][-1]
                    nuclei_mask_path=os.path.join(patient+'/nuc_mask',nuclei_mask)

                    bound_mask=[x for x in os.listdir(patient+'/bound_mask') if image.split('.')[0] in x][-1]
                    bound_mask_path=os.path.join(patient+'/bound_mask',bound_mask)


                    input_image=np.array(Image.open(image_path))

                    input_nuclei_mask_16bit=np.array(Image.open(nuclei_mask_path))
                    input_nuclei_mask=np.zeros(input_nuclei_mask_16bit.shape,dtype=np.uint8)
                    input_nuclei_mask[np.where(input_nuclei_mask_16bit!=0)]=255

                    input_bound_mask=np.array(Image.open(bound_mask_path))

                    r,c=input_image.shape#4663,3881

                    new_r_count=(math.ceil((r-512)/512)+1)#10
                    new_c_count=(math.ceil((c-512)/512)+1)#8


                    pad_r1=((new_r_count-1)*512-r+512)//2 #228
                    pad_r2=((new_r_count-1)*512-r+512)-pad_r1 #229
                    pad_c1=((new_c_count-1)*512-c+512)//2 #107
                    pad_c2=((new_c_count-1)*512-c+512)-pad_c1#108

                    image_padded=np.pad(input_image, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant', constant_values=0)#/np.amax(input_image)
                    nuclei_mask_padded=np.pad(input_nuclei_mask, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant', constant_values=0)
                    bound_mask_padded=np.pad(input_bound_mask, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant', constant_values=0)

                    window_shape=(512,512)

                    img_patches=skimage.util.view_as_windows(image_padded, window_shape, step=step)
                    img_patches=img_patches.reshape((-1,512,512))

                    nuclei_mask_patches=skimage.util.view_as_windows(nuclei_mask_padded, window_shape, step=step)
                    nuclei_mask_patches=nuclei_mask_patches.reshape((-1,512,512))

                    bound_mask_patches=skimage.util.view_as_windows(bound_mask_padded, window_shape, step=step)
                    bound_mask_patches=bound_mask_patches.reshape((-1,512,512))

                    for i,(img_patch,nuclei_mask_patch,bound_mask_patch) in enumerate(zip(img_patches,nuclei_mask_patches,bound_mask_patches)):
                        NO_PATCHES+=1

                        image_patch=Image.fromarray((img_patch).astype(np.uint16))
                        nuclei_mask_patch=Image.fromarray((nuclei_mask_patch).astype(np.uint8))
                        bound_mask_patch=Image.fromarray((bound_mask_patch).astype(np.uint8))
                        image_patch.save(os.path.join(img_dir_temp,image.split('.')[0]+'_{}_{}.tif'.format(patient.split('/')[-1],i)))
                        nuclei_mask_patch.save(os.path.join(nuclei_mask_patch_dir_temp,image.split('.')[0]+'_{}_{}.tif'.format(patient.split('/')[-1],i)))
                        bound_mask_patch.save(os.path.join(bound_mask_patch_dir_temp,image.split('.')[0]+'_{}_{}.tif'.format(patient.split('/')[-1],i)))
                except:
                    exception_list.append(patient.split('/')[-1]+'/'+ image.split('.')[0])
        if self.progress_bar:
            print("NUMBER OF PATCHES ARE {}".format(NO_PATCHES),'\nException\n',*exception_list,sep='\n')