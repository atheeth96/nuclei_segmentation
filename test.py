from Utils.Prediction import post_process,whole_img_pred
from tqdm import tqdm
import torch

from Models.ModelsTorch import AttnUNet,load_model
import torch.nn.functional as F
import os 
import pandas  as pd

import shutil
import random


patient_list=[x for x in os.listdir('Dapi_patient_data/') if 'S-' in x and '.zip' not in x]
print("Available slides :-\n ",*patient_list,sep='\n')
patient_name=input("ENTER PATIENT SLIDE ID : ")
assert patient_name in patient_list,"Please enter valid ID  out of the list {}".format(patient_list)
IMAGE_PATH='/datalab/training-assets/R_medical/atheeth/DAPI/nucleiseg_dapi/Dapi_patient_data/{}'.format(patient_name)
img_list=os.listdir(IMAGE_PATH+'/ROI')
QC_PATH='Dapi_patient_data/{}/OverallQCMasks'.format(patient_name)

pred_dir_name=os.path.join(IMAGE_PATH,'predictions')
processed_dir=os.path.join(IMAGE_PATH,'processed')
if not os.path.exists(pred_dir_name):
    os.mkdir(pred_dir_name)
if not os.path.exists(processed_dir):
    os.mkdir(processed_dir)
    
    

model=AttnUNet(img_ch=1,output_ch=2)
load_model('model_2019_11_13/model_optim.pth',model)


pred_dir=os.path.join(os.getcwd(),pred_dir_name)

if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
    print("Made {} directory".format(pred_dir.split('/')[-1]))
else:
    print("{} directory already exists in {}".format(pred_dir.split('/')[-1],'/'.join(pred_dir.split('/')[:-1])))

loop=tqdm(img_list)
for count,img_name in enumerate(loop):
    gt_narray_name=\
    '/datalab/training-assets/R_medical/atheeth/DAPI/nucleiseg_dapi/Dapi_patient_data/{}/NucSeg/NucMap_{}_{}.tif'.\
    format(patient_name,patient_name,img_name.split('.')[0])
    gt_narray_image=imread(gt_narray_name)
    gt_narray_image=gt_narray_image.astype(np.uint16)
    
    nuclei_temp,bound_temp=whole_img_pred(IMAGE_PATH,img_name,model)
    
    imsave(pred_dir_name+'/bound_'+img_name.split('.')[0]+'.png',bound_temp)
    imsave(pred_dir_name+'/nuclei_'+img_name.split('.')[0]+'.png',nuclei_temp)
    
    qc_file_path=os.path.join(QC_PATH,'OverallQCMask_{}_{}.tif'.format(patient_name,img_name.split('.')[0]))
    
    comb_img,metrics=post_process(nuclei,boundary,gt_narray_image,qc_file_path)
    imsave(processed_dir+'/'+img_name.split('.')[0]+'.tif',comb_img)
    
