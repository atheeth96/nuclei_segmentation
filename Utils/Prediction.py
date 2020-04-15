import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/vahadaneabhi01/datalab/training-assets/R_medical/atheeth/DAPI/nucleiseg_dapi')
from Losses import whole_dice_metric
import skimage
from skimage.io import imread,imsave
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage import img_as_uint
from tqdm import tqdm
import os 
import numpy as np
import math
from skimage import img_as_ubyte


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
# import matplotlib.pyplot as plt
import cv2


def whole_img_pred(IMAGE_PATH,img_name,model):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    model.eval()
    
        
    step=512
    dice_score=None
  
    input_image=imread(os.path.join(IMAGE_PATH+'/ROI',img_name)).astype(np.float32)
    
 

    r,c=input_image.shape#4663,3881

    new_r_count=(math.ceil((r-512)/512)+1)#5
    new_c_count=(math.ceil((c-512)/512)+1)#5


    pad_r1=((new_r_count-1)*512-r+512)//2 #200
    pad_r2=((new_r_count-1)*512-r+512)-pad_r1 #200
    pad_c1=((new_c_count-1)*512-c+512)//2 #0
    pad_c2=((new_c_count-1)*512-c+512)-pad_c1#0

    image_padded=np.pad(input_image, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant', constant_values=0)


    window_shape=(512,512)

    img_patches=skimage.util.view_as_windows(image_padded, window_shape, step=step)
    img_patches=img_patches.reshape((-1,512,512))
    img_patches=img_patches.transpose((0,2,1))
    img_patches=np.expand_dims(img_patches,axis=1)

    max_patch_level=np.amax(img_patches,axis=(1,2,3)).reshape(-1)
    for i in range(img_patches.shape[0]):
        img_patches[i]=img_patches[i]/max_patch_level[i]


    bound_temp=[]
    nuclei_temp=[]

    for i in range(new_r_count):

        temp_img_patches=torch.from_numpy(img_patches[i*new_r_count:(i+1)*new_r_count]).type(torch.FloatTensor).to(device)
        pred=torch.sigmoid(model(temp_img_patches))
        del temp_img_patches

        nuclei,bound=torch.chunk(pred,2,dim=1)
        del pred
        nuclei,bound=nuclei.detach().cpu().numpy(),bound.detach().cpu().numpy()

        nuclei=np.squeeze(nuclei,axis=1).transpose((0,2,1))
        nuclei=np.concatenate(nuclei,axis=1)
        nuclei_temp.append(nuclei)

        bound=np.squeeze(bound,axis=1).transpose((0,2,1))
        bound=np.concatenate(bound,axis=1)
        bound_temp.append(bound)

    nuclei_temp=np.array(nuclei_temp)
    nuclei_temp=np.concatenate(nuclei_temp,axis=0)

    bound_temp=np.array(bound_temp)
    bound_temp=np.concatenate(bound_temp,axis=0)

    nuclei_temp=nuclei_temp[pad_r1:nuclei_temp.shape[0]-pad_r2,pad_c1:nuclei_temp.shape[1]-pad_c2]*255
    bound_temp=bound_temp[pad_r1:bound_temp.shape[0]-pad_r2,pad_c1:bound_temp.shape[1]-pad_c2]*255

    nuclei_temp=nuclei_temp.astype(np.uint8)
    bound_temp=bound_temp.astype(np.uint8)
    

      
    return nuclei_temp,bound_temp


def multiple_erosion(img,iter=5):
    for j in (range(5)):
        img=morphology.binary_erosion(img, selem=morphology.selem.disk(1))
    return img

def multiple_dialte(img,iter=5):
    for j in (range(5)):
        img=morphology.binary_dilation(img, selem=morphology.selem.disk(1))
    return img

def post_process(PRED_PATH,patient_name,img_name,threshold='otsu',apply_qc=False,print_prompt=False):
    
    path_to_patient_data='Dapi_patient_data/{}'.format(patient_name)
    
#     selected_ip=imread(path_to_patient_data+'/ROI/'+img_name)
    selected_binary_gt=imread(path_to_patient_data+'/nuc_mask/'+img_name)

    gt_narray_name=\
    '/datalab/training-assets/R_medical/atheeth/DAPI/nucleiseg_dapi/Dapi_patient_data/{}/NucSeg/NucMap_{}_{}.tif'.\
    format(patient_name,patient_name,img_name.split('.')[0])
    gt_narray_image=imread(gt_narray_name)
    gt_narray_image=gt_narray_image.astype(np.uint16)
    
    
    bound_img_path=os.path.join(PRED_PATH,'{}_bound_'.format(patient_name)+img_name.split('.')[0]+'.png')
    bound_img=imread(bound_img_path)
    nuc_img_path=os.path.join(PRED_PATH,'{}_nuclei_'.format(patient_name)+img_name.split('.')[0]+'.png')
    nuc_img=imread(nuc_img_path)
    comb_img=img_recon(nuc_img,bound_img)
    comb_img=comb_img.astype(np.uint16)

#     plt.imshow(comb_img);plt.show()
    if apply_qc:
        qc_path='Dapi_patient_data/{}/OverallQCMasks'.format(patient_name)
        qc_file_path=os.path.join(qc_path,'OverallQCMask_{}_{}.tif'.format(patient_name,img_name.split('.')[0]))
        qc_mask=imread(qc_file_path)
        qc_mask[qc_mask!=0]=1
        comb_img=comb_img*qc_mask
        del qc_mask
    


    gt_image=remap_label(gt_narray_image)
    total=len(np.unique(gt_image))
    del gt_narray_image,nuc_img,bound_img
    comb_img=remap_label(comb_img)
    list_1=None

    list_q,_,list_cm=get_fast_pq(gt_image, comb_img)
    gt_image=(gt_image>0).astype(np.uint8)
    comb_img_dice=(comb_img>0).astype(np.uint8)
    dice=whole_dice_metric(comb_img_dice,gt_image)
    del comb_img_dice
    metrics=[list_q,list_cm,dice,total]
                                       
    return comb_img,metrics




def coord2array(coord):
    x=[]
    y=[]
    for i in coord:
        x.append(i[0])
        y.append(i[1])
    return (x,y)


def img_recon(nuclei,boundary):
    
    def gen_inst_dst_map(nuclei):  
        shape = nuclei.shape[:2] # HW
        labeled_img=label(nuclei)
        labeled_img = remove_small_objects(labeled_img, min_size=50)
        regions=regionprops(labeled_img)
    

        canvas = np.zeros(shape, dtype=np.uint8)
        for region in regions:
            coordinates=coord2array(list(region.coords))
            nuc_map=np.zeros(shape)
            nuc_map[coordinates]=1  
            nuc_map=morphology.binary_dilation(nuc_map, selem=morphology.selem.disk(2)).astype(np.uint8)
            nuc_dst = ndi.distance_transform_edt(nuc_map)
            nuc_dst = 255 * (nuc_dst / np.amax(nuc_dst))       
            canvas += nuc_dst.astype('uint8')
        return canvas
    
    nuclei=nuclei>0.45*255#threshold_otsu(nuclei)
    
    
    nuclei=nuclei.astype(np.uint8)
    
    
    nuclei=area_closing(nuclei,20)
    nuclei=closing(nuclei.astype(np.uint8),morphology.selem.square(2))
    nuclei = binary_fill_holes(nuclei)

    nuclei=ndimage.binary_fill_holes(nuclei).astype(int)
    

    
    boundary=boundary>0.3*255#120
    boundary=boundary.astype(np.uint8)
    
    
    nuclei_seeds_ini=nuclei-boundary
    nuclei_seeds_ini[np.where(nuclei_seeds_ini<=0)]=0
    nuclei_seeds_ini[np.where(nuclei_seeds_ini>0)]=1
    
    nuclei_seeds=morphology.binary_erosion(nuclei_seeds_ini, selem=morphology.selem.disk(2)).astype(np.uint8)
    

    labeled_img=label(nuclei_seeds)
    labeled_img = remove_small_objects(labeled_img, min_size=50)
    
    regions=regionprops(labeled_img)

    final_image=np.zeros_like(nuclei_seeds_ini)
    distance = gen_inst_dst_map(nuclei_seeds_ini)
    markers = ndi.label(nuclei_seeds)[0]
    final_image = watershed(-distance, markers, mask=nuclei,watershed_line=False)
    return final_image

