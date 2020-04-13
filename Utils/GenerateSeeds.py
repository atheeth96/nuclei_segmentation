import skimage
from skimage.io import imread,imsave
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage import img_as_uint
from tqdm import tqdm
import os 
import numpy as np
import math

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random


def multiple_erosion(img,iter=5):
    for j in (range(5)):
        img=morphology.binary_erosion(img, selem=morphology.selem.disk(1))
    return img

def multiple_dialte(img,iter=5):
    for j in (range(5)):
        img=morphology.binary_dilation(img, selem=morphology.selem.disk(1))
    return img



def generate_nuclei_seeds(IMAGE_PATH,img_list,pred_dir_name,model,print_prompt=True):
    
    pred_dir=os.path.join(os.getcwd(),pred_dir_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    model.eval()
    
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
        if print_prompt:
            print("Made {} directory".format(pred_dir.split('/')[-1]))
    else:
        if print_prompt:
            print("{} directory already exists in {}".format(pred_dir.split('/')[-1],'/'.join(pred_dir.split('/')[:-1])))
        
    step=512
    avg_dice=0
    loop=tqdm(img_list)
    for count,img_name in enumerate(loop):
        
        
        
        input_image=imread(os.path.join(IMAGE_PATH,img_name))
        
    
        r,c=input_image.shape#4663,3881

        new_r_count=(math.ceil((r-512)/512)+1)#5
        new_c_count=(math.ceil((c-512)/512)+1)#5


        pad_r1=((new_r_count-1)*512-r+512)//2 #200
        pad_r2=((new_r_count-1)*512-r+512)-pad_r1 #200
        pad_c1=((new_c_count-1)*512-c+512)//2 #0
        pad_c2=((new_c_count-1)*512-c+512)-pad_c1#0

        image_padded=np.pad(input_image, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant', constant_values=0)/np.amax(input_image)


        window_shape=(512,512)

        img_patches=skimage.util.view_as_windows(image_padded, window_shape, step=step)
        img_patches=img_patches.reshape((-1,512,512))
        img_patches=img_patches.transpose((0,2,1))
        img_patches=np.expand_dims(img_patches,axis=1)
        
    
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
        
        thresh_nuclei=threshold_otsu(nuclei_temp)
        nuclei_temp=nuclei_temp>thresh_nuclei
        
        thresh_bound=threshold_otsu(bound_temp)
        bound_temp=bound_temp>thresh_bound
        
        bound_temp=multiple_dialte(bound_temp)
        bound_temp=multiple_erosion(bound_temp)
        
        nuclei_temp=multiple_erosion(nuclei_temp)
        nuclei_temp=multiple_dialte(nuclei_temp)
        
        comb_img=nuclei_temp^bound_temp
        bound_coor=np.where(bound_temp==1)
        comb_img[bound_coor]=0
        comb_img=multiple_erosion(comb_img,3)
        comb_img=multiple_dialte(comb_img,3)

        imsave(pred_dir_name+'/'+img_name,img_as_uint(comb_img))

        
  
    print("DONE")




