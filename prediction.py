import skimage
from skimage.io import imread,imsave
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage import img_as_uint
from tqdm import tqdm_notebook as tqdm
import os 
import numpy as np
import math
from skimage import img_as_ubyte

from Generators.DatasetTorch import Color

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import matplotlib.pyplot as plt
import cv2


def whole_dice_metric(y_pred,y_true):
    smooth = 10e-16
    # single image so just roll it out into a 1D array
    
    m1 =np.reshape(y_pred,(-1))/255
    m2 =np.reshape(y_true,(-1))/255
    
    
    intersection = (m1 * m2)

    score = 2. * (np.sum(intersection) + smooth) / (np.sum(m1) +(np.sum(m2) + smooth))
        
    return score



def whole_img_pred(IMAGE_PATH,img_list,pred_dir_name,model,print_prompt=False):
    
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
        
        
        
        input_image=imread(os.path.join(IMAGE_PATH+'/ROI',img_name))
        boundary_mask=imread(os.path.join(IMAGE_PATH+'/bound_mask',img_name))
        nuclei_mask=imread(os.path.join(IMAGE_PATH+'/nuc_mask',img_name))
        
        y_true=np.stack((nuclei_mask,boundary_mask),axis=0)
        y_true=np.expand_dims(y_true,axis=0)
    
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
        
        y_pred=np.stack((nuclei_temp,bound_temp),axis=0)
        y_pred=np.expand_dims(y_pred,axis=0)
#         nuclei_thresh=threshold_otsu(nuclei_temp)
#         nucei_thresholded=nuclei_temp>nuclei_thresh
        
        dice_score=whole_dice_metric(nuclei_temp,nuclei_mask)
        avg_dice+=dice_score
        if print_prompt:
            loop.set_postfix(Dice_score =dice_score,average=avg_dice/(count+1))
        
    
        
        
        imsave(pred_dir_name+'/bound_'+img_name.split('.')[0]+'.png',bound_temp)
        
        imsave(pred_dir_name+'/nuclei_'+img_name.split('.')[0]+'.png',nuclei_temp)

        
    if print_prompt:
        print("DONE\nAverage Dice Score = {}".format(avg_dice/(count+1)))



def multiple_erosion(img,iter=5):
    for j in (range(5)):
        img=morphology.binary_erosion(img, selem=morphology.selem.disk(1))
    return img

def multiple_dialte(img,iter=5):
    for j in (range(5)):
        img=morphology.binary_dilation(img, selem=morphology.selem.disk(1))
    return img

def post_process(PRED_PATH,patient_name,img_list,df_summary,df_whole,processed_dir,threshold='otsu',print_prompt=False):
    qc_path='Dapi_patient_data/{}/OverallQCMasks'.format(patient_name)
    
    
    
    path_to_patient_data='Dapi_patient_data/{}'.format(patient_name)
    current_no_entries=len(df_summary)
    
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)
        
    selected_roi=random.choice(img_list)
    selected_ip=imread(path_to_patient_data+'/ROI/'+selected_roi)
    selected_prediction=imread(path_to_patient_data+'/predictions/nuclei_'+selected_roi.split('.')[0]+'.png')
    selected_gt=imread(path_to_patient_data+'/nuc_mask/'+selected_roi)
    avg_error=0
    loop=tqdm(img_list)
    for i,img_name in enumerate(loop):
        loop.set_description('Image : {} of Slide {}'.format(img_name.split('.')[0],patient_name))
        qc_file_path=os.path.join(qc_path,'OverallQCMask_{}_{}.tif'.format(patient_name,img_name.split('.')[0]))
        qc_mask=imread(qc_file_path)
        qc_mask[qc_mask!=0]=1
        bound_img_path=os.path.join(PRED_PATH,'bound_'+img_name.split('.')[0]+'.png')
        bound_img=imread(bound_img_path)
        if type(threshold)==float:
            thresh_bound=threshold*255#threshold_otsu(bound_img)

            thresh_nuclei=threshold*255#threshold_otsu(nuc_img)
        else:
            thresh_bound=threshold_otsu(bound_img)

            thresh_nuclei=threshold_otsu(nuc_img)
        
        bound_img=bound_img>thresh_bound
        
        
        nuc_img_path=os.path.join(PRED_PATH,'nuclei_'+img_name.split('.')[0]+'.png')
        nuc_img=imread(nuc_img_path)
        nuc_img=nuc_img>thresh_nuclei
        
        
    
        bound_img=multiple_dialte(bound_img)
        bound_img=multiple_erosion(bound_img)
        
        nuc_img=multiple_erosion(nuc_img)
        nuc_img=multiple_dialte(nuc_img)
        
        comb_img=nuc_img^bound_img
        bound_coor=np.where(bound_img==1)
        comb_img[bound_coor]=0
        comb_img=multiple_erosion(comb_img,3)
        comb_img=multiple_dialte(comb_img,3)
        
        gt=int(df_whole[(df_whole['ROI']==img_name.split('.')[0]) & (df_whole['Slide']==patient_name)]["ALLCELLS"])
        
        imsave(processed_dir+'/'+img_name,img_as_uint(comb_img))
        
        if img_name==selected_roi:
            f, ax = plt.subplots(1,4,figsize=(30,30))
            plt.rcParams.update({'font.size': 32})
            ax[0].imshow(selected_ip,cmap='gray')
            ax[0].title.set_text('Input DAPI image')
            ax[1].imshow(selected_prediction,cmap='gray')
            ax[1].title.set_text('Model Prediction')
            ax[2].imshow(comb_img,cmap='gray')
            ax[2].title.set_text('Final output')
            ax[3].imshow(selected_gt,cmap='gray')
            ax[3].title.set_text('GT')
            
        count_img=comb_img.copy()
        count_img=count_img&qc_mask
        
        labels=skimage.measure.label(count_img)
        df_summary.loc[current_no_entries+i+1]=[patient_name,img_name.split('.')[0],np.amax(labels),gt,(gt-np.amax(labels))/gt]
        
        if print_prompt:
            print("For RoI {} the predicted count is : {},and the GT  is : {}\n".format(img_name.split('.')[0],np.amax(labels),gt)\
                  +Color.RED+"Error rate {}".format((gt-np.max(labels))/gt)+Color.END)
        avg_error+=np.abs((gt-np.max(labels)))/gt
    if print_prompt:
        print("avg error : ",avg_error/(i+1))
                                       
    return df_summary