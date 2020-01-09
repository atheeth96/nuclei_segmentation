
import datetime
import time

import math
import matplotlib.pyplot as plt
import numpy as np



from tqdm import tqdm_notebook as tqdm 

import os, shutil
from zipfile import ZipFile

import skimage
from skimage.io import imread,imsave
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage import img_as_uint

from extractor_class.patch_extractor import PatchExtractor

import torch
import torchvision
from losses import SoftDiceLoss,MultiClassBCE,dice_metric
import torch.nn as nn
from torch.utils.data import DataLoader
from Models.ModelsTorch import AttnUNet,R2U_Net,R2AttU_Net,save_model,load_model
from Generators.DatasetTorch import DataSet,ToTensor,Scale,Color,\
RandomHorizontalFlip,GaussianNoise,RandomCropResize,ContrasrStretching
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import pandas as pd 

from preprocess import fuse_roi,generate_train_mask
from prediction import whole_img_pred,post_process


import tensorboard
print(tensorboard.__version__)


df_wt=pd.read_excel('Summary_v3_15102019.xlsx',sheet_name='within_tumor')
df_wt.columns=[x.replace(" ","_") for x in df_wt.columns]
df_ot=pd.read_excel('Summary_v3_15102019.xlsx',sheet_name='outside_tumor')
df_ot.columns=[x.replace(" ","_") for x in df_ot.columns]
df_whole=pd.read_excel('Summary_v3_15102019.xlsx',sheet_name='whole_tissue')
df_whole.columns=[x.replace(" ","_") for x in df_whole.columns]


####### MOVING DATA #############

srikanth_data='/home/vahadaneabhi01/datalab/training-assets/R_medical/srikanth/data'
srikanth_slides=os.listdir(srikanth_data)
atheeth_list=[x for x in os.listdir('Dapi_patient_data/') if 'S-' in x and '.zip' not in x]
slides_to_transfer=list(set(srikanth_slides)-(set(atheeth_list)&set(srikanth_slides)))
# for slide in tqdm(slides_to_transfer):
#     os.system('cp -r {} {}'.format(srikanth_data+'/'+slide,'Dapi_patient_data/'))


#################################




####### FUSING ROIS #############

patients_done=['S-190413-00208','S-190413-00163','S-190501-00409','S-190413-00113','S-190413-00054','S-190413-00238',\
              'S-190413-00104','S-190413-00137','S-190413-00217','S-190413-00098','S-190413-00140','S-190413-00107',\
              'S-190501-00428','S-190501-00422','S-190501-00425','S-190501-00425']

patient_list=list(set(slides_to_transfer)-set(patients_done))#['S-190413-00146','S-190413-00110'] #'S-190413-00241','S-190413-00143',
# for patient in patient_list:
#     print(patient)
#     fuse_roi(patient,'Dapi_patient_data')


#################################




####### DEVELOP MASK #############

# for patient in ['S-190501-00431']:
#     generate_train_mask(patient,'Dapi_patient_data')
    

#################################




###################################### EXTRACT PATCHES #################################################################


patient_list=[os.path.join('Dapi_patient_data',x) for x in os.listdir('Dapi_patient_data') if 'S-19' in x and x!='S-190501-00425']

test_list=[os.path.join('Dapi_patient_data',x) for x in\
           ['S-190413-00092','S-190410-00044','S-190413-00078','S-190413-00075','S-190413-00072']]


print("TEST\n",*test_list,sep = "\n")
train_list=[x for x in patient_list if x not in test_list]
train_list=np.random.choice(train_list,15)
print('\n')
print("TRAIN\n",*train_list,sep = "\n")

img_patches_dir_train='img_patches_train_15'
nuc_pathces_dir_train='nuc_patches_train_15'
bound_patches_dir_train='bound_patches_train_15'

img_patches_dir_test='img_patches_test_5'
nuc_pathces_dir_test='nuc_patches_test_5'
bound_patches_dir_test='bound_patches_test_5'

# train_extractor=PatchExtractor(train_list,img_patches_dir_train\
#                                ,nuc_pathces_dir_train,bound_patches_dir_train)

# train_extractor.extract_patches()

# test_extractor=PatchExtractor(test_list,img_patches_dir_test\
#                                ,nuc_pathces_dir_test,bound_patches_dir_test)

# test_extractor.extract_patches()



############################################################################################################################




####### DATA SEG #############

# First wave¶
# Train List

# S-190413-00092
# S-190410-00044
# S-190413-00078
# S-190413-00075

# Test List S-190413-00081
# S-190413-00072

# Second wave
# TEST

# S-190413-00092
# S-190410-00044
# S-190413-00078
# S-190413-00075
# S-190413-00072

# TRAIN

# S-190501-00431
# S-190413-00110
# S-190413-00140
# S-190413-00095
# S-190413-00134
# S-190413-00081
# S-190413-00054
# S-190413-00208
# S-190413-00054
# S-190413-00208
# S-190413-00146
# S-190413-00104
# S-190413-00208
# S-190413-00208
# S-190413-00098

#################################




########################################### CREATE DATASET AND PUSH TO DATA LOADER #############################################
batch_size_train=8
batch_size_test=4
augmentation_transform=[GaussianNoise(),RandomCropResize(),ContrasrStretching()]
add_augmentaion=False
if add_augmentaion: 
    transform=torchvision.transforms.Compose([Scale(input_image=True),\
                                                  torchvision.transforms.RandomApply(augmentation_transform, p=0.35)\
                                                  ,ToTensor()])
else:
    transform=torchvision.transforms.Compose([Scale(scale_type='maximum'),ToTensor()])    
test_transform=torchvision.transforms.Compose([Scale(scale_type='maximum'),ToTensor()])


train_dataset=DataSet('img_patches_train_15','nuc_patches_train_15','bound_patches_train_15',transform=transform)
train_loader=DataLoader(train_dataset,batch_size=batch_size_train,num_workers=0,shuffle=True)
print(train_dataset.__len__()," Train samples")

test_dataset=DataSet('img_patches_test_5','nuc_patches_test_5','bound_patches_test_5',transform=test_transform)
test_loader=DataLoader(test_dataset,batch_size=batch_size_test,num_workers=0,shuffle=False)
print(test_dataset.__len__()," Test samples")


    

################################################################################################################################



########################################## VISUALIZE LOADER ################################################


def visualize_loader(loader,index):
    for i,sample in enumerate(loader):
        #print(sample['image'].shape)
        if i==1:
            image=(sample['image'][index]).numpy()
            
            
            #image=np.zeros(image_i.shape,dtype=np.uint8)
            #image[np.where(image_i!=0)]=255

            mask=(sample['nuclei_mask'][index]).numpy()
            
        
            boundary=(sample['bound_mask'][index]).numpy()
            output=sample['nuclei_mask']
            output=torch.cat((output,sample['nuclei_mask']),dim=1)
           
    
            image=np.squeeze(image.transpose(1,2,0),axis=2)
            print(image.shape,np.amax(image))
            mask=np.squeeze(mask.transpose(1,2,0),axis=2) 
            
            boundary=np.squeeze(boundary.transpose(1,2,0),axis=2)
            #print(image)
            fig=plt.figure()
            plt.imshow(image*255)
            fig2=plt.figure()
            plt.imshow(mask)
            fig3=plt.figure()
            plt.imshow(boundary)
            break
visualize_loader(test_loader,0)


#######################################################################################################



########################################## LOSS AND METRIC #########################################


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        #probs = F.sigmoid(logits)
        m1 = logits.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score
    
class MultiClassBCE(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        cat_class=targets.size(1)
        m1 = logits.view(num,cat_class, -1)
        m2 = targets.view(num,cat_class, -1)
        final_loss=0
        loss_list=[]
        weights=[0.6,0.4]
        for cat in range(cat_class):
            if cat==1:
                loss=nn.BCELoss()(m1[:,cat,:],m2[:,cat,:])
            else:
                loss=nn.BCELoss()(m1[:,cat,:],m2[:,cat,:])
            final_loss+=weights[cat]*loss
            

        return final_loss
    
def dice_metric(y_pred,y_true):
    smooth = 1
    num = y_true.size(0)
    categories=y_true.size(1)
    m1 = y_pred.view(num,categories, -1)
    m2 = y_true.view(num,categories, -1)
    weights=[0.5,0.5]
    final_score=0
    score_list=[]
    for cat in range(categories):
        
        
        intersection = (m1[:,cat,:] * m2[:,cat,:])

        score = 2. * (intersection.sum(1) + smooth) / (m1[:,cat,:].sum(1) + m2[:,cat,:].sum(1) + smooth)
        score = score.sum() / num
        score=score.detach().item()
        final_score+=score*weights[cat]
        score_list.append(score)
    return final_score,score_list
    

#######################################################################################################



####### DEFINE MODEL  #############

model=AttnUNet(img_ch=1,output_ch=2)
# model=R2U_Net(img_ch=1,output_ch=2,t=2)
model_start_date=datetime.datetime.now().strftime("%Y_%m_%d")
BEST_MODEL_PATH=os.path.join(os.getcwd(),'model_{}'.format(model_start_date))
if not os.path.exists(BEST_MODEL_PATH):
    os.mkdir(BEST_MODEL_PATH)
    print('model_{} dir has been made'.format(model_start_date))
print("Model's state_dict:")
writer = SummaryWriter('model_{}/dapi_seg_experiment_{}'.format(model_start_date,1))
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
images= next(iter(train_loader))
model = model.to(device)
writer.add_graph(model,images['image'].to(device, dtype = torch.float))
# writer.flush()

#################################




########################################## START TRAINING #########################################


print(torch.cuda.get_device_name(torch.cuda.current_device()))
optimizer_selected='adam'
batchsize=4
no_steps=train_dataset.__len__()//batchsize
restart_epochs=8
num_epochs=10


#criterion = SoftDiceLoss()#
criterion=MultiClassBCE()

history={'train_loss':[],'test_loss':[],'train_dice':[],'test_dice':[]}
if optimizer_selected=='adam':
    optimizer = torch.optim.Adam(model.parameters(),lr=10e-03, betas=(0.9, 0.98))#,weight_decay=0.02)
else:
    optimizer = torch.optim.SGD(model.parameters(),lr=10e-03, momentum=0.8,nesterov=True)

scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, restart_epochs*no_steps,\
                                                     eta_min=10e-012, last_epoch=-1)


best_val=0
for epoch in range(num_epochs):
    
    print("Learning Rate : {}".format(optimizer.state_dict()['param_groups'][-1]['lr']))
    # loop over the dataset multiple times
    
    run_avg_train_loss=0
    run_avg_train_dice=0
    
    run_avg_train_dice_nuclei=0
    run_avg_train_dice_bound=0
    run_avg_test_dice_nuclei=0
    run_avg_test_dice_bound=0
    
    run_avg_test_loss=0
    run_avg_test_dice=0
    
    for mode in ['train','eval']:
     
        if mode == 'train':
            
            model.train()
            loop=tqdm(train_loader)
            
            for i, sample_batched in (enumerate(loop)):
                loop.set_description('Epoch {}/{}'.format(epoch + 1, num_epochs))
                
                #Clear Gradients
                optimizer.zero_grad()
                
                # get the inputs; data is a list of [dapi, nuclei, boundary]
                images_batch, nuc_mask_batch,bound_mask_batch = sample_batched['image'],\
                sample_batched['nuclei_mask'],sample_batched['bound_mask']
                nuclei_mask_batch=torch.cat((nuc_mask_batch,bound_mask_batch),dim=1)
             
                images_batch, nuclei_mask_batch = images_batch.to(device, dtype = torch.float)\
                ,nuclei_mask_batch.to(device, dtype = torch.float)#,bound_mask_batch.to(device,dtype=torch.float)

                # forward + backward + optimize
                outputs = torch.sigmoid(model(images_batch))
                
                loss = criterion(outputs, nuclei_mask_batch)
                dice_score,dice_list=dice_metric(outputs,nuclei_mask_batch)
                run_avg_train_loss=(run_avg_train_loss*(0.9))+loss.detach().item()*0.1
                run_avg_train_dice=(run_avg_train_dice*(0.9))+dice_score*0.1
                run_avg_train_dice_nuclei=(run_avg_train_dice_nuclei*(0.9))+dice_list[0]*0.1
                run_avg_train_dice_bound=(run_avg_train_dice_bound*(0.9))+dice_list[1]*0.1
                if i%100==99:
                    predicted_nuclei_train,predicted_boundary_train=torch.chunk(outputs,2,dim=1)
                    gt_nuclei,gt_boundary=torch.chunk(nuclei_mask_batch.detach().cpu(),2,dim=1)
                    
                    img_tensor=torch.cat((predicted_nuclei_train.detach().cpu()\
                                         ,gt_nuclei\
                                          ,predicted_boundary_train.detach().cpu(),gt_boundary),axis=0)
                    
                    
                    #print(img_tensor.shape)
                    img_grid2 = torchvision.utils.make_grid(img_tensor,nrow=batch_size_train,padding=100)
                    torchvision.utils.save_image\
                    (img_grid2,os.path.join(BEST_MODEL_PATH,\
                                            'train_iter_{}.png'.format(epoch*len(train_loader)+i+1)))
                    writer.add_image('TRAIN_EPOCH_{}_ITER_{}'.format(epoch,i), img_grid2)
                    writer.add_histogram('First conv weights',model.Conv1.conv[0].weight,i)
                    
                    writer.add_scalar('Training dice score nuclei',
                            run_avg_train_dice_nuclei,
                            epoch * len(train_loader) + i)
                    writer.add_scalar('Training dice score boundary',
                            run_avg_train_dice_bound,
                            epoch * len(train_loader) + i)
                    writer.add_scalar('Training Loss',
                            run_avg_train_loss,
                            epoch * len(train_loader) + i)
                    writer.flush()
                    
                loss.backward()
                optimizer.step()
                
                scheduler.step()
                
                
                loop.set_postfix(loss=run_avg_train_loss,dice_score=run_avg_train_dice,\
                                 nuclei_dice=run_avg_train_dice_nuclei,\
                                boundary_dice=run_avg_train_dice_bound)
               
            history['train_loss'].append(run_avg_train_loss)
            history['train_dice'].append(run_avg_train_dice)
                
                 
                    
        elif mode =='eval':
            #Clear Gradients
            optimizer.zero_grad()
            samples_test=len(test_loader)
            model.eval()
            val_loss=0
            test_agg=0
            for j, test_sample in enumerate(test_loader):

                test_images_batch, test_nuclei_mask_batch,test_bound_mask_batch = test_sample['image'], test_sample['nuclei_mask'],\
                test_sample['bound_mask']
                test_mask_batch=torch.cat((test_nuclei_mask_batch,test_bound_mask_batch),dim=1)
                test_images_batch, test_mask_batch = test_images_batch.to(device, dtype = torch.float),\
                test_mask_batch.to(device, dtype = torch.float)
                test_outputs = torch.sigmoid(model(test_images_batch))
                
                test_loss = criterion(test_outputs, test_mask_batch)
                #final_test_loss+=test_loss.detach().item()
                test_dice,test_dice_list=dice_metric(test_outputs,test_mask_batch)
                #final_test_dice+=test_dice
                run_avg_test_loss=(run_avg_test_loss*(0.9))+test_loss.detach().item()*0.1
                run_avg_test_dice=(run_avg_test_dice*(0.9))+test_dice*0.1
                run_avg_test_dice_nuclei=run_avg_test_dice_nuclei*0.9+test_dice_list[0]*0.1
                run_avg_test_dice_bound=run_avg_test_dice_bound*0.9+test_dice_list[1]*0.1
                if j%100==99:
                    predicted_nuclei_test,predicted_boundary_test=torch.chunk(test_outputs.detach(),2,dim=1)
                    gt_nuclei_test,gt_boundary_test=torch.chunk(test_mask_batch.detach().cpu(),2,dim=1)
                    
                    img_tensor_test=torch.cat((predicted_nuclei_test.detach().cpu()\
                                              ,gt_nuclei_test\
                                               ,predicted_boundary_test.detach().cpu(),gt_boundary_test),axis=0)
                
                    
                    img_grid = torchvision.utils.make_grid(img_tensor_test,nrow=batch_size_test,padding=10)
                    torchvision.utils.save_image\
                    (img_grid,os.path.join(BEST_MODEL_PATH,\
                                            'test_iter_{}.png'.format(epoch*len(train_loader)+j+1)))
                    writer.add_image('TEST_EPOCH_{}_ITER_{}'.format(epoch,j), img_grid)
                    writer.add_scalar('Testing dice score nuclei',\
                                      run_avg_test_dice_nuclei,epoch * len(test_loader) + j)
                    writer.add_scalar('Testing dice score boundary',\
                                      run_avg_test_dice_bound,epoch * len(test_loader) + j)
                    writer.add_scalar('Testing Loss',\
                                      run_avg_test_loss,epoch * len(test_loader) + j)
                    writer.flush()
                
            print("test_loss: {}\ntest_dice :{}, list [nuclei,boundary] :{}"\
                  .format(run_avg_test_loss,run_avg_test_dice,[run_avg_test_dice_nuclei,run_avg_test_dice_bound]))
            history['test_loss'].append(run_avg_test_loss)
            history['test_dice'].append(run_avg_test_dice)
            if run_avg_test_dice>best_val:
                best_val=run_avg_test_dice
                save_model(model,optimizer,BEST_MODEL_PATH+'/model_optim.pth',scheduler=scheduler)
            
                print("saved model with test dice score: {}".format(best_val))

    
save_model(model,optimizer,BEST_MODEL_PATH+'/model_final.pth',scheduler=scheduler)
        
        

print('Finished Training')





#######################################################################################################



########################################## PREDICT AND STITCH BACK WHOLE IMAGE #########################################


#model=R2U_Net(img_ch=1,output_ch=2,t=2)
model=AttnUNet(img_ch=1,output_ch=2)
#print(model)
patient_name='S-190413-00241'
model.load_state_dict(torch.load('model_2019_11_13/model_optim.pth'))
IMAGE_PATH='/datalab/training-assets/R_medical/DAPI/Dapi_patient_data/{}'.format(patient_name)
pred_dir_name=os.path.join(IMAGE_PATH,'predictions')
#whole_img_pred(IMAGE_PATH,os.listdir(os.path.join(IMAGE_PATH,'ROI')),pred_dir_name,model)
#zipFolder('Dapi_patient_data/{}/predictions/'.format(patient_name),'{}_predictions'.format(patient_name))


img_list=os.listdir(IMAGE_PATH+'/ROI')
PRED_PATH=os.path.join(IMAGE_PATH,'predictions')
processed_dir=os.path.join(IMAGE_PATH,'processed')
post_process(PRED_PATH,img_list,processed_dir)



BASE_PATH='Dapi_patient_data/'
patient_list=[x for x in os.listdir(BASE_PATH) if 'S-' in x and '.zip' not in x]

def tabulate_anomalies(BASE_PATH,patient_list)
    patient_list.sort()
    print(patient_list)
    count=0
    df_new=pd.DataFrame(columns=['Slide','ROI','Nuclei_count_in_whole_tissue'\
                     ,'Nuclei_count_in_within_tumor','Nuclei_count_in_outside_tissue',\
                    'Nuclei_count_in_images',])
    for patient in tqdm(patient_list):
    roi_list=os.listdir(os.path.join(BASE_PATH,patient+'/ROI'))
    roi_list.sort()
    ground_truth_list=os.listdir(os.path.join(BASE_PATH,patient+'/NucSeg'))
        for roi in roi_list:
            ground_truth_path=os.path.join(os.path.join(BASE_PATH,patient+'/NucSeg'),\
                                           [x for x in ground_truth_list if roi.split('.')[0] in x][-1])
        #         print(os.path.join(BASE_PATH,patient+'/NucSeg'),\
        #                                        [x for x in ground_truth_list if roi.split('.')[0] in x][-1],j)
            ground_truth=imread(ground_truth_path)
            all_cells_gt=np.amax(ground_truth)
            roi_name=roi.split('.')[0]
            #print(roi_name,patient)
            all_cells_wt=int(df_wt[(df_wt['ROI']==roi_name) & (df_wt['Slide']==patient)]["ALLCELLS"])
            all_cells_ot=int(df_ot[(df_ot['ROI']==roi_name) & (df_ot['Slide']==patient)]["ALLCELLS"])
            all_cells_whole=int(df_whole[(df_whole['ROI']==roi_name) & (df_whole['Slide']==patient)]["ALLCELLS"])


            df_new.loc[count]=[patient,roi_name,all_cells_whole,all_cells_wt,all_cells_ot,all_cells_gt]
            count+=1
    df_new.to_csv('anomalies.csv',index=False)





