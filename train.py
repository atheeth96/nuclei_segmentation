import datetime
import time

import math
# import matplotlib.pyplot as plt
import numpy as np



from tqdm import tqdm
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
from Losses import SoftDiceLoss,MultiClassBCE,dice_metric
import torch.nn as nn
from torch.utils.data import DataLoader
from Models.ModelsTorch import AttnUNet,R2U_Net,R2AttU_Net,save_model,load_model
from Generators.DatasetTorch import DataSet,ToTensor,Scale,Color,\
RandomHorizontalFlip,GaussianNoise,RandomCropResize,ContrasrStretching,visualize_loader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import pandas as pd 

from Utils.Preprocess import fuse_roi,generate_train_mask
from Utils.Prediction import whole_img_pred,post_process
import tensorboard

###### Fusing ROIS #####

# Need to run this only once
# Comment out the for loops after first run

patients_done=['S-190413-00208','S-190413-00163','S-190501-00409','S-190413-00113','S-190413-00054','S-190413-00238',\
              'S-190413-00104','S-190413-00137','S-190413-00217','S-190413-00098','S-190413-00140','S-190413-00107',\
              'S-190501-00428','S-190501-00422','S-190501-00425','S-190501-00425']

patient_list=list(set(slides_to_transfer)-set(patients_done))

# for patient in patient_list:
#     print(patient)
#     fuse_roi(patient,'Dapi_patient_data')

###### Develop mask ######

# for patient in ['S-190501-00431']:
#     generate_train_mask(patient,'Dapi_patient_data')


###### Extract Patches ######

# Need to run this only once
# Comment out the for class methods after first run

patient_list=[os.path.join('Dapi_patient_data',x) for x in os.listdir('Dapi_patient_data') if 'S-19' in x and x!='S-190501-00425']

test_list=[os.path.join('Dapi_patient_data',x) for x in\
           ['S-190413-00092','S-190410-00044','S-190413-00078','S-190413-00075','S-190413-00072']]


print("TEST\n",*test_list,sep = "\n")
train_list=[x for x in patient_list if x not in test_list]
train_list=np.random.choice(train_list,15)
print('\n')
print("TRAIN\n",*train_list,sep = "\n")

img_patches_dir_train='processed_data/img_patches_train'
nuc_pathces_dir_train='processed_data/nuc_patches_train'
bound_patches_dir_train='processed_data/bound_patches_train'

img_patches_dir_test='processed_data/img_patches_test'
nuc_pathces_dir_test='processed_data/nuc_patches_test'
bound_patches_dir_test='processed_data/bound_patches_test'

# train_extractor=PatchExtractor(train_list,img_patches_dir_train\
#                                ,nuc_pathces_dir_train,bound_patches_dir_train)

# train_extractor.extract_patches()

# test_extractor=PatchExtractor(test_list,img_patches_dir_test\
#                                ,nuc_pathces_dir_test,bound_patches_dir_test)

# test_extractor.extract_patches()

######  Create Dataset and push to Dataloader ###### 

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


train_dataset=DataSet(img_patches_dir_train,nuc_pathces_dir_train,bound_patches_dir_train,transform=transform)
train_loader=DataLoader(train_dataset,batch_size=batch_size_train,num_workers=0,shuffle=True)
print(train_dataset.__len__()," Train samples")

test_dataset=DataSet(img_patches_dir_test,nuc_pathces_dir_test,bound_patches_dir_test,transform=test_transform)
test_loader=DataLoader(test_dataset,batch_size=batch_size_test,num_workers=0,shuffle=False)
print(test_dataset.__len__()," Test samples")


##### Visualize loader ####

image,mask,boundary=visualize_loader(test_loader,0)

##### Define model ####

model=AttnUNet(img_ch=1,output_ch=2)
model_start_date=datetime.datetime.now().strftime("%Y_%m_%d")
BEST_MODEL_PATH=os.path.join(os.getcwd(),'Trained_models/model_{}'.format(model_start_date))
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

#### Start training ######

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
                model.zero_grad()
                
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
                test_dice,test_dice_list=dice_metric(test_outputs,test_mask_batch)
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



