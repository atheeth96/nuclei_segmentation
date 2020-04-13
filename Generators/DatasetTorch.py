from skimage.io import imread
import numpy as np
import os
import skimage
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np
import matplotlib.pyplot as plt


def visualize_loader(loader,index):
    for i,sample in enumerate(loader):
        if i==1:
            image=(sample['image'][index]).numpy()
           
            mask=(sample['nuclei_mask'][index]).numpy()
            
        
            boundary=(sample['bound_mask'][index]).numpy()
            output=sample['nuclei_mask']
            output=torch.cat((output,sample['nuclei_mask']),dim=1)
           
    
            image=np.squeeze(image.transpose(1,2,0),axis=2)
            print(image.shape,np.amax(image))
            mask=np.squeeze(mask.transpose(1,2,0),axis=2) 
            
            boundary=np.squeeze(boundary.transpose(1,2,0),axis=2)
           
            
            return image*255,mask,boundary
            
            
            
def Sort_Tuple(tup):  
      
    # getting length of list of tuples 
    lst = len(tup)  
    for i in range(0, lst):  
          
        for j in range(0, lst-i-1):  
            if (tup[j][0] > tup[j + 1][0]):  
                temp = tup[j]  
                tup[j]= tup[j + 1]  
                tup[j + 1]= temp  
    return tup  


class DataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_dir, nuclei_mask_dir,bound_mask_dir,img_type='tif', transform=None):
        assert img_type=='tif',"Image format must be tif"
        assert (img_type=='h5py' and transform==None) or (img_type!='h5py' and transform!=None),'Transforms cannot be applied to tensors from h5py file'
        self.img_type=img_type
        self.img_dir=img_dir
        self.nuclei_mask_dir = nuclei_mask_dir
        self.bound_mask_dir=bound_mask_dir
        self.transform = transform
        self.img_list=[]
        img_tuple=[(int(x.split('.')[0].split('_')[-1]),x) for x in os.listdir(self.img_dir)]
        img_tuple=Sort_Tuple(img_tuple)
        for i in range(len(img_tuple)):
            self.img_list.append(img_tuple[i][1])
#Returns length of data-set unlike its keras counter part that returns no_batches
    def __len__(self):
        return len([x for x in os.listdir(self.img_dir) if x.split('.')[-1]=='tif'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        img_name = os.path.join(self.img_dir,
                                self.img_list[idx])
        #print(img_name)
        nuclei_mask_name = os.path.join(self.nuclei_mask_dir,
                                [x for x in os.listdir(self.nuclei_mask_dir) if img_name.split('/')[-1] in x][-1])
        bound_mask_name = os.path.join(self.bound_mask_dir,
                                [x for x in os.listdir(self.bound_mask_dir) if img_name.split('/')[-1] in x][-1])
        image = np.expand_dims(imread(img_name),axis=2).astype(np.int64)
        
        nuclei_mask=np.expand_dims(imread(nuclei_mask_name),axis=2).astype(np.uint8)
        bound_mask=np.expand_dims(imread(bound_mask_name),axis=2).astype(np.uint8)
        sample = {'image': image, 'nuclei_mask': nuclei_mask,'bound_mask':bound_mask}

        if self.transform:
            sample = self.transform(sample)

        return sample
class Scale(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self,scale_type='maximum'):
        assert scale_type in ['maximum','pseudo_max','1.5times_mean'],'Only following scale types are allowed : maximum, pseudo_max, 1.5times_mean'
        self.scale_type=scale_type

    def __call__(self,sample):
        image, nuclei_mask,bound_mask = sample['image'], sample['nuclei_mask'],sample['bound_mask']
#         image_max=np.amax(image)
        image_mean=np.mean(image)
        if self.scale_type=='1.5times_mean':
            upper_limit=1.5*image_mean
            #lower_limit=0.65*image_mean
            
            #image[image<lower_limit]=0
            image[image>upper_limit]=upper_limit
            
            image_scale=image/upper_limit
        elif self.scale_type=='pseudo_max':
            scale=np.percentile(image,90)
            image=image/scale
            image[image>=1]=1
            
        else:
            image=image/np.amax(image)
        
        mask_scale=255

        nuclei_mask = nuclei_mask/mask_scale
        nuclei_mask=nuclei_mask.astype(np.uint8)
        bound_mask = bound_mask/mask_scale
        bound_mak=bound_mask.astype(np.uint8)
        return {'image': image,
                'nuclei_mask': nuclei_mask,
               'bound_mask': bound_mask}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, nuclei_mask,bound_mask = sample['image'], sample['nuclei_mask'],sample['bound_mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        nuclei_mask = nuclei_mask.transpose((2, 0, 1))
        bound_mask = bound_mask.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).type(torch.FloatTensor),
                'nuclei_mask': torch.from_numpy(nuclei_mask).float(),
               'bound_mask': torch.from_numpy(bound_mask).float()}
    

class RandomHorizontalFlip(object):
 

    def __call__(self, sample):
        
        image, nuclei_mask,bound_mask = sample['image'], sample['nuclei_mask'],sample['bound_mask']
        
   
        image=  np.flip(image, 0)
        nuclei_mask= np.flip(nuclei_mask, 0)
        bound_mask=np.flip(bound_mask, 0)
            
        return {'image': image,
                'nuclei_mask': nuclei_mask,
               'bound_mask': bound_mask}
    

class GaussianNoise(object):
 
   
       
    
    def __call__(self, sample):
        
        image, nuclei_mask,bound_mask = sample['image'], sample['nuclei_mask'],sample['bound_mask']
        
        image=skimage.util.random_noise(image)
        
            
        return {'image': image,
                'nuclei_mask': nuclei_mask,
               'bound_mask': bound_mask}
    
class ContrasrStretching(object):
 
  
    
    def __call__(self, sample):
        
        image, nuclei_mask,bound_mask = sample['image'], sample['nuclei_mask'],sample['bound_mask']
        
        p2, p98 = np.percentile(image, (2, 98))
        
        image=skimage.exposure.rescale_intensity(image,in_range=(p2,p98))
        
            
        return {'image': image,
                'nuclei_mask': nuclei_mask,
               'bound_mask': bound_mask}

    
    
class RandomCropResize(object):
 
    def __init__(self, size=(512,512)):
        
        self.size=size
        
    
    def get_params(self,img, output_size=(256,256)):
        
        w, h = img.shape[:2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw
    

    def __call__(self, sample):
        
        image, nuclei_mask,bound_mask = sample['image'], sample['nuclei_mask'],sample['bound_mask']
        
       
        i,j,h,w=self.get_params(image)

        image=skimage.transform.resize(image[i:i+w,j:j+ h],(512,512))
        nuclei_mask=skimage.transform.resize(nuclei_mask[i:i+w,j:j+ h],(512,512))
        bound_mask=skimage.transform.resize(bound_mask[i:i+w,j:j+ h],(512,512))
            
            
#             image= F2.resize(image, self.size, self.interpolation)
            
#             nuclei_mask= F2.resize(nuclei_mask, self.size, self.interpolation)
#             bound_mask= F2.resize(bound_mask, self.size, self.interpolation)
            
        return {'image': image,
                'nuclei_mask': nuclei_mask,
               'bound_mask': bound_mask}
    
class  Color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'