from skimage.io import imread
import numpy as np
import os
import skimage
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np



class DataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, biomarker_img_dir, mask_dir, transform=None):
        
       
        self.biomarker_img_dir=biomarker_img_dir
        self.mask_dir = mask_dir
 
        self.transform = transform
        self.img_list=[]
        self.img_list=os.listdir(self.biomarker_img_dir)
#Returns length of data-set unlike its keras counter part that returns no_batches
    def __len__(self):
        return len([x for x in os.listdir(self.biomarker_img_dir) if x.split('.')[-1]=='tif'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        img_name = os.path.join(self.biomarker_img_dir,
                                self.img_list[idx])
        
        mask_name = os.path.join(self.mask_dir,
                                [x for x in os.listdir(self.mask_dir) if 'LabelMap_'+'_'.join(img_name.split('/')[-1].split('_')[1:]) in x][-1])
       
        image = np.expand_dims(imread(img_name),axis=2).astype(np.int64)
        
        mask=np.expand_dims(imread(mask_name),axis=2).astype(np.uint8)
        
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample
class Scale(object):
  

    def __call__(self,sample):
        image, mask = sample['image'], sample['mask']
        image=image/np.amax(image)
        
        mask_scale=255

        mask = mask/mask_scale
        
        return {'image': image,
                'mask': mask}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image).type(torch.FloatTensor),
                'mask': torch.from_numpy(mask).float()}
    


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