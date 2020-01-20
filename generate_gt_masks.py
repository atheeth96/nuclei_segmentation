
import datetime
from zipfile import ZipFile
import math
import time
from skimage import morphology

from extractor_class.patch_extractor import PatchExtractor
from skimage.filters import threshold_otsu

import skimage
from skimage.io import imread,imsave
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook as tqdm 

# from callbacks_class.callbacks_dapi import CosineWithRestart
# from Models.Models import attn_unet
import os, shutil
from preprocess import generate_train_mask



patient_list=['S-190413-00146','S-190413-00110']
for patient in patient_list:
    generate_train_mask(patient,'Dapi_patient_data')
    