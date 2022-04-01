# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:05:46 2022

@author: ap
"""
# -*- coding: utf-8 -*-

#180. satır image size göre değiştirilmelidir. Bu kod 128x128 için yazılmıştır.
import os,sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import segmentation_models as sm
import cv2
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import albumentations as A
import geopandas as gpd
import time
from skimage import io
from tqdm import tqdm
sm.set_framework('tf.keras')
sm.framework()
print("sm: ",sm.__version__,"keras:",keras.__version__,"tf: ",tf.__version__)

DATA_DIR = 'D:/Dolin_Segmentation/dataset320'
height=320
width=320

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'val_mask')

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(18, 6))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

class Dataset:
    
    CLASSES = ["background","dolin"]
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        print(type(preprocessing))
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = np.where(mask>0,1,0)#bu ra benim 
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
            
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

BACKBONE = 'efficientnetb3'
#'densenet121'fenadeğil
#'resnet101'
#'resnet34'
#'efficientnetb6'
#'inceptionresnetv2'
preprocess_input = sm.get_preprocessing(BACKBONE)
#create model
model = sm.Unet(BACKBONE, activation="sigmoid")

# load best weights
model.load_weights("D:/Dolin_Segmentation/modeller/efficientnetb3.h5")

CLASSES = ["background","dolin"]
valid_2 = Dataset(x_valid_dir,y_valid_dir,classes=CLASSES)

X=[]
Y=[]
for i in range(len(valid_2)):
    x,y=valid_2[i]
    
    
    X.append(x)
    Y.append(y)
X=np.array(X,dtype="float32")
Y=np.array(Y,dtype=int)
prd=(model.predict(X)>=0.5).astype(int)
Y_pred=prd
Y_val=Y
import seaborn as sns

FP = len(np.where(Y_pred - Y_val  == -1)[0])
FN = len(np.where(Y_pred - Y_val  == 1)[0])
TP = len(np.where(Y_pred + Y_val ==2)[0])
TN = len(np.where(Y_pred + Y_val == 0)[0])
cmat = [[TP, FN], [FP, TN]]

plt.figure(figsize = (6,6))
sns.heatmap(cmat/np.sum(cmat), cmap="Reds", annot=True, fmt = '.2%', square=1,   linewidth=2.)
plt.xlabel("predictions")
plt.ylabel("real values")
plt.show()