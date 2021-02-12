# -*- coding: utf-8 -*-
"""predict.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GjW55cnI_D8jw3vAFFozLZQW94bnaNvF
"""

!pip install segmentation_models
!pip install tensorflow==2.2.0
!pip install keras==2.3.1
#!pip install -U segmentation-models==1.0.0
!pip install -U git+https://github.com/albu/albumentations --no-cache-dir
!pip install geopandas

import os,sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import segmentation_models as sm
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import albumentations as A
import geopandas as gpd
import time
print("sm: ",sm.__version__,"keras:",keras.__version__,"tf: ",tf.__version__)

from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = '/content/drive/MyDrive/Dolin_Segmentation'
x_test_dir = os.path.join(DATA_DIR, 'predicted_image')
height=320
width=320

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

BACKBONE = 'densenet121'
#'densenet121'fenadeğil
#'resnet101'
#'resnet34'
#'efficientnetb6'
#'inceptionresnetv2'
BATCH_SIZE = 16
CLASSES = ['dolins']
LR = 0.0001 #0.0001 best for adam
EPOCHS = 100

preprocess_input = sm.get_preprocessing(BACKBONE)

#create model
model = sm.Unet(BACKBONE, classes=1, activation="sigmoid")

# load best weights
model.load_weights("/content/drive/MyDrive/Dolin_Segmentation/results/densenet121/best_model.h5")

image_path="/content/drive/MyDrive/Dolin_Segmentation/test_image.png"
c=cv2.imread(image_path)
plt.imshow(c)
plt.show()
print(c.shape)

#----------------------------------------------------------------------------------------------------
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from skimage import io
image_path="/content/drive/MyDrive/Dolin_Segmentation/test_image.png"
save_folder="/content/drive/MyDrive/Dolin_Segmentation/predicted_image"
h=320
w=320

def create_image(image_path,save_folder,h,w):
    #font = cv2.FONT_HERSHEY_SIMPLEX
    img=cv2.imread(image_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    back=np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    thickness=3#Karelaj kalınlık
    #color=(0,255,0)#karelaj renk
    #alan=img.shape[0]*img.shape[1]
    #print(img.shape)

    h=h
    w=w
    img2=np.zeros([h,w,3],dtype=np.float32)
    sutun=int(img.shape[1]/w)
    satir=int(img.shape[0]/h)
    #print("satir sayısı= ",satir,"sütün sayısı= ",sutun)
    x1=0
    y1=0
    x2=w
    y2=h
    
    for j in tqdm(range(satir)):
        #print("satir= ",j)
        for i in range(sutun):
            #print("Sütun= ",i)
            #print("buaraya geliyorum")
            ROI=img[y1:y2,x1:x2]
            #name=os.path.join(save_folder,"file"+str(j)+str(i)+".jpg")
            part=preprocess_input(ROI)
            part=np.expand_dims(part,axis=0)
            pred=model.predict(part).round()
            pred=pred[...,0].squeeze()
            pred=pred*255
            #print(y1,y2,x1,x2)
            back[y1:y2,x1:x2]=pred
            #ROI = cv2.putText(ROI, str(j)+str(i)+"ROI", (10,100), font,1, (0,0,255), 2, cv2.LINE_AA)
            #cv2.imwrite(name,ROI)
            #cv2.rectangle(back,(x1,y1),(x2,y2),color=color,thickness=thickness)
            #cv2.putText(back, str(j)+str(i),(x1,y1), font,3, (0,255,0), 5, cv2.LINE_AA)
            x1=x1+w
            x2=x1+w
            if x2>img.shape[1]:
                x1=0
                x2=w
                y1=y1+h
                y2=y1+h
                break     
    plt.imshow(back)
    plt.show()
    cv2.imwrite(os.path.join(save_folder,"result.png"),back)      
    print("Predicted image was created")

t1=time.time()
create_image(image_path,save_folder,h,w)
t2=time.time()
sure=t2-t1
print(sure)

mask=cv2.imread("/content/drive/MyDrive/Dolin_Segmentation/predicted_image/result.png",0)
#mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
plt.imshow(mask)
plt.show()

#create shp file from mask
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
t3=time.time()
mask=cv2.imread("/content/drive/MyDrive/Dolin_Segmentation/predicted_image/result.png",0)
image=cv2.imread("/content/drive/MyDrive/Dolin_Segmentation/test_image.png")
#im=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
im=np.where(mask<255,0,1).astype(np.uint8)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)#gürültüleri temizlemek için
im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)#gürültüleri temizlemek için
im = cv2.morphologyEx(im, cv2.MORPH_ELLIPSE, kernel)#gürültüleri temizlemek için


print(np.unique(im))
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(im, 0, 1)
contours,_ = cv2.findContours(edge.copy(), 
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)


for cnt in contours:
    cnt=np.squeeze(cnt)
    #print(cv2.contourArea(cnt))
    
 
   
    for i in range(len(cnt)):
        
        cnt[i][0]=354537.595924+(cnt[i][0]*0.3) #left coordinate *0.3 ise cell size
        cnt[i][1]=4416682.20762-(cnt[i][1]*0.3)  #Up coordinate
cv2.drawContours(im, contours, -1, (0,255,0), thickness = 2)
rslts=im.copy()
#rslts=np.zeros((512,512,3),dtype=np.float32)

cv2.polylines(image,contours,True,(0,255,0),3)


#creating shp file-----------------------------------------------
import geopandas as gpd
from shapely.geometry import Polygon,Point
from fiona.crs import from_epsg

fp="/content/drive/MyDrive/Dolin_Segmentation/predicted_image/dolin_2_zoom20.shp"
data=gpd.read_file(fp)
#data.plot()
#plt.show()
print(data.crs)
print(data.head(5))
for h in contours:
    A=cv2.contourArea(h)
    
new=gpd.GeoDataFrame()
for i,poly in enumerate(contours):
    
   
   
    p=Polygon(np.squeeze(poly))
    if p.area>1:
        new.loc[i,"geometry"]=p
  


new.crs=data.crs

#print(new.head(5),new.crs)#projection
new.to_file("/content/drive/MyDrive/Dolin_Segmentation/predicted_image/predicted_dolins.shp",)
t4=time.time()
sure2=t4-t3
print(sure2)
new.plot()
plt.show()