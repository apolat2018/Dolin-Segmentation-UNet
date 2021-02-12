
"""görüntüler ve mask imageları varsa bu verilerden train test data seti hazırlar"""
import os, sys
import random
import shutil
import numpy as np
from tqdm import tqdm

random.seed(42)

path_="C:\\Users\\ap\\Documents\\GitHub\\geospatial_deneme\\dolins\\DATA_ORTO"
dataset_path="C:\\Users\\ap\\Documents\\GitHub\\geospatial_deneme\\dolins\\DATA_ORTO\\dataset320"
img_path="C:\\Users\\ap\\Documents\\GitHub\\geospatial_deneme\\dolins\\DATA_ORTO\\image_320"
mask_path="C:\\Users\\ap\\Documents\\GitHub\\geospatial_deneme\\dolins\\DATA_ORTO\\mask_320"

train_dir=os.path.join(dataset_path,"train")
test_dir=os.path.join(dataset_path,"test")

train_mask_dir=os.path.join(dataset_path,"train_mask")
test_mask_dir=os.path.join(dataset_path,"test_mask")


if os.path.isdir(dataset_path)==False:
    os.mkdir(dataset_path)

if os.path.isdir(train_dir)==False:
    os.mkdir(train_dir)
    print("Klasör oluşturuldu")
else:
    print("Klasör var")
if os.path.isdir(test_dir)==False:
    os.mkdir(test_dir)
    print("Klasör oluşturuldu")
else:
    print("Klasör var")

if os.path.isdir(train_mask_dir)==False:
    os.mkdir(train_mask_dir)
    print("Klasör oluşturuldu")
else:
    print("Klasör var")

if os.path.isdir(test_mask_dir)==False:
    os.mkdir(test_mask_dir)
    print("Klasör oluşturuldu")
else:
    print("Klasör var")
    
img_f=sorted(os.listdir(img_path))
mask_f=sorted(os.listdir(mask_path))

file_number=len(img_f)#total image number

test_number=round(file_number*0.33)
train_number=file_number-test_number
print("Toplam görüntü sayısı: ",file_number,"Train: ",train_number,"Test : ",test_number)


test_ids=[]
train_ids=np.arange(0,file_number)
random.shuffle(train_ids)

#create test ids
for i in range(0,test_number):
    x=random.choice(train_ids)
    test_ids.append(x)
    train_ids=np.delete(train_ids,np.argwhere(train_ids == x))
#-----------------------------------------------
for i in tqdm(test_ids):
    im=os.path.join(img_path,img_f[i])
    msk=os.path.join(mask_path,mask_f[i])
    im_target=os.path.join(test_dir,img_f[i])
    msk_target=os.path.join(test_mask_dir,mask_f[i])
    #print(im,msk)
    #print(im_target,msk_target)
    shutil.copyfile(im,im_target)
    shutil.copyfile(msk,msk_target)

for i in tqdm(train_ids):
   
    im=os.path.join(img_path,img_f[i])
    msk=os.path.join(mask_path,mask_f[i])
    im_target=os.path.join(train_dir,img_f[i])
    msk_target=os.path.join(train_mask_dir,mask_f[i])
    #print(im,msk)
    #print(im_target,msk_target)
    shutil.copyfile(im,im_target)
    shutil.copyfile(msk,msk_target)

print("All is well")
