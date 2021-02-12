"""
Opens the raster file and the shape file. Saves the raster file as an image.
Creates a mask image from the shape file and saves the masked image as well.
Dr.Ali POLAT(2021)
"""
import rasterio
import rasterio.mask 
import fiona 
import matplotlib.pyplot as plt
from rasterio.plot import show,reshape_as_raster,reshape_as_image
import cv2
import numpy as np
from skimage import io
import os

data_dir="your data directory"

raster_file="your image file"
shp_file="your shape file"
save_image_file="target image file name"
save_mask_file="target mask file name"

dataset=rasterio.open(os.path.join(data_dir,raster_file))

image=dataset.read()
print(image.shape)
image=reshape_as_image(image)
#image=cv2.convertScaleAbs(image)#uint16 kayıt yapmıyor uint8 e çevirmek için

image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
cv2.imwrite(os.path.join(data_dir,save_image_file),image)

