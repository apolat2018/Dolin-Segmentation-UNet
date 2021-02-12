#tamame siyah olan görüntüler silindi, Unbalanced data probleminden kaçınmak için
import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

mask_path="mask images directory"
image_path="RGB image directory"
before=len(os.listdir(mask_path))
x=0
for i in tqdm(os.listdir(mask_path)):
  file_mask=os.path.join(mask_path,i)
  file_image=os.path.join(image_path,i)
  im=cv2.imread(file_mask)
 

  if np.all(im==0):
        os.remove(file_mask)
        os.remove(file_image)
        print(file_image,file_mask)
        """plt.imshow(im)
        plt.show()"""
       
      

after=len(os.listdir(mask_path))
print(after-before," file deleted")