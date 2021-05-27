#tamame siyah olan görüntüler silindi, Unbalanced data probleminden kaçınmak için

import numpy as np
import os
import cv2
from tqdm import tqdm

mask_path="D:\\rocksegmentation\\mask"
image_path="D:\\rocksegmentation\\image"
print(mask_path)
before=len(os.listdir(mask_path))

for i in tqdm(os.listdir(mask_path)):
  file_mask=os.path.join(mask_path,i)
  file_image=os.path.join(image_path,i)
  im=cv2.imread(file_mask)
  toplam=im.shape[0]*im.shape[1]
  a=np.count_nonzero(im)
  oran=(a/toplam)*100
 
  if oran<1:
    os.remove(file_mask)
    os.remove(file_image)
    #print(file_image,file_mask)

after=len(os.listdir(mask_path))
print(after-before," file deleted")