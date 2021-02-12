""" 
creates patches from RGB image and maskimage
Ali POLAT (2021).
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from skimage import io


def create_image(image_path,save_folder,h,w,set_name="set1"):

    if os.path.isdir(save_folder)==False:
        os.mkdir(save_folder)

    img=cv2.imread(image_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    font=cv2.FONT_HERSHEY_SIMPLEX
    alan=img.shape[0]*img.shape[1]
    

    h=h
    w=w
    img2=np.zeros([h,w,3],dtype=np.uint8)
    sutun=int(img.shape[1]/w)
    satir=int(img.shape[0]/h)
    print("satir sayısı= ",satir,"sütün sayısı= ",sutun)
    x1=0
    y1=0
    x2=w
    y2=h

    
    for j in tqdm(range(satir)):
        print("satir= ",j)

        for i in range(sutun):
            print("Sütun= ",i)
            
        

            ROI=img[y1:y2,x1:x2]
            name=os.path.join(save_folder,"file"+str(j)+"_"+str(i)+".png")
            
            ROI=cv2.cvtColor(ROI,cv2.COLOR_RGB2BGR)
            cv2.imwrite(name,ROI)
            #create grid
            #cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),10)
            x1=x1+w
            x2=x1+w
            
            if x2>img.shape[1]:
                x1=0
                x2=w
                y1=y1+h
                y2=y1+h


    #save grid image
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #cv2.imwrite(os.path.join(save_folder,"grid.jpg"),img)
 
#create_image("C:\\Users\\ap\\Documents\\GitHub\\geospatial_deneme\\dolins\\DATA_ORTO\\image_.png","C:\\Users\\ap\\Documents\\GitHub\\geospatial_deneme\\dolins\\DATA_ORTO\\image_320",320,320,"set1_")

def create_mask(image_path,save_folder,h,w,set_name="set1"):
    if os.path.isdir(save_folder)==False:
        os.mkdir(save_folder)


    img=cv2.imread(image_path)
   

    alan=img.shape[0]*img.shape[1]
    #print(img.shape)

    h=h
    w=w
    img2=np.zeros([h,w,3],dtype=np.uint8)
    sutun=int(img.shape[1]/w)
    satir=int(img.shape[0]/h)
    print("satir sayısı= ",satir,"sütün sayısı= ",sutun)
    x1=0
    y1=0
    x2=w
    y2=h

    
    for j in tqdm(range(satir)):
        print("satir= ",j)
        for i in range(sutun):
            print("Sütun= ",i)
            
        

            ROI=img[y1:y2,x1:x2]
            name=os.path.join(save_folder,"file"+str(j)+"_"+str(i)+".png")
            
            ROI=cv2.cvtColor(ROI,cv2.COLOR_RGB2BGR)
            cv2.imwrite(name,ROI)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),10)
            x1=x1+w
            x2=x1+w
            
            if x2>img.shape[1]:
                x1=0
                x2=w
                y1=y1+h
                y2=y1+h

    """plt.imshow(img)    
    plt.show()  
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(save_folder,"grid.jpg"),img)"""
        
    print("files created")
    


create_mask("C:\\Users\\ap\\Documents\\GitHub\\geospatial_deneme\\dolins\\DATA_ORTO\\mask_.png","C:\\Users\\ap\\Documents\\GitHub\\geospatial_deneme\\dolins\\DATA_ORTO\\mask_320",320,320,"set1_")
