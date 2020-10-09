import numpy as np
import json 
from scipy import misc
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max
from skimagemorphology import watershed
from scipy import ndimage.


class DataLoader:
    def __init__(self,batchSize,inputShape,dataList):
        self.inputShape=inputShape
        self.batchSize=batchSize
        self.dataList=dataList

    def generator(self):
        pass

def dataAugmentation(images,labels):
    newImages=[]
    newLabels=[]
    for i,im in enumerate(images):
        newImages.append(im)
        newLabels.append(labels[i])
        newImages.append(np.flip(im,axis=0))
        newLabels.append(np.flip(labels[i],axis=0))
        newImages.append(np.flip(im,axis=1))
        newLabels.append(np.flip(labels[i],axis=1))
        newImages.append(np.rot90(im,k=1))
        newLabels.append(np.rot90(labels[i],k=1))
        newImages.append(np.rot90(im,k=2))
        newLabels.append(np.rot90(labels[i],k=2))
        newImages.append(np.rot90(im,k=3))
        newLabels.append(np.rot90(labels[i],k=3))
    return np.array(newImages),np.array(newLabels)


