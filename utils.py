import numpy as np
import json 
from scipy import misc
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import math


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

def createGaussianLabel(imagePath,labelPath,inputShape,imageShape,GaussianSize):
    x, y = np.meshgrid(np.linspace(-1,1,GaussianSize), np.linspace(-1,1,GaussianSize))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 0.5, 0.0
    gua = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )*255
    with open(labelPath) as f:
        label = json.load(f)
    img=np.zeros(shape=inputShape,dtype=np.int)
    guLabel=np.zeros(shape=inputShape,dtype=np.int)
    for d in label:
        x=min(max(int(int(d['y'])*(inputShape[0]/imageShape[0])),0),inputShape[0])
        y=min(max(int(int(d['x'])*(inputShape[1]/imageShape[1])),0),inputShape[1])
        x_,y_=img[max(int(x-math.floor(GaussianSize/2)),0):min(int(x+math.ceil(GaussianSize/2)),255),max(int(y-math.floor(GaussianSize/2)),0):min(int(y+math.ceil(GaussianSize/2)),255),0].shape

        if(int(d['label_id'])==1):
            guLabel[max(int(x-math.floor(GaussianSize/2)),0):min(int(x+math.ceil(GaussianSize/2)),255),max(0,int(y-math.floor(GaussianSize/2))):min(int(y+math.ceil(GaussianSize/2)),255),0]=gua[math.floor(GaussianSize/2)-x_//2:math.floor(GaussianSize/2)+x_//2+x_%2,math.floor(GaussianSize/2)-y_//2:math.floor(GaussianSize/2)+y_//2+y_%2]

        if(int(d['label_id'])==2):
            guLabel[max(int(x-math.floor(GaussianSize/2)),0):min(int(x+math.ceil(GaussianSize/2)),255),max(0,int(y-math.floor(GaussianSize/2))):min(int(y+math.ceil(GaussianSize/2)),255),1]=gua[math.floor(GaussianSize/2)-x_//2:math.floor(GaussianSize/2)+x_//2+x_%2,math.floor(GaussianSize/2)-y_//2:math.floor(GaussianSize/2)+y_//2+y_%2]

        if(int(d['label_id'])==3):
            guLabel[max(int(x-math.floor(GaussianSize/2)),0):min(int(x+math.ceil(GaussianSize/2)),255),max(0,int(y-math.floor(GaussianSize/2))):min(int(y+math.ceil(GaussianSize/2)),255),2]=gua[math.floor(GaussianSize/2)-x_//2:math.floor(GaussianSize/2)+x_//2+x_%2,math.floor(GaussianSize/2)-y_//2:math.floor(GaussianSize/2)+y_//2+y_%2]
        
    return guLabel
