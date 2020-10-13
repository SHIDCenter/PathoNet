import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed as ws
import cv2
import models
from  config import Config
import imageio
import argparse
from config import Config

class Pipeline:

    def __init__(self,conf:  Config ):
        self.modelName=conf.model
        self.weights=conf.pretrainedModel
        self.classes=conf.classes
        self.minDistance=conf.minDistance
        self.inputShape=conf.inputShape
        self.thresholds=conf.thresholds
        self.model=models.modelCreator(self.modelName,self.inputShape,self.classes,weights=self.weights)
    def watershed(self,pred):
        cells=[]
        for ch in range(3):
            gray=pred[:,:,ch]
            D = ndimage.distance_transform_edt(gray)
            localMax = peak_local_max(D, indices=False, min_distance=self.minDistance,exclude_border=False,labels=gray)
            markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
            labels = ws(-D, markers, mask=gray)
            for label in np.unique(labels):
                if label == 0:
                    continue
                mask = np.zeros(gray.shape, dtype="uint8")
                mask[labels == label] = 255
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
                c = max(cnts, key=cv2.contourArea)
                ((x, y), _) = cv2.minEnclosingCircle(c)
                cells.append([x,y,ch])
        return np.array(cells)

    def predict(self,img):
        img=img/255.
        img=np.expand_dims(img,0)
        pred=self.model.predict(img)
        pred=np.squeeze(pred)
        np.place(pred[:,:,0],pred[:,:,0]<self.thresholds[0],0)
        np.place(pred[:,:,1],pred[:,:,1]<self.thresholds[1],0)
        np.place(pred[:,:,2],pred[:,:,2]<self.thresholds[2],0)
        np.place(pred,pred>0,255)
        pred=np.squeeze(pred).astype(np.uint8)
        cells=self.watershed(pred)
        return cells
