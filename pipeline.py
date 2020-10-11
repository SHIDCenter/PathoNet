import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed as ws
import cv2
import models
from  config import Config
import imageio
import argparse


def watershed(pred):
    cells=[]
    for ch in range(3):
        gray=pred[:,:,ch]
        D = ndimage.distance_transform_edt(gray)
        localMax = peak_local_max(D, indices=False, min_distance=5,exclude_border=False,labels=gray)
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

def predict(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    conf=Config()
    model=models.modelCreator(conf.model,conf.inputShape,conf.classes,weights=conf.pretrainedModel)
    img=imageio.imread(args.imagePath)
    img=(img/255.).astype(np.float32)
    img=np.expand_dims(img,0)
    pred=model.predict(img)
    np.place(pred[:,:,0],pred[:,:,0]<conf.thresholds[0],0)
    np.place(pred[:,:,1],pred[:,:,1]<conf.thresholds[1],0)
    np.place(pred[:,:,2],pred[:,:,2]<conf.thresholds[2],0)
    np.place(pred,pred>0,255)
    pred=np.squeeze(pred)
    cells=watershed(pred)
    np.save(args.savePath,cells)

def get_parser():
    parser = argparse.ArgumentParser('predict')
    parser.add_argument('--imagePath', '-img', required=True)
    parser.add_argument('--savePath', '-s', required=True)
    return parser


if __name__ == "__main__":
   predict()