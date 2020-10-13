from pipeline import Pipeline
from config import Config
import os  
import glob
import imageio
import cv2
import numpy as np
import argparse
import json
from scipy import misc
import matplotlib.pyplot as plt
from tabulate import tabulate

def read_labels(name,inputShape,imageShape):
    with open(name,'r') as f:
        temp = json.load(f)
        labels=[]
        for d in temp:
            x=min(max(int(int(d['x'])*(inputShape[0]/imageShape[0])),0),inputShape[0])
            y=min(max(int(int(d['y'])*(inputShape[1]/imageShape[1])),0),inputShape[1])
            c=int(d['label_id'])-1
            labels.append([x,y,c])
        labels=np.array(labels)
    return labels


def get_parser():
    
    parser = argparse.ArgumentParser('eval')
    parser.add_argument('--inputPath', '-i', required=True)
    parser.add_argument('--configPath', '-c', required=True)
    return parser

def metric(pred,label):
    distance_thr=20
    a = np.repeat(pred[...,np.newaxis], label.shape[-1], axis=2)
    b =label.reshape((1,label.shape[0],label.shape[1]))
    b= np.repeat(b,pred.shape[0],axis=0)
    c=np.subtract(a,b)
    d=np.sqrt(c[:,0,:]**2+c[:,1,:]**2)
    d=np.concatenate(((np.ones(label.shape[-1])*distance_thr)[np.newaxis,...],d),axis=0)
    e=np.argmin(d,axis=0)
    TP=np.unique(np.delete(e,np.argwhere(e==0))).shape[0]
    FP=pred.shape[0]-TP
    FN=label.shape[-1]-TP
    return [TP,FP,FN]



def eval(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    conf=Config()
    conf.load(args.configPath)
    pipeline=Pipeline(conf) 
    data = [args.inputPath+"/"+f for f in os.listdir(args.inputPath) if '.jpg' in f]
    res=np.zeros((len(data),3,3))
    for i,d in enumerate(data):
        img=imageio.imread(d)
        labels=read_labels(d.replace(".jpg",".json"),conf.inputShape,conf.imageShape).reshape((-1,3))
        img=misc.imresize(img,conf.inputShape)
        pred=pipeline.predict(img)
        if len(pred!=0):
            for j,ch in enumerate(range(3)):
                label=labels[np.argwhere(labels[:,2]==j)].reshape((-1,3))[:,:2].T
                res[i,j,:]=metric(pred[np.argwhere(pred[:,2]==j)].reshape((-1,3))[:,:2],label)

    pre=np.sum(res[...,0],axis=0)/(np.sum(res[...,0],axis=0)+np.sum(res[...,1],axis=0)+0.00000001)
    rec=np.sum(res[...,0],axis=0)/(np.sum(res[...,0],axis=0)+np.sum(res[...,2],axis=0)+0.00000001)
    F1=2*(pre*rec)/(pre+rec+0.00000001)
    print(tabulate([['Immunopositive', pre[0],rec[0],F1[0]], ['Immunonegative',pre[1],rec[1],F1[1] ],['Lymphocyte',pre[2],rec[2],F1[2]]], headers=['Class', 'Prec.','Rec.','F1']))


if __name__ == "__main__":
   eval()



