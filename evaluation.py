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
from sklearn.metrics import accuracy_score

def read_labels(name,inputShape,imageShape):
    with open(name,'r') as f:
        temp = json.load(f)
        labels=[]
        for d in temp:
            print(imageShape)
            if imageShape[0] ==255 and imageShape[1]==255:
                x=int(d['x'])
                y=int(d['y'])
            else:
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
        labels=read_labels(d.replace(".jpg",".json"),conf.inputShape,img.shape).reshape((-1,3))
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

# Function to compute the Root Mean Square Error (RMSE)
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# Function to compute RMSE and cutoff accuracy for Ki67-score and TIL-score
def eval_pts(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    conf=Config()
    conf.load(args.configPath)
    pipeline=Pipeline(conf) 
    data = [args.inputPath+"/"+f for f in os.listdir(args.inputPath) if '.jpg' in f]
    
    # Group data paths by patients
    grouped_data = {}
    for im in data:
        key = im.partition('_')[0]
        grouped_data.setdefault(key, []).append(im)
    grouped_data = list(grouped_data.values())
    
    #Compute RMSE for each patient
    pred_ki67_pt=[] #collect 23 scores for prediction
    gt_ki67_pt=[] #collect 23 scores for ground truth
    pred_TIL_pt=[] #collect 23 scores for prediction
    gt_TIL_pt=[] #collect 23 scores for ground truth

    for pt_idx, pt_path in enumerate(grouped_data):
        res=np.zeros((len(grouped_data[pt_idx]),3,3))  #img x classes x TP-FP-FN
        lb=[] #collect all labels for one patient
        for i,d in enumerate(grouped_data[pt_idx]): #Iterate across images for one pt
            img=imageio.imread(d)
            labels=read_labels(d.replace(".jpg",".json"),conf.inputShape,conf.imageShape).reshape((-1,3)) #x-y-label
            lb.append(labels)
            img=misc.imresize(img,conf.inputShape) #resize 256 x 256
            pred=pipeline.predict(img) #x-y-label
            if len(pred!=0):
                for j,ch in enumerate(range(3)):
                    label=labels[np.argwhere(labels[:,2]==j)].reshape((-1,3))[:,:2].T #x-y of label equal to j
                    res[i,j,:]=metric(pred[np.argwhere(pred[:,2]==j)].reshape((-1,3))[:,:2], label)   
        #Compute GT ki67-score for one patient
        labels_pt=np.concatenate(lb)
        gt_ki67=np.sum(labels_pt[:,2]==0)/(np.sum(labels_pt[:,2]==0)+np.sum(labels_pt[:,2]==1))
        #Compute GT TIL-score for one patient
        gt_TIL=np.sum(labels_pt[:,2]==2)/(np.sum(labels_pt[:,2]==2)+np.sum(labels_pt[:,2]==0)+np.sum(labels_pt[:,2]==1))
        #Compute predicted ki67-score for one patient (TP0+FP0)/(TP0+FP0+TP1+FP1) --> #Ki67pos/(#Ki67pos+#Ki67neg)
        res_sum=res.sum(axis=0)
        pred_ki67=(res_sum[0,0]+res_sum[0,1])/(res_sum[0,0]+res_sum[0,1]+res_sum[1,0]+res_sum[1,1]) 
        #Compute predicted TIL-score for one patient TP2+FP2/(TP2+FP2+TP0+FP0+TP1+FP1) --> #TIL/(#TIL+#Ki67pos+#Ki67neg)
        pred_TIL=(res_sum[2,0]+res_sum[2,1])/(res_sum[2,0]+res_sum[2,1]+res_sum[0,0]+res_sum[0,1]+res_sum[1,0]+res_sum[1,1])
        #Collect results of all patients
        pred_ki67_pt.append(pred_ki67)
        gt_ki67_pt.append(gt_ki67)
        pred_TIL_pt.append(pred_TIL)
        gt_TIL_pt.append(gt_TIL)
        
    #Compute rmse over patients
    rmse_ki67=rmse(np.array(pred_ki67_pt), np.array(gt_ki67_pt))
    rmse_TIL=rmse(np.array(pred_TIL_pt), np.array(gt_TIL_pt))
        
    #Compute cut-offs accuracies
    thr1=0.16
    thr2=0.30
    pred_ki67_cutoff=[0 if v < thr1 else 2 if v > thr2 else 1 for v in pred_ki67_pt]
    gt_ki67_cutoff=[0 if v < thr1 else 2 if v > thr2 else 1 for v in gt_ki67_pt]
        
    thr3=0.10
    thr4=0.60 #In the paper is wrong (taken from reference PMID: 29233559)
    pred_TIL_cutoff=[0 if v <= thr3 else 2 if v >= thr4 else 1 for v in pred_TIL_pt]
    gt_TIL_cutoff=[0 if v <= thr3 else 2 if v >= thr4 else 1 for v in gt_TIL_pt]

    acc_ki67_pt=accuracy_score(gt_ki67_cutoff, pred_ki67_cutoff)
    acc_TIL_pt=accuracy_score(gt_TIL_cutoff, pred_TIL_cutoff)
    
    print("Ki-67 index (RMSE) is:", rmse_ki67)
    print("TILs score (RMSE) is:", rmse_TIL)
    print("Ki-67 cut-off accuracy is:", acc_ki67_pt)
    print("TILs cut-off accuracy is:", acc_TIL_pt)
    
    return rmse_ki67, rmse_TIL, acc_ki67_pt, acc_TIL_pt


if __name__ == "__main__":
   eval()
   eval_pts()



