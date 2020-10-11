import numpy as np
import json 
import math
from imageio import imread


class DataLoader:
    def __init__(self,batchSize,inputShape,dataList,guaMaxValue):
        self.inputShape=inputShape
        self.batchSize=batchSize
        self.dataList=dataList
        self.guaMaxValue=guaMaxValue
    def generator(self):
        while(1):
            batch=np.random.choice(self.dataList,size=self.batchSize,replace=False)
            images=[]
            labels=[]
            for b in batch:
                img=imread(b)
                images.append(img)
                temp=np.load(b.replace(".jpg",".npy")).astype(int)
                np.place(temp,temp==255,self.guaMaxValue)
                labels.append(temp)
            images=np.array(images)
            yield (np.array(images)/255.).astype(np.float32),np.array(labels)

def dataAugmentation(images,labels):
    newImages=[]
    newLabels=[]
    for i,img in enumerate(images):
        newImages.append(img)
        newLabels.append(labels[i])
        newImages.append(np.flip(img,axis=0))
        newLabels.append(np.flip(labels[i],axis=0))
        newImages.append(np.flip(img,axis=1))
        newLabels.append(np.flip(labels[i],axis=1))
        newImages.append(np.rot90(img,k=1))
        newLabels.append(np.rot90(labels[i],k=1))
        newImages.append(np.rot90(img,k=2))
        newLabels.append(np.rot90(labels[i],k=2))
        newImages.append(np.rot90(img,k=3))
        newLabels.append(np.rot90(labels[i],k=3))
    return np.array(newImages),np.array(newLabels)

def createGaussianLabel(labelPath,inputShape,imageShape,GaussianSize):
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

class LrPolicy:
    def __init__(self,lr):
        self.lr=lr
    def stepDecay(self,epoch):
        step = 10
        num =  epoch // step 
        if epoch>=30:
            lrate = self.lr/1000
        elif num % 3 == 0:
            lrate = self.lr
        elif num % 3 == 1:
            lrate = self.lr/100
        else :
            lrate = self.lr/1000
        print('Learning rate for epoch {} is {}.'.format(epoch+1, lrate))
        return np.float(lrate)

