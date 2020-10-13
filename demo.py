from pipeline import Pipeline
from config import Config
import os  
import glob
import imageio
import cv2
import argparse
import numpy as np

def get_parser():
    
    parser = argparse.ArgumentParser('demo')
    parser.add_argument('--inputPath', '-i', required=True)
    parser.add_argument('--outputPath', '-o', required=True)
    parser.add_argument('--configPath', '-c', required=True)
    return parser

def visualizer(img,points):
    r=1
    colors=[
            (255,0,0),
            (0,255,0),
            (0,0,255)
            ]
    image=np.copy(img)
    for p in points:
        x,y,c=p[0],p[1],p[2]
        cv2.circle(image, (int(x), int(y)), int(r), colors[int(c)], 2)
    return image


def demo(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    conf=Config()
    conf.load(args.configPath)
    pipeline=Pipeline(conf)
    if os.path.isdir(args.inputPath):  
        data = [args.inputPath+"/"+f for f in os.listdir(args.inputPath) if '.jpg' in f]
        for d in data:
            print(d)
            img=imageio.imread(d)
            pred=pipeline.predict(img)
            output=visualizer(img,pred)
            imageio.imwrite(args.outputPath+d.split("/")[-1],output)

    else:
        img=imageio.imread(args.inputPath)
        imageio.imwrite(args.outputPath+args.inutPath("/")[-1])



if __name__ == "__main__":
   demo()