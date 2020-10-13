import argparse
import glob
from scipy import misc
from utils import dataAugmentation,createGaussianLabel
import numpy as np

def get_parser():
    
    parser = argparse.ArgumentParser('preprocess')
    parser.add_argument('--inputPath', '-i', required=True)
    parser.add_argument('--outputPath', '-o', required=True)
    parser.add_argument('--outputsize','-s', type=sizes,default=(256,256,3))
    parser.add_argument("--augmentation", '-a',  type=str2bool, nargs='?',const=True, default=False)
    parser.add_argument("--GaussianSize", '-g',  type=int, default=9)
    return parser


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def sizes(s):
    try:
        x, y, c = map(int, s.split(','))
        return (x, y, c)
    except:
        raise argparse.ArgumentTypeError("size must be x,y,c")

def preprocess(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    jpgFiles = glob.glob(args.inputPath+'\\'+'*.jpg') 
    for f in  jpgFiles:
        image=misc.imread(f)
        img=misc.imresize(image,args.outputsize)
        label=createGaussianLabel(f.replace(".jpg",".json"),args.outputsize,image.shape,args.GaussianSize)
        if args.augmentation:
            images,labels=dataAugmentation([img],[label])
            for i in range(len(images)):
                name=args.outputPath+'\\'+(f.replace(".jpg","").split('\\')[-1])+"_"+str(i)
                misc.imsave(name+'.jpg',images[i]) 
                np.save(name+'.npy',labels[i].astype(np.uint8))
        else:
            name=args.outputPath+(f.replace(".jpg","").split('\\')[-1])
            misc.imsave(name+'.jpg',img) 
            np.save(name+'.npy',label)

if __name__ == "__main__":
   preprocess()