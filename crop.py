import matplotlib.pyplot as plt
import math
import json
import os
import argparse
from rich.progress import track
import logging
logging.basicConfig(format='[%(levelname)s] %(asctime)s --> %(message)s',
                    datefmt='%m/%d/%Y-%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description='Crop the image')
    parser.add_argument('--inputImagePath', '-i', type=str, help='Input image path',required=True)
    parser.add_argument('--inputJsonPath', '-ij', type=str, help='Input json path',required=False,default=None)
    parser.add_argument('--outputImagePath', '-o', type=str, help='Where to save the output image',required=False,default=os.path.join(os.getcwd(),'output'))
    parser.add_argument('--outputJsonPath', '-oj', type=str, help='Where to save the output json',required=False,default=os.path.join(os.getcwd(),'output'))
    parser.add_argument('--cropSize', '-cs', type=tuple, help='Crop size as (width, height)',required=False,default=(256,256))
    args = parser.parse_args()
    return args

class Crop:
    def __init__(self,path,jpath,crop_size ,output_dir ,outputj_dir):
        logger.info('module Initialized.')
        self.path = path
        self.jpath = jpath
        self.crop_size = crop_size
        self.output_dir = output_dir
        self.outputj_dir = outputj_dir
        self.recnames = [x for x in os.listdir(self.path) if x.endswith('.jpg')]
        logger.info(f'founded {len(self.recnames)} image(s).')
        self.jrecnames = [x for x in os.listdir(self.jpath) if x.endswith('.json')]
        if self.jrecnames == []:
            logger.warning('No json file found.\n Ignoring all.')
            self.no_json = True
        else:
            logger.info(f'founded {len(self.jrecnames)} json(s).')
            self.no_json = False
        self.crop()

    def crop(self):
        if not os.path.exists(self.output_dir):
            logger.info(f'Creating output directory {self.output_dir}')
            os.makedirs(self.output_dir)
        else:
            logger.info(f'Output directory {self.output_dir} already exists.')
        if not self.no_json:
            if self.jpath is not None:   
                if not os.path.exists(self.outputj_dir):
                    logger.info(f'Creating output directory {self.outputj_dir}')
                    os.makedirs(self.outputj_dir)
                else:
                    logger.info(f'Output directory {self.outputj_dir} already exists.')
        else:
            logger.warning('Ignoring json output folder.')

        for rec_name in track(self.recnames, description='Cropping images'):
            jrec_name = f'{rec_name[:-3]}json'
            self.rec_name = rec_name
            self.jrec_name = jrec_name
            im = plt.imread(os.path.join(self.path,rec_name))
            if jrec_name in self.jrecnames:
                js = self.read_json(os.path.join(self.jpath,jrec_name))
            else:
                js=None
            self.crop_single_image(im,js)
    
    def select_cells(self,js,x_start,y_start,x_c,y_c):
        selected = [cell for cell in js if (( x_start+x_c >cell['x'] >= x_start ) and (y_start+y_c >cell['y'] >= y_start))]
        if selected == []:
            empty = True
        else :
            empty = False

        for cell in selected:
            cell['x'] = cell['x'] - x_start
            cell['y'] = cell['y'] - y_start
        return selected,empty

    def read_json(self, path):
        with open(path, 'r') as f:
            js = json.load(f)
        return js

    def crop_single_image(self,im,js):
        X,Y,_ = im.shape
        x_c, y_c = self.crop_size

        # Number of crops in x and y direction
        num_x = math.ceil(X/x_c) + 1
        num_y = math.ceil(Y/y_c) + 1

        # Number of pixels to be cropped in x and y direction
        r_x = (num_x*x_c) - X
        r_y = (num_y*y_c) - Y

        # Number of Overlapping pixels in x and y direction
        o_x = math.floor(r_x/num_x+1)
        o_y = math.floor(r_y/num_y+1)


        for i in range(num_x):
            for j in range(num_y):
                if i == 0:
                    x_start = 0
                elif i == num_x-1:
                    x_start = X - x_c
                else:
                    x_start = (i*x_c) - (i*o_x)
                if j == 0:
                    y_start = 0
                elif j == num_y-1:
                    y_start = Y - y_c
                else:
                    y_start = (j*y_c) - (j*o_y)

                #crop and save json
                if not self.no_json:
                    selectedCells,empty = self.select_cells(js, x_start, y_start, x_c, y_c)
                else:
                    empty = False
                if not empty:
                    if not self.no_json:
                        json.dump(selectedCells, open(os.path.join(self.outputj_dir,f'crop_{i}_{j}_{self.jrec_name}'), 'w'))
                    # crop and save image
                    crop_im = im[x_start:x_start+x_c, y_start:y_start+y_c, :]
                    crop_dir = os.path.join(self.output_dir, f'crop_{i}_{j}_{self.rec_name}')
                    plt.imsave(crop_dir, crop_im)
                else:
                    logger.warning(f'No cell in crop {i} {j} \n Ignoring crop.')


                


if __name__ == '__main__':
    args = get_args()
    crop = Crop(args.inputImagePath,args.inputJsonPath,args.cropSize,args.outputImagePath,args.outputJsonPath)
