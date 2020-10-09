from keras.callbacks import ModelCheckpoint,Callback,LearningRateScheduler
import random
import models
from utils import DataLoader
from config import Config
import numpy as np
from scipy import misc
import gc
from keras.optimizers import Adam
from imageio import imread
from keras.models import load_model

def step_decay(epoch):
    step = 10
    num =  epoch // step 
    if num % 3 == 0:
        lrate = 1e-3
    elif num % 3 == 1:
        lrate = 1e-4
    else:
        lrate = 1e-5
    print('Learning rate for epoch {} is {}.'.format(epoch+1, lrate))
    return np.float(lrate)

def train(config: Config):
    trainString="" %
    print('Compiling model...')
