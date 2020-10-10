from keras.callbacks import ModelCheckpoint,Callback,LearningRateScheduler
import random
import models
from utils import DataLoader, stepDecay
from config import Config
import numpy as np
from scipy import misc
import gc
from keras.optimizers import Adam
from imageio import imread
from keras.models import load_model
from datetime import datetime
import os
import json


def train(conf: Config):
    time=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    trainString="%s_%s_%s_%s" % (conf.model,conf.optimizer,str(conf.lr),time)
    os.makedirs(conf.logPath+"/"+trainString)
    with open(conf.logPath+"/"+trainString+'/config.json', 'w') as f:
        json.dump(conf.__dict__, f)
    print('Compiling model...')
    model_checkpoint = ModelCheckpoint(conf.logPath+'/Checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=False)
    change_lr = LearningRateScheduler(stepDecay)
    model.compile(optimizer = Conf.optimizer, loss = conf.loss)
    
    trainDataLoader=DataLoader(conf.batchSize,conf.inputShape,)
    validationDataLoader=DataLoader(conf.batchSize,conf.inputShape,)
    H=model.fit_generator(generator=generator(batch_size),
                    validation_data=generator(batch_size,"validation"),
                    steps_per_epoch=len(train_data)//batch_size,
                    validation_steps=len(validation_data)//batch_size,
                    epochs=30,
                    verbose=1,
                    initial_epoch=0,
                    callbacks = [model_checkpoint, change_lr]
                    )

train(Config())
