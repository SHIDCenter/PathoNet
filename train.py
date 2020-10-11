from keras.callbacks import ModelCheckpoint,Callback,LearningRateScheduler,TensorBoard
import random
import models
from utils import DataLoader, LrPolicy
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

def train():
    conf=Config()
    time=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    trainString="%s_%s_%s_%s" % (conf.model,conf.optimizer,str(conf.lr),time)
    os.makedirs(conf.logPath+"/"+trainString)
    with open(conf.logPath+"/"+trainString+'/config.json', 'w') as f:
        json.dump(conf.__dict__, f)
    print('Compiling model...')
    model_checkpoint = ModelCheckpoint(conf.logPath+"/"+trainString+'/Checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=False)
    change_lr = LearningRateScheduler(LrPolicy(Config.lr).stepDecay)
    tbCallBack=TensorBoard(log_dir=conf.logPath+"/"+trainString+'/logs', histogram_freq=0,  write_graph=True, write_images=True)
    model=models.modelCreator(conf.model,conf.inputShape,conf.classes,conf.pretrainedModel)
    model.compile(optimizer = conf.optimizer, loss = conf.loss)
    data = [conf.logPath+"/"+trainString+"/"+f for f in os.listdir(conf.trainDataPath)]
    random.shuffle(data)
    thr=int(len(data)*conf.validationSplit)
    trainData=data[thr:]
    valData=data[:thr]
    trainDataLoader=DataLoader(conf.batchSize,conf.inputShape,trainData)
    validationDataLoader=DataLoader(conf.batchSize,conf.inputShape,valData)
    print('Fitting model...')
    model.fit_generator(generator=generator(batch_size),
                    validation_data=generator(batch_size,"validation"),
                    steps_per_epoch=len(train_data)//batch_size,
                    validation_steps=len(validation_data)//batch_size,
                    epochs=conf.epoches,
                    verbose=1,
                    initial_epoch=0,
                    callbacks = [model_checkpoint, change_lr,tbCallBack]
                    )

if __name__ == "__main__":
   train()