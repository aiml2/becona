#!/usr/bin/env python3
#TODO CHANGE this file to load each model and continue training with Adam for several epochs with declining LR
import time
import sys
import os
import numpy as np
import re 
sys.path.append("../../code/utils")
import utils
import os
from keras.models import Model,load_model
from keras.applications.xception import Xception,preprocess_input,decode_predictions
from keras.preprocessing import image as im
from keras.optimizers import Nadam
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,LearningRateScheduler,ReduceLROnPlateau

modelsDir = 'input/'
outputdir = 'output/'
#models = ["TB_FTMC_Xception_v71"]
dataDirPrefix = "../../B2split_"
#eras = [0,1] 
crossValSet = range(5) 
batch_size=32
steps_per_epoch=36 #determine with count
validation_steps=9
inputWidth = 299
inputHeight = 299
target_size=(inputWidth, inputHeight) 
class_mode='categorical'
# Data Augmentation code 
imgDatagen = im.ImageDataGenerator(
        #shear_range=0.1, 1.3
        zoom_range=[0.9,1],
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=360,
        channel_shift_range=10,
        preprocessing_function=preprocess_input)
checkpointEnd="_ep{epoch:02d}-vl{val_loss:.6f}.hdf5"
nbOfEpochs=100

#Continue with Nadam as optimizer
for cvIndex in crossValSet: 
    for modelFN in os.listdir(modelsDir):
        if 'split'+str(cvIndex) in modelFN:
            model = load_model(modelsDir+modelFN)
            print(modelFN)
            #lrvar =model.optimizer.lr
            #newlr = K.eval(lrvar)
            #print(str(lrvar) + ' == ' + str(K.eval(lrvar)))
            print("########## TRAINING model:" + str(modelFN) + ", cvindex" + str(cvIndex))
            eraName="_Era4"
            newName = modelFN.split('_Era')[0] + eraName
            checkpointName='/tmp/'+newName
            savename= outputdir+newName+"_ep{epoch:02d}.hdf5"
            trainDir =dataDirPrefix+str(cvIndex)+'_Train'
            valDir = dataDirPrefix+str(cvIndex)+'_Val'
            #Make data generators
            trainGen = imgDatagen.flow_from_directory(
                        trainDir,
                        target_size=target_size,
                        batch_size=batch_size,
                        class_mode=class_mode,
                        follow_links=True)

            valGen = imgDatagen.flow_from_directory(
                    valDir,
                    target_size=target_size,
                    batch_size=batch_size,
                    class_mode=class_mode,
                    follow_links=True)

            #Start training this era
    ##CALLBACKS
    #checkpoint callback
            checkpointer = ModelCheckpoint(filepath=checkpointName+checkpointEnd, verbose=1, save_best_only=True)
            tbcallback = TensorBoard(log_dir='/tmp/tblogs/'+newName, histogram_freq=0, write_graph=True, write_images=False)
            #Nadam does LR decay itself
            #lr_decay = LearningRateScheduler(schedule=lambda epoch: newlr * (0.9 ** epoch)) 
            early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto')

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
            for layer in model.layers[:115]:
                layer.trainable = False
            for layer in model.layers[116:]:
                layer.trainable = True

            model.compile(optimizer=Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004), loss='categorical_crossentropy')
    # we use NAdam with a default learning rate. 
            model.fit_generator(
                    trainGen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=nbOfEpochs, 
                    validation_data=valGen,
                    validation_steps=validation_steps,
                    callbacks=[checkpointer,tbcallback,early_stopping])
