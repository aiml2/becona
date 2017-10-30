import numpy as np
#from keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from keras.applications.xception import Xception,preprocess_input,decode_predictions
from keras.preprocessing import image as im
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.optimizers import SGD
from AbstractFineTuneModelConfig import AbstractFineTuneModelConfig


#Same as FTMC_Xception_v4 but with 128 dense last layer 
class FTMC_Xception_v6(AbstractFineTuneModelConfig):
    dataSetVersion=2.0 #Batch2 dataset
    configId="-mXc_v6"
    #dataDirPrefix='split'
    class_mode='categorical'
    #cvIndex=1
    inputWidth = 299
    inputHeight = 299
    target_size=(inputWidth, inputHeight) 
    # if K.image_data_format() == 'channels_first':
    #        input_shape = (3, inputWidth, inputHeight)
    #    else:
    input_shape = (inputWidth, inputHeight, 3)
    batch_size=32
    steps_per_epoch=36 #determine with count
    validation_steps=9
    checkpointEnd="_ep{epoch:02d}-vl{val_loss:.2f}.hdf5"

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')

    def __init__(self):
        self.nbOfClasses=6
        self.nbOfEras = 2 
#Start with base model, without last layer.
        self.baseModel = Xception(input_shape=self.input_shape, weights='imagenet', include_top=False)
# add a global spatial average pooling layer
        x = self.baseModel.output
        x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
# and a logistic layer -- let's say we have 200 classes
        predictions = Dense(self.nbOfClasses, activation='softmax')(x)
# We actually have 3 classes now 1, 2.0 and 2.1
#Idea also learn non-flower images, might help in extracting

# this is the model we will train
        self.model = Model(inputs=self.baseModel.input, outputs=predictions)

# train the model on the new data for a few epochs
        self.imgDatagen = im.ImageDataGenerator(
                #shear_range=0.1, 1.3
                zoom_range=[0.9,1],
                horizontal_flip=True,
                vertical_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                rotation_range=360,
                channel_shift_range=10,
                preprocessing_function=preprocess_input)
        pass

    def trainEra(self, eraInt, dataDirPrefix, cvIndex):
        if(eraInt < self.nbOfEras):
            print("Training era :", eraInt)
            trainDir =dataDirPrefix+str(cvIndex)+'_Train'
            valDir = dataDirPrefix+str(cvIndex)+'_Val'
            trainDatagen = self.imgDatagen.flow_from_directory(
                    trainDir,
                    target_size=self.target_size,
                    batch_size=self.batch_size,
                    class_mode=self.class_mode,
                    follow_links=True)

            assert len(trainDatagen.class_indices) == self.nbOfClasses

            valDatagen = self.imgDatagen.flow_from_directory(
                    valDir,
                    target_size=self.target_size,
                    batch_size=self.batch_size,
                    class_mode=self.class_mode,
                    follow_links=True)

            assert len(valDatagen.class_indices) == self.nbOfClasses
            if(eraInt == 0):
                self._trainEra0(trainDatagen, valDatagen, cvIndex)
            if(eraInt == 1):
                self._trainEra1(trainDatagen, valDatagen, cvIndex)

        else:
            raise Exception("Can't train an unexisting era.")


    def _trainEra0(self, trainGen, valGen, cvIndex):
        eraName="_Era0"
        nbOfEpochs=10
        checkpointName='/tmp/BECONA'+str(self.dataSetVersion)+self.configId+"_split"+str(cvIndex)+eraName
        savename='BECONA'+str(self.dataSetVersion)+self.configId+"_split"+str(cvIndex)+eraName+"_ep{epoch:02d}.hdf5"
##CALLBACKS
#checkpoint callback
        checkpointer = ModelCheckpoint(filepath=checkpointName+self.checkpointEnd, verbose=1, save_best_only=True)
#early stopping too;
#add mode="auto" for it to work?


#model.load_weights('/tmp/w-flowerIV3.hdf5')
#model.load_weights('/tmp/w-fine-flowerIV3.hdf5')
#model.load_weights('/tmp/w-dense256-s-valtraingen-BECONA-4.4.hdf5')
#model.load_weights('BECONA-D256-s-valtraingen-finetuned-10ep-4.4.h5')


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
        for layer in self.baseModel.layers:
            layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        self.model.fit_generator(
                trainGen,
                steps_per_epoch=self.steps_per_epoch, # *batch size = total training samples
                epochs=nbOfEpochs,
                validation_data=valGen,
                validation_steps=self.validation_steps, #validation steps with generator
                callbacks=[checkpointer,self.early_stopping]) 

        self.model.save(savename.format(epoch=nbOfEpochs))

        pass

    def _trainEra1(self, trainGen, valGen, cvIndex):
        eraName="_Era1"
        checkpointName='/tmp/BECONA'+str(self.dataSetVersion)+self.configId+"_split"+str(cvIndex)+eraName
        savename='BECONA'+str(self.dataSetVersion)+self.configId+"_split"+str(cvIndex)+eraName+"_ep{epoch:02d}.hdf5"
##CALLBACKS
#checkpoint callback
        checkpointer = ModelCheckpoint(filepath=checkpointName+self.checkpointEnd, verbose=1, save_best_only=True)
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

        for i, layer in enumerate(self.baseModel.layers):
            print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
        for layer in self.model.layers[:115]:
            layer.trainable = False
        for layer in self.model.layers[116:]:
            layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
        self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
# Same fit_generator?
        self.model.fit_generator(
                trainGen,
                steps_per_epoch=self.steps_per_epoch,
                epochs=15,
                validation_data=valGen,
                validation_steps=self.validation_steps,
                callbacks=[checkpointer,self.early_stopping])

        self.model.save(savename.format(epoch=15))

        self.model.compile(optimizer=SGD(lr=0.00001, momentum=0.9), loss='categorical_crossentropy')
        self.model.fit_generator(
                trainGen,
                steps_per_epoch=self.steps_per_epoch,
                epochs=10,
                validation_data=valGen,
                validation_steps=self.validation_steps,
                callbacks=[checkpointer,self.early_stopping])

        self.model.save(savename.format(epoch=25))

        self.model.compile(optimizer=SGD(lr=0.000001, momentum=0.9), loss='categorical_crossentropy')

        self.model.fit_generator(
                trainGen,
                steps_per_epoch=self.steps_per_epoch,
                epochs=5,
                validation_data=valGen,
                validation_steps=self.validation_steps,
                callbacks=[checkpointer,self.early_stopping])

        self.model.save(savename.format(epoch=30))
        pass
