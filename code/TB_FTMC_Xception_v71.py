import numpy as np
from keras.applications.xception import Xception,preprocess_input,decode_predictions
from keras.preprocessing import image as im
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,LearningRateScheduler,ReduceLROnPlateau
from keras.optimizers import Adam 
from AbstractFineTuneModelConfig import AbstractFineTuneModelConfig


#Same as Xceptionv7 but with TensorBoard Logging, refactoring and LearningRateScheduler, and ReduceLROnPlateau
#   frozen till layer 115 in era 1
#Use typical steps_per_epoch/validation_steps cfr. docs & count.py
class TB_FTMC_Xception_v71(AbstractFineTuneModelConfig):
    dataSetVersion=2.0 #Batch2 dataset
    configId="-mXc_v71"
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
    checkpointEnd="_ep{epoch:02d}-vl{val_loss:.4f}.hdf5"


    def __init__(self):
        self.nbOfClasses=6
        self.nbOfEras = 2 
#Start with base model, without last layer.
        self.baseModel = Xception(input_shape=self.input_shape, weights='imagenet', include_top=False)
# add a global spatial average pooling layer
        x = self.baseModel.output
        x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
        x = Dense(256, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
        predictions = Dense(self.nbOfClasses, activation='softmax')(x)

# this is the model we will train
        self.model = Model(inputs=self.baseModel.input, outputs=predictions)

# train the model on the new data for a few epochs
        self.imgDatagen = im.ImageDataGenerator(
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
        nbOfEpochs=50
        modelRunName =  'BECONA'+str(self.dataSetVersion)+self.configId+"_split"+str(cvIndex)+eraName
        checkpointName='/tmp/'+modelRunName
        savename= modelRunName+"_ep{epoch:02d}.hdf5"
##CALLBACKS
#checkpoint callback
        checkpointer = ModelCheckpoint(filepath=checkpointName+self.checkpointEnd, verbose=1, save_best_only=True)
        tbcallback = TensorBoard(log_dir='/tmp/tblogs/'+modelRunName, histogram_freq=0, write_graph=True, write_images=False)
        lr_decay = LearningRateScheduler(schedule=lambda epoch: 0.001 * (0.9 ** epoch)) 
        early_stopping = EarlyStopping(monitor='val_loss', patience=11, mode='auto')
#early stopping too;
#add mode="auto" for it to work?

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
        for layer in self.baseModel.layers:
            layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')

        self.model.fit_generator(
                trainGen,
                steps_per_epoch=self.steps_per_epoch, # *batch size = total training samples
                epochs=nbOfEpochs,
                validation_data=valGen,
                validation_steps=self.validation_steps, #validation steps with generator
                callbacks=[checkpointer,tbcallback,lr_decay,early_stopping]) 

        self.model.save(savename.format(epoch=nbOfEpochs))

        pass

    def _trainEra1(self, trainGen, valGen, cvIndex):
        eraName="_Era1"
        nbOfEpochs=100
        modelRunName =  'BECONA'+str(self.dataSetVersion)+self.configId+"_split"+str(cvIndex)+eraName
        checkpointName='/tmp/'+modelRunName
        savename= modelRunName+"_ep{epoch:03d}.hdf5"
##CALLBACKS
#checkpoint callback
        checkpointer = ModelCheckpoint(filepath=checkpointName+self.checkpointEnd, verbose=1, save_best_only=True)
        tbcallback = TensorBoard(log_dir='/tmp/tblogs/'+modelRunName, histogram_freq=0, write_graph=True, write_images=False)
        reduceOnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode='auto')

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
# we use Adam with a low learning rate
        self.model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
# Same fit_generator?
        self.model.fit_generator(
                trainGen,
                steps_per_epoch=self.steps_per_epoch,
                epochs=nbOfEpochs,
                validation_data=valGen,
                validation_steps=self.validation_steps,
                callbacks=[checkpointer,tbcallback,reduceOnPlateau,early_stopping])

        self.model.save(savename.format(epoch=nbOfEpochs))
        pass
