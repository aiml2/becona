import numpy as np
from keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from keras.preprocessing import image as im
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.optimizers import SGD


dataSetVersion=0.1 #For non cropped
modelVersion="-mIV3-0.4"
dataDirPrefix='split'
#cvIndex=1
trainDir =dataDirPrefix+str(cvIndex)+'_Train'
valDir = dataDirPrefix+str(cvIndex)+'_Val'
target_size=(299, 299) 
batch_size=32
nbOfClasses=3
checkpointEnd="_ep{epoch:02d}-vl{val_loss:.2f}.hdf5"
checkpointName='/tmp/BECONA'+str(dataSetVersion)+modelVersion+"_split"+str(cvIndex)
savename='BECONA'+str(dataSetVersion)+modelVersion+"_split"+str(cvIndex)+"_ep{epoch:02d}-{trmode}.hdf5"


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
#can also be done using pooling="max" in base inceptionv3 model??
#If i use that maybe have to enable it to be trainable then?
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
# was 1024 for 200 classes
# first run tried 512
#then 1024 was horrible (also 1./255 and preprocess_input)
#success with 256, most self-supplied pictures recognized with high accuracy
#bretty good results with 128, lower accuracy on some samples though
x = Dense(256, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
#predictions = Dense(200, activation='softmax')(x)
predictions = Dense(nbOfClasses, activation='softmax')(x)
# We actually have 3 classes now 1, 2.0 and 2.1
#Idea also learn non-flower images, might help in extracting

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)



# train the model on the new data for a few epochs
train_datagen = im.ImageDataGenerator(
        #shear_range=0.1, 1.3
        zoom_range=[0.8,1],
        horizontal_flip=True,
        vertical_flip=True,
	width_shift_range=0.1,
	height_shift_range=0.2,
	rotation_range=360,
	channel_shift_range=0.4,
	preprocessing_function=preprocess_input)

val_datagen = im.ImageDataGenerator(
        #shear_range=0.1, 1.3
        zoom_range=[0.8,1],
        horizontal_flip=True,
        vertical_flip=True,
	width_shift_range=0.1,
	height_shift_range=0.2,
	rotation_range=360,
	channel_shift_range=0.4,
	preprocessing_function=preprocess_input) 

train_generator = train_datagen.flow_from_directory(
        trainDir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
	follow_links=True)


validation_generator = val_datagen.flow_from_directory(
        valDir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
	follow_links=True)

print(train_generator.class_indices)

print(validation_generator.class_indices)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')

def trainCoarse():
    trmode="coarse"
##CALLBACKS
#checkpoint callback
    checkpointer = ModelCheckpoint(filepath=checkpointName+trmode+checkpointEnd, verbose=1, save_best_only=True)
#early stopping too;
#add mode="auto" for it to work?


#model.load_weights('/tmp/w-flowerIV3.hdf5')
#model.load_weights('/tmp/w-fine-flowerIV3.hdf5')
#model.load_weights('/tmp/w-dense256-s-valtraingen-BECONA-4.4.hdf5')
#model.load_weights('BECONA-D256-s-valtraingen-finetuned-10ep-4.4.h5')


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    model.fit_generator(
            train_generator,
            steps_per_epoch=19, # *batch size = total training samples
            epochs=10,
            validation_data=validation_generator,
            validation_steps=5, #validation steps with generator
            callbacks=[checkpointer,early_stopping]) 

    model.save(savename.format(epoch=10,trmode="coarse"))


def trainFine():
    trmode="fine"
##CALLBACKS
#checkpoint callback
    checkpointer = ModelCheckpoint(filepath=checkpointName+trmode+checkpointEnd, verbose=1, save_best_only=True)
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
# Same fit_generator?
    model.fit_generator(
            train_generator,
            steps_per_epoch=40,
            epochs=10,
            #epochs=10,
            #        epochs=7,
            validation_data=validation_generator,
            validation_steps=20,
            callbacks=[checkpointer,early_stopping]) #validation steps with generator

    model.save(savename.format(epoch=10,trmode="finetuned"))

    model.fit_generator(
            train_generator,
            steps_per_epoch=92,
            epochs=10,
            #       epochs=7,
            validation_data=validation_generator,
            validation_steps=24,
            callbacks=[checkpointer,early_stopping]) #validation steps with generator

    model.save(savename.format(epoch=20,trmode="finetuned"))

    model.fit_generator(
            train_generator,
            steps_per_epoch=92,
            epochs=20,
            #       epochs=7,
            validation_data=validation_generator,
            validation_steps=24,
            callbacks=[checkpointer,early_stopping]) #validation steps with generator

    model.save(savename.format(epoch=40,trmode="finetuned"))

