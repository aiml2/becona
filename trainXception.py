import numpy as np
from keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from keras.preprocessing import image as im
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping

#Define tests to run
def runTests():
	testPaths = ['FarCenter.jpg','NewRedBot.jpg','NewRedTop.jpg','OldRedBot.jpg','OldRedTop.jpg','OldRusty.jpg','SpanKlemHand1.jpg','SpanKlemHand2.jpg','SpanKlemTwoColor.jpg']
	for img_name in testPaths:
		img = im.load_img("prediction_testing_HQ/"+img_name, target_size=(299,299))
		x = im.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		preds = model.predict(x)
		print('Prediction for "'+ img_name + '" :' , preds)
		#print('Prediction for "'+ img_name + '" :' , decode_predictions(preds, top=5)[0])
	return

# create the base pre-trained model
base_model = xception(weights='imagenet', include_top=False)

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
predictions = Dense(3, activation='softmax')(x)
# We actually have 3 classes now 1, 2.0 and 2.1
#Idea also learn non-flower images, might help in extracting

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

#runTests()

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
        'split3_Train',
        target_size=(299, 299), #def size of Inception is 299x299
        batch_size=32,
        class_mode='categorical',
	follow_links=True)

print(train_generator.class_indices)

validation_generator = val_datagen.flow_from_directory(
        'split3_Val',
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical',
	follow_links=True)

print(validation_generator.class_indices)

##CALLBACKS
#checkpoint callback
checkpointer = ModelCheckpoint(filepath='/tmp/w-dense256-s-valtraingen-BECONA-3.4.hdf5', verbose=1, save_best_only=True)
#early stopping too;
#add mode="auto" for it to work?
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')


model.fit_generator(
        train_generator,
        steps_per_epoch=19,
        epochs=1,
        validation_data=validation_generator,
        #validation_steps=1, #validation steps with preprocessingo only generator
        validation_steps=5, #validation steps with actual generator 
	callbacks=[checkpointer,early_stopping],
	verbose=1) 


#model.load_weights('/tmp/w-flowerIV3.hdf5')
#model.load_weights('/tmp/w-fine-flowerIV3.hdf5')
#model.load_weights('/tmp/w-dense256-s-valtraingen-BECONA-3.4.hdf5')
#model.load_weights('BECONA-D256-s-valtraingen-finetuned-10ep-3.4.h5')

runTests()

model.fit_generator(
        train_generator,
        steps_per_epoch=19, # *batch size = total training samples
        epochs=9,
        validation_data=validation_generator,
        validation_steps=5, #validation steps with generator
	callbacks=[checkpointer,early_stopping]) 

model.save('BECONA-D256-s-valtraingen-coarse-10ep-3.4.h5')

runTests()

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
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
# Same fit_generator?
model.fit_generator(
        train_generator,
        steps_per_epoch=92,
        epochs=4,
        #epochs=10,
#        epochs=7,
        validation_data=validation_generator,
        validation_steps=24,
	callbacks=[checkpointer,early_stopping]) #validation steps with generator

model.save('BECONA-D256-s-valtraingen-finetuned-10ep-3.4.h5')

runTests()

model.fit_generator(
        train_generator,
        steps_per_epoch=92,
        epochs=10,
#       epochs=7,
        validation_data=validation_generator,
        validation_steps=24,
	callbacks=[checkpointer,early_stopping]) #validation steps with generator

model.save('BECONA-D256-s-valtraingen-finetuned-20ep-3.4.h5')

runTests()
