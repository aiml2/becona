import sys
import numpy as np
from keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from keras.preprocessing import image as im
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping

directory = sys.argv[1]

# train the model on the new data for a few epochs
train_datagen = im.ImageDataGenerator(
#        shear_range=0.1,
        zoom_range=[0.9,1],
        horizontal_flip=True,
        vertical_flip=True,
	width_shift_range=0.2,
	height_shift_range=0.2,
	rotation_range=360,
	channel_shift_range=10,
	preprocessing_function=preprocess_input)

val_datagen = im.ImageDataGenerator(
        shear_range=0.2,
        zoom_range=[0.8,1],
        horizontal_flip=True,
        vertical_flip=True,
	width_shift_range=0.3,
	height_shift_range=0.2,
	rotation_range=360,
	#zca_whitening=True,
	channel_shift_range=10,
	preprocessing_function=preprocess_input) 

train_generator = train_datagen.flow_from_directory(
        directory,
        target_size=(299, 299), #def size of Inception is 299x299
        batch_size=32,
        class_mode='categorical',
	follow_links=True,
	save_to_dir='augtrain',
	save_prefix='aug_')

print(train_generator.class_indices)

validation_generator = val_datagen.flow_from_directory(
        directory,
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical',
	follow_links=True,
	save_to_dir='augval',
	save_prefix='aug_')

print(validation_generator.class_indices)

i = 0
iters = 2

for batch in train_generator:
	i += 1
	if i > iters: # save 20 images
		break  # otherwise the generator would loop indefinitely
i = 0
for batch in validation_generator:
	i += 1
	if i > iters: # save 20 images
		break  # otherwise the generator would loop indefinitely
