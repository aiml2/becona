import sys
import os
import numpy as np
sys.path.append("code/utils")
import utils
#command line args as follows:
print("Uses all models ending with finetuned.h5 in an ensemble")
print("Uses images_dir.gt as ground truth")
print("Command line args: models_dir images_dir WITH / terminator")

modelsDir = sys.argv[1]
dirArg= sys.argv[2]

groundTruth,class_indices= utils.getGroundTruth(dirArg)
decode = dict (zip(class_indices.values(),class_indices.keys()))
print(class_indices)
nbOfClasses = len(class_indices) 
print("Nb of classes in groundTruth file:", nbOfClasses)

total = 0 
wrong = 0
right = 0
y_true = []
y_pred = []
class_names = class_indices.keys()
ensemble = [] 

from keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from keras.preprocessing import image as im
from keras.models import Model,load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix 




for modelFN in os.listdir(modelsDir):
	if modelFN.endswith("ep20.hdf5") and modelFN.startswith("BECONA0.2-mIV3_v2_split"):
		print(modelsDir+modelFN)
		ensemble.append(load_model(modelsDir+modelFN))

for filename in os.listdir(dirArg):
	if filename.endswith(".JPG") or filename.endswith(".jpg") or filename.endswith(".png"):
		total += 1
		img = im.load_img(dirArg+filename, target_size=(299,299))
		x = im.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		predictions = []
		y_true.append(groundTruth[filename])
		for model in ensemble:	
			predictions.append(model.predict(x))
		print('Predictions for "'+ filename + '" :' , predictions)
		ensemblePred = np.average(predictions,axis=0)
		print('Averaged prediction for "'+ filename + '" :' , ensemblePred)
		actual = np.argmax(ensemblePred)
		y_pred.append(actual)
		conf = np.amax(ensemblePred)
		if groundTruth[filename] == actual :
			right += 1
			print("CORRECT "  + str(decode[actual]) + " (conf=" + str(conf) + ") for "  + filename)
		else:
			wrong += 1
			print("WRONG NOT"  + str(decode[actual]) + " (ensPreds=" + str(ensemblePred) + ") for "  + filename+ " ; Should be " + decode[groundTruth[filename]])
			
print("Total Correct: " + str(right) + "/" + str(total))
print("Total Wrong: " + str(wrong) + "/" + str(total))

print(classification_report(y_true, y_pred, target_names=class_names))
print(confusion_matrix(y_pred,y_true))
