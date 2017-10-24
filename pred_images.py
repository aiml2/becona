import sys
import os
import numpy as np
sys.path.append("code/utils")
import utils
#command line args as follows:
print("Command line args: model directory")
print("Uses images_dir.gt as ground truth")
print("Command line args: models_dir images_dir WITH / terminator")
print("Will try to create a groundTruth (.gt) file if possible")

modelArg = sys.argv[1]
dirArg = sys.argv[2]

if not os.path.isfile(utils.getGTFileName(dirArg)):
    nbofClasses = utils.makeGroundTruth(dirArg)

groundTruth,class_indices= utils.getGroundTruth(dirArg)
decode = dict (zip(class_indices.values(),class_indices.keys()))
print(class_indices)
nbOfClasses = len(class_indices) 
print("Nb of classes in groundTruth file:", nbOfClasses)
total = 0 
wrong = 0
right = 0
X = []
y_true = []
y_pred = []
class_names = class_indices.keys()
names = []

from keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from keras.preprocessing import image as im
from keras.models import Model,load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix 

model=load_model(modelArg)

#for filename in os.listdir(dirArg):
	#if filename.endswith(".JPG") or filename.endswith(".jpg") or filename.endswith(".png"):
for filename in groundTruth:
    img = im.load_img(dirArg+filename, target_size=(299,299))
    x = im.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    X.append(x)
    y_true.append(groundTruth[filename])
    names.append(filename)
    total +=1
    prediction = model.predict(x)
    print('Predictions for "'+ filename + '" :' , prediction)
    actual = np.argmax(prediction)
    y_pred.append(actual)
    conf = np.amax(prediction)
    if groundTruth[filename] == actual :
        right +=1
        print("CORRECT "  + str(decode[actual]) + " (conf=" + str(conf) + ") for "  + filename)
    else:
        wrong +=1
        print("WRONG NOT "  + str(decode[actual]) + " (ensPreds=" + str(prediction) + ") for "  + filename + " ; Should be " + decode[groundTruth[filename]])

#predictions = model.predict(np.array(X))
#for idx, prediction in predictions:
	#filename = names[idx] 
	#print('Predictions for "'+ filename + '" :' , prediction)
	#actual = np.argmax(prediction)
	#conf = np.amax(prediction)
	#if groundTruth[filename] == actual :
		#right +=1
		#print("CORRECT "  + str(decode[actual]) + " (conf=" + str(conf) + ") for "  + filename)
	#else:
		#wrong +=1
		#print("WRONG NOT "  + str(decode[actual]) + " (ensPreds=" + str(prediction) + ") for "  + filename + " ; Should be " + decode[groundTruth[filename]])
		
print("Total Correct: " + str(right) + "/" + str(total))
print("Total Wrong: " + str(wrong) + "/" + str(total))

print(classification_report(y_true, y_pred, target_names=class_names))
print(confusion_matrix(y_pred,y_true))
