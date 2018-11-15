import time
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


# Rewrite this for ensemble modeling in input1, input2, ...
# Include rescaling to apporpriate base model sizes? 
# see https://towardsdatascience.com/ensembling-convnets-using-keras-237d429157eb
# https://stackoverflow.com/questions/41903928/add-a-resizing-layer-to-a-keras-sequential-model   

modelArg = sys.argv[1]
dirArg = sys.argv[2]

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)



def prepGroundTruth(dirArg):
    if not os.path.isfile(utils.getGTFileName(dirArg)):
        nbofClasses = utils.makeGroundTruth(dirArg)
    groundTruth,class_indices= utils.getGroundTruth(dirArg)
    print(class_indices)
    return class_indices,groundTruth
    

class_indices,groundTruth = prepGroundTruth(dirArg)
class_names = class_indices.keys()
nbOfClasses = len(class_indices) 
print("Nb of classes in groundTruth file:", nbOfClasses)
decode = dict (zip(class_indices.values(),class_indices.keys()))
# total = 0 
# wrong = 0
# right = 0
#X = []
Y_pred = []

from keras.preprocessing import image as im

def prepImages(dirArg,groundTruth,inputShape):
    names = []
    total = 0
    Y_true = []
    X = np.empty((0,) + inputShape)
    starttime = time.time()
    for filename in groundTruth:
        fullpath = os.path.join(dirArg,filename)
        img = im.load_img(fullpath, target_size=(299,299))
        x = im.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # if total == 0:
        #     print(x)
        X = np.append(X, x, axis=0)
        #print(X.shape)
        #X.append(x)
        Y_true.append(groundTruth[filename])
        names.append(fullpath)
        total +=1
    endtime = time.time()
    print('prep TIME difference = ', endtime-starttime)
    return X,Y_true,total,names

from keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from keras.models import Model,load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix 

# model=load_model(modelArg)
# inputShape = model.input_shape # == (None, 299, 299, 3)
#X = np.empty((0, 299, 299, 3))
#Hardcoded input shape to prep first before loading model
X,Y_true,total,names = prepImages(dirArg,groundTruth,(299,299,3))

model=load_model(modelArg)
inputShape = model.input_shape
#for filename in os.listdir(dirArg):
	#if filename.endswith(".JPG") or filename.endswith(".jpg") or filename.endswith(".png"):

names = np.array(names)
print(names)
print("total is", str(total))
# print(X.shape)
# print(X[0])
starttime = time.time()
predictions = model.predict(np.array(X), batch_size=32)
endtime = time.time()
print('Prediction TIME difference = ', endtime-starttime)

print(predictions)
Y_pred = np.argmax(predictions, axis=1)
print(Y_pred)
Y_predConf = np.amax(predictions, axis=1)
print(Y_predConf)
print(Y_true)
IsTrue = np.equal(Y_pred, Y_true)
IsFalse = np.logical_not(IsTrue)
print(IsTrue)
right=np.sum(IsTrue)
wrong=np.sum(IsFalse)
print("names shape", str(names.shape))
print(IsFalse.shape)
print(names[IsFalse])

sizes = lambda x : sizeof_fmt(os.stat(x).st_size)
wrongsWithConfid = np.asmatrix([names[IsFalse],Y_predConf[IsFalse],list(map(sizes,names[IsFalse]))]).T
print(wrongsWithConfid)



summary=np.asmatrix([names, Y_true, IsTrue, Y_pred, Y_predConf]).T
print(summary)

#     print('Predictions for "'+ filename + '" :' , prediction)
#     actual = np.argmax(prediction)
#     Y_pred.append(actual)
#     conf = np.amax(prediction)
#     if groundTruth[filename] == actual :
#         right +=1
#         print("CORRECT "  + str(decode[actual]) + " (conf=" + str(conf) + ") for "  + filename)
#     else:
#         wrong +=1
#         print("WRONG NOT "  + str(decode[actual]) + " (ensPreds=" + str(prediction) + ") for "  + filename + " ; Should be " + decode[groundTruth[filename]])
#
# #predictions = model.predict(np.array(X))
# #for idx, prediction in predictions:
# 	#filename = names[idx] 
# 	#print('Predictions for "'+ filename + '" :' , prediction)
# 	#actual = np.argmax(prediction)
# 	#conf = np.amax(prediction)
# 	#if groundTruth[filename] == actual :
# 		#right +=1
# 		#print("CORRECT "  + str(decode[actual]) + " (conf=" + str(conf) + ") for "  + filename)
# 	#else:
# 		#wrong +=1
# 		#print("WRONG NOT "  + str(decode[actual]) + " (ensPreds=" + str(prediction) + ") for "  + filename + " ; Should be " + decode[groundTruth[filename]])
# 		
print("Total Correct: " + str(right) + "/" + str(total))
print("Total Wrong: " + str(wrong) + "/" + str(total))
#
classification_report = classification_report(Y_true, Y_pred, target_names=class_names)
print(classification_report)
confusion_matrix = confusion_matrix(Y_pred,Y_true)
print(confusion_matrix)

np.savez('lastresults.npz', modelArg=modelArg, dirArg=dirArg, summary=summary, wrongsWithConfid=wrongsWithConfid, confusion_matrix=confusion_matrix, classification_report=classification_report)
data = np.load('lastresults.npz')
print(data['modelArg'])
print(data['wrongsWithConfid'])
