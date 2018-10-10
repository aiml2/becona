import time
import sys
import os
import numpy as np
sys.path.append("../../code/utils")
import utils

#Steps for evaluating:

#1.0) for each of the CV split:
    # 1.1) Read in evaluation file list and ground truths (cached GT file)
    # 1.2.0)  Read in the filtered models relevant for this split. 
    # 1.3.0) For each of the models:
    # 1.3.1)  Batch predict the validation set of the CV split
    # 1.3.2)  Write file with the results: conf matrix, stat measures and which images are wrong/right 
import os

modelsDir = 'input/'
outputdir = 'output/'

def filterModels(modelsDir,cvindices=range(5)):
    seperator="_"
    modprefix="-m"
    configIdBase=["IV3","Xc"]
    configIds = [3,4]
    configIdVersion=[seperator+"v"+str(n) for n in configIds]
    eraindices=range(2)
    #cvindices=range(5)
    eras=[seperator+"Era"+str(n) for n in eraindices]
    cvs=[seperator+"split"+str(n) for n in cvindices]
    end=".hdf5"
    start="BECONA2.0"
    val_seperator='-'

    allMatched=[]
    for modelFN in os.listdir(modelsDir):
            if modelFN.endswith(end) and modelFN.startswith(start):
                allMatched.append(modelFN[len(start):-len(end)])

#    print(allMatched)
    allMatched.sort()
#    print(len(allMatched))

    allFiltered = [] 
    for mod in configIdBase:
        for ver in configIdVersion:
            for cv in cvs:
                for era in eras:
                    minEra0Val = None
                    for fn in allMatched:
                        if fn.startswith(modprefix+mod+ver+cv+era):
                            if era == "_Era0":
                                    minEra0Val = fn
                            else:
                                print(fn)
                                allFiltered.append(start+fn+end)
                    if not (minEra0Val == None):
                        allFiltered.append(start+minEra0Val+end)
    return allFiltered
#command line args as follows:



def prepGroundTruth(dirArg):
    if not os.path.isfile(utils.getGTFileName(dirArg)):
        nbofClasses = utils.makeGroundTruth(dirArg)
    groundTruth,class_indices= utils.getGroundTruth(dirArg)
    print(class_indices)
    return class_indices,groundTruth
    


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

#print(filterModels(modelsDir,[3]))

from keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from keras.models import Model,load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix 

#Hardcoded input shape to prep first before loading models

for split in range(5):
    prefix = "../../B2split_" 
    suffix = "_Val"
    dirArg = prefix+str(split)+suffix
    print("prediction dir ="+ dirArg)
    class_indices,groundTruth = prepGroundTruth(dirArg)
    class_names = class_indices.keys()
    nbOfClasses = len(class_indices) 
    print("Nb of classes in groundTruth file:", nbOfClasses)
    X,Y_true,total,names = prepImages(dirArg,groundTruth,(299,299,3))
    names = np.array(names)
    Y_pred = []
    splitModels = filterModels(modelsDir,[split])
    print(splitModels)
    for modelFile in splitModels:
        model=load_model(modelsDir+modelFile)
        inputShape = model.input_shape
        starttime = time.time()
        predictions = model.predict(np.array(X), batch_size=32)
        endtime = time.time()
        predtime = endtime-starttime
        Y_pred = np.argmax(predictions, axis=1)
        Y_predConf = np.amax(predictions, axis=1)
        IsTrue = np.equal(Y_pred, Y_true)
        IsFalse = np.logical_not(IsTrue)
        right=np.sum(IsTrue)
        wrong=np.sum(IsFalse)
        wrongsWithConfid = np.asmatrix([names[IsFalse],Y_predConf[IsFalse]]).T
        summary=np.asmatrix([names, Y_true, IsTrue, Y_pred, Y_predConf]).T
        perctcorrect = right/total 
        perctwrong = wrong/total
        classification_report = classification_report(Y_true, Y_pred, target_names=class_names)
        confusion_matrix = confusion_matrix(Y_pred,Y_true)
        np.savez(outputdir+"_results_"+modelFile+'.npz', modelFile=modelFile, perctcorrect=perctcorrect, perctwrong=perctwrong, dirArg=dirArg, summary=summary, wrongsWithConfid=wrongsWithConfid, confusion_matrix=confusion_matrix, classification_report=classification_report,predtime=predtime)
        #Load with data = np.load('filename.npz')
