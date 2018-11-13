#!/bin/python3
#TODO: write tests for filteModels
#TODO: refactor filtermodels into utils package
#TODO: refactor so only relevant evaluation code is here
#TODO: split model filtering to seperate task creating symlinks
#TODO: add labels to pyplotlib 
import time
import sys
import os
import numpy as np
import re 
sys.path.append("../../code/utils")
import utils
import matplotlib.pyplot as plt
import os

#Steps for evaluating:

#1.0) for each of the CV split:
    # 1.1) Read in evaluation file list and ground truths (cached GT file)
    # 1.2.0)  Read in the filtered models relevant for this split. 
    # 1.3.0) For each of the models:
    # 1.3.1)  Batch predict the validation set of the CV split
    # 1.3.2)  Write file with the results: conf matrix, stat measures and which images are wrong/right 

modelsDir = 'input/'
outputdir = 'output/'

def filterModels(modelsDir,
        configIdBase=["IV3","Xc"],
        configIds = [3,4,5,6,7,71],
        eraindices=range(2),
        cvindices=range(5)):
    seperator="_"
    start="BECONA2.0"
    modprefix="-m"
    configIdVersion=[seperator+"v"+str(n) for n in configIds]
    #cvindices=range(5)
    eras=[seperator+"Era"+str(n) for n in eraindices]
    cvs=[seperator+"split"+str(n) for n in cvindices]
    end=".hdf5"

    allMatched=[]
    for modelFN in os.listdir(modelsDir):
            if modelFN.endswith(end) and modelFN.startswith(start):
                allMatched.append(modelFN[len(start):-len(end)])

#    print(allMatched)
    allMatched.sort()
#    print(len(allMatched))
    epre = re.compile(r'''.*ep(\d+).*''')
    vlre = re.compile(r'''.*vl(\d+)\.(\d+).*''')

    allFiltered = [] 
    bestModConfNameAvg = ''
    bestModConfAvg = 1;
    bestModConfNameMean = ''
    bestModConfMean = 1;
    allbests = []
    filteredLegend = []
    for mod in configIdBase:
        for ver in configIdVersion:
            currModel = mod+ver
            print(currModel)
            filterMatch = False 
            bestOfEachCVsplit = []
            for cv in cvs:
                best_in_cv = 1 
                for era in eras:
                    for fn in allMatched:
                        if fn.startswith(modprefix+currModel+cv+era):
                            filterMatch = True 
                            print(fn)
                            ep = epre.match(fn).group(1)
                            #print(ep)
                            vlmatch0 = vlre.match(fn).group(1)
                            vlmatch1 = vlre.match(fn).group(2)
                            vl = float(vlmatch0+'.'+vlmatch1)
                            if(vl<best_in_cv):
                                best_in_cv = vl
                            #print(vlmatch0)
                            #print(vlmatch1)
                            allFiltered.append({'filename':start+fn+end,
                                'confid':currModel,
                                'split':int(cv[6:]),
                                'era':int(era[4:]),
                                'ep':int(ep),
                                'valloss':vl})
                print(cv + ' -- best vl ==' + str(best_in_cv))
                bestOfEachCVsplit.append(best_in_cv)
            if filterMatch:
                print(bestOfEachCVsplit)
                avg = np.average(bestOfEachCVsplit)
                mean = np.mean(bestOfEachCVsplit)
                #print('###########################' +currModel+ ' average ==' + str(avg))
                #print('###########################' +currModel+ ' mean ==' + str(mean))
                if(avg<bestModConfMean):
                    bestModConfMean = mean 
                    bestModConfNameMean = currModel
                if(avg<bestModConfAvg):
                    bestModConfAvg = avg
                    bestModConfNameAvg = currModel
                filteredLegend.append(currModel)
                allbests.append(bestOfEachCVsplit)
            else:
                print('NONEFORTHISVER'+currModel)
                if not(bestOfEachCVsplit == [1,1,1,1,1]):
                    print('==============================ASSERTION ERROR==============================')
    print(bestModConfNameAvg + 'wins with' + str(bestModConfAvg) + ' validation loss!!')
    print(bestModConfNameMean + 'wins with' + str(bestModConfMean) + ' validation loss!!')
    print(filteredLegend)
    print(allbests)
    datatoboxplot = np.transpose(np.array(allbests))
    print(datatoboxplot)
    #print(allFiltered)
    fig1, ax1 = plt.subplots()
    ax1.set_title('Boxplots of best model version over the 5-fold Cross-validation splits')
    ax1.boxplot(datatoboxplot)
    ax1.set_xticklabels(filteredLegend, rotation=45, fontsize=8)
    plt.show()
    return allFiltered
#command line args as follows:

def prepGroundTruth(dirArg):
    if not os.path.isfile(utils.getGTFileName(dirArg)):
        nbofClasses = utils.makeGroundTruth(dirArg)
    groundTruth,class_indices= utils.getGroundTruth(dirArg)
    #print(class_indices)
    return class_indices,groundTruth
    


#from keras.preprocessing import image as im

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
    #print('prep TIME difference = ', endtime-starttime)
    return X,Y_true,total,names

filterModels(modelsDir)
#filterModels(modelsDir,configIds=[71])


#from keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
#from keras.models import Model,load_model
#from keras.layers import Dense, GlobalAveragePooling2D
#from keras import backend as K
#from keras.callbacks import ModelCheckpoint,EarlyStopping
#from sklearn.metrics import classification_report, confusion_matrix 
#
##Hardcoded input shape to prep first before loading models
#
#for split in range(5):
#    prefix = "../../B2split_" 
#    suffix = "_Val"
#    dirArg = prefix+str(split)+suffix
#    print("prediction dir ="+ dirArg)
#    class_indices,groundTruth = prepGroundTruth(dirArg)
#    class_names = class_indices.keys()
#    nbOfClasses = len(class_indices) 
#    print("Nb of classes in groundTruth file:", nbOfClasses)
#    X,Y_true,total,names = prepImages(dirArg,groundTruth,(299,299,3))
#    names = np.array(names)
#    Y_pred = []
#    splitModels = filterModels(modelsDir,[split])
#    print(splitModels)
#    for modelFile in splitModels:
#        model=load_model(modelsDir+modelFile)
#        inputShape = model.input_shape
#        starttime = time.time()
#        predictions = model.predict(np.array(X), batch_size=32)
#        endtime = time.time()
#        predtime = endtime-starttime
#        Y_pred = np.argmax(predictions, axis=1)
#        Y_predConf = np.amax(predictions, axis=1)
#        IsTrue = np.equal(Y_pred, Y_true)
#        IsFalse = np.logical_not(IsTrue)
#        right=np.sum(IsTrue)
#        wrong=np.sum(IsFalse)
#        wrongsWithConfid = np.asmatrix([names[IsFalse],Y_predConf[IsFalse]]).T
#        summary=np.asmatrix([names, Y_true, IsTrue, Y_pred, Y_predConf]).T
#        perctcorrect = right/total 
#        perctwrong = wrong/total
#        classificationReport = classification_report(Y_true, Y_pred, target_names=class_names)
#        confusionMatrix = confusion_matrix(Y_pred,Y_true)
##TODO add all predictions to saved file
#        np.savez(outputdir+"_results_"+modelFile+'.npz', modelFile=modelFile, perctcorrect=perctcorrect, perctwrong=perctwrong, dirArg=dirArg, summary=summary, wrongsWithConfid=wrongsWithConfid, confusionMatrix=confusionMatrix, classificationReport=classificationReport,predtime=predtime)
#        #Load with data = np.load('filename.npz')
