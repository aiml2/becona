#Steps for evaluating:

#1.0) for each of the CV split:
    # 1.1) Read in evaluation file list and ground truths (cached GT file)
    # 1.2.0)  Read in the filtered models relevant for this split. 
    # 1.3.0) For each of the models:
    # 1.3.1)  Batch predict the validation set of the CV split
    # 1.3.2)  Write file with the results: conf matrix, stat measures and which images are wrong/right 
import os

modelsDir = 'input/'

def filterModels(modelsDir):
    seperator="_"
    modprefix="-m"
    configIdBase=["IV3","Xc"]
    configIds = [3,4]
    configIdVersion=[seperator+"v"+str(n) for n in configIds]
    eraindices=range(2)
    #cvindices=range(5)
    cvindices=range(1)
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
                                allFiltered.append(fn)
                    if not (minEra0Val == None):
                        allFiltered.append(start+minEra0Val+end)
    return allFiltered

print(filterModels(modelsDir))
