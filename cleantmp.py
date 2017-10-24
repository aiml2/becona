import sys
import os
import numpy as np
import code 
from code.utils.utils import getConfigId,classFromFilename
import re

modelsDir = '/tmp/'
targetDir = "./tmpmodels/"
assert os.path.isdir(modelsDir)
assert os.path.isdir(targetDir)

seperator="_"
modprefix="-m"
configIdBase=["IV3","Xc"]
configIds = [3,4]
configIdVersion=[seperator+"v"+str(n) for n in configIds]
eraindices=range(2)
cvindices=range(5)
eras=[seperator+"Era"+str(n) for n in eraindices]
cvs=[seperator+"split"+str(n) for n in cvindices]
end=".hdf5"
start="BECONA2.0"
val_seperator='-'

allMatched=[]
for modelFN in os.listdir(modelsDir):
	if modelFN.endswith(end) and modelFN.startswith(start):
            allMatched.append(modelFN[len(start):-len(end)])

print(allMatched)
allMatched.sort()
print(len(allMatched))

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
                    allFiltered.append(minEra0Val)

print(allFiltered)
print(len(allFiltered))

# ids = [getConfigId(fn) for fn in allFiltered]
# print(ids)
# filteredDict =  dict(zip(allFiltered,ids))

from keras.models import Model,load_model
from keras.optimizers import SGD

for name in allFiltered:
    modelName = start+name+end
    oldModelName = modelsDir+modelName
    newModelName = targetDir+modelName
    oldmodel=load_model(oldModelName)
    model = classFromFilename(name)().model
    print(model)
    assert len(oldmodel.layers) == len(model.layers)
    for idx,layer in enumerate(oldmodel.layers):
        assert type(oldmodel.layers[idx]) == type(model.layers[idx])
        model.layers[idx].set_weights(layer.get_weights())
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
    model.save(newModelName)
    if os.path.isfile(newModelName):
        os.remove(oldModelName)
