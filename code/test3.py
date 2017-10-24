models = ["FTMC_Xception_v3", "FTMC_Xception_v4"]
datapaths = ["../B2split_", "../B2split_"]
eras = range(2)
crossValSet = range(5)

classes = {}
for idx, modelName in enumerate(models):
    module = __import__(modelName)
    class_ = getattr(module,modelName)
    classes[modelName] = class_

print(classes)

for cvIndex in crossValSet: 
    for idx,modelName in enumerate(models):
        modelInstance = classes[modelName]()
        for era in eras:
            print("########## TRAINING era:" + str(era) + ", cvindex" + str(cvIndex)+"of model " + modelName)
            modelInstance.trainEra(era, datapaths[idx], cvIndex)
