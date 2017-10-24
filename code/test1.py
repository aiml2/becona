models = ["FTMC_InceptionV3_v2", "FTMC_InceptionV3_v1"]
datapaths = ["../tagged_clean/split_crop_", "../tagged_clean/split_uncrop_"]
eras = range(2)
crossValSet = range(5)

for idx, modelName in enumerate(models):
    module = __import__(modelName)
    class_ = getattr(module,modelName)
    for cvIndex in crossValSet: 
        modelInstance = class_()
        for era in eras:
            print("########## TRAINING era:" + str(era) + ", cvindex" + str(cvIndex)+"of model " + modelName)
            modelInstance.trainEra(era, datapaths[idx], cvIndex)
