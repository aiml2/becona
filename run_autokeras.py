import time
import sys
import os
import numpy as np
sys.path.append("code/utils")
import utils
import autokeras
from autokeras.image_supervised import load_image_dataset
from autokeras.image_supervised import ImageClassifier
from autokeras.image_supervised import read_csv_file
from autokeras.image_supervised import read_images

if __name__ == '__main__':
#command line args as follows:
    print("Command line args: trainDir valDir")
    print("Uses images_dir.csv as label data")
    print("Command line args: images_dir WITH / terminator")
    print("Will try to create an AutoKeras (.csv) file if possible")

    trainDirArg = sys.argv[1]
    valDirArg = sys.argv[2]

    if not os.path.isfile(utils.getCSVFileName(trainDirArg)):
        nbofClasses = utils.makeAutoKerasCSV(trainDirArg)

    if not os.path.isfile(utils.getCSVFileName(valDirArg)):
        nbofClasses = utils.makeAutoKerasCSV(valDirArg)

    x_train, y_train = load_image_dataset(csv_file_path=utils.getCSVFileName(trainDirArg),images_path=trainDirArg)
    print(x_train.shape)
    print(x_train[0].shape)
    print(x_train[0][0].shape)
    print(y_train.shape)

# fn,fl = read_csv_file(utils.getCSVFileName(trainDirArg))
# print(fn)
# x = read_images(fn,trainDirArg)
# print(np.array(x).shape)
# print(np.array(x)[0].shape)
# print(np.array(x)[0][0].shape)
# print(np.array(x)[1][0].shape)
# print(np.array(x)[1][1].shape)

    x_val, y_val = load_image_dataset(csv_file_path=utils.getCSVFileName(valDirArg), images_path=valDirArg)
    print(x_val.shape)
    print(y_val.shape)

    clf = ImageClassifier(verbose=True)
    clf.fit(x_train, y_train, time_limit=48 * 60 * 60)
    clf.final_fit(x_train, y_train, x_val, y_val, retrain=True)
    y = clf.evaluate(x_val, y_val)
    print(y)
    clf.load_searcher().load_best_model().produce_keras_model().save('autokeras_model.h5')
