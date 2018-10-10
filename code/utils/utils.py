import os
import sys
import re
sys.path.append("../../")
import code 

#Returns the groundTruth filename for a given directory
# eg. /predictiondir -> /predictiondir.gt
# eg. ../predictiondir/ -> ../predictiondir.gt
def getGTFileName(directory):
    ext = ".gt"
    if directory.endswith("/"):
        return directory[:-1]+ext
    else:
        return directory+ext

#Returns the groundTruth filename for a given directory
# eg. /predictiondir -> /predictiondir.csv
# eg. ../predictiondir/ -> ../predictiondir.csv
def getCSVFileName(directory):
    ext = ".csv"
    if directory.endswith("/"):
        return directory[:-1]+ext
    else:
        return directory+ext

#Reads a ground truth file of a given directory
# gt file should start with class_indices
#Returns a dictonary with "filename -> groundTruth value"
#and the amount of classes found assuming integers starting from 0 are used concutivly
def getGroundTruth(directory):
    groundTruth = {}
    with open(getGTFileName(directory)) as f:
        for ln,line in enumerate(f):
            if ln == 0:
                classIndices = eval(line)
            else:
               (key, val) = line.rsplit(" ",1)
               groundTruth[key] = int(val)
    return groundTruth,classIndices

# Repurposed from keras see:
# https://github.com/fchollet/keras/blob/419105bd371460720332dcacee0b681322375e9e/keras/preprocessing/image.py 
def _list_valid_filenames_in_directory(directory, white_list_formats,
                                       class_indices, follow_links):
    """List paths of files in `subdir` relative from `directory` whose extensions are in `white_list_formats`.

    # Arguments
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        class_indices: dictionary mapping a class name to its index.

    # Returns
        classes: a list of class indices
        filenames: the path of valid files in `directory`, relative from
            `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be ["class1/file1.jpg", "class1/file2.jpg", ...]).
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

    classes = []
    filenames = []
    subdir = os.path.basename(directory)
    basedir = os.path.dirname(directory)
    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                classes.append(class_indices[subdir])
                # add filename relative to directory
                absolute_path = os.path.join(root, fname)
                filenames.append(os.path.relpath(absolute_path, basedir))
    return classes, filenames


def getWhiteListFormats():
    return {"JPEG","jpeg","jpg","JPG","png","PNG"}

def getWhiteListFormatsTuple():
    return tuple(getWhiteListFormats()) 

def getClassIndices(directory):
    #Use odering used by keras flow_from_directory see
    # https://github.com/fchollet/keras/blob/419105bd371460720332dcacee0b681322375e9e/keras/preprocessing/image.py 
    class_indices={}
    classes = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)
            num_class = len(classes)
            class_indices = dict(zip(classes, range(len(classes))))
    return class_indices

def makeGroundTruth(directory):
    class_indices=getClassIndices(directory)

    if not len(class_indices) > 0: 
        print("No classes found in directory " + directory)
        return 0
    print(class_indices)
    with open(getGTFileName(directory),"w") as f:
        f.write(str(class_indices)+"\n")
        for className in class_indices:
            print(className +"->" + str(class_indices[className]))
            abspath = os.path.abspath(directory+className)
            white_list_formats=getWhiteListFormats()
            classes,filenames = _list_valid_filenames_in_directory(abspath, white_list_formats,class_indices,True)
            for fn in filenames:
                f.write(fn+" "+str(class_indices[className])+"\n")
    return len(class_indices)


def makeAutoKerasCSV(directory):
    class_indices=getClassIndices(directory)

    if not len(class_indices) > 0: 
        print("No classes found in directory " + directory)
        return 0
    print(class_indices)
    with open(getCSVFileName(directory),"w") as f:
        f.write("File Name,Label\n")
        for className in class_indices:
            print(className +"->" + str(class_indices[className]))
            abspath = os.path.abspath(directory+className)
            white_list_formats=getWhiteListFormats()
            classes,filenames = _list_valid_filenames_in_directory(abspath, white_list_formats,class_indices,True)
            for fn in filenames:
                f.write(fn+","+str(class_indices[className])+"\n")
    return len(class_indices)

configIdClassDict = {"IV3" : "FTMC_InceptionV3", "Xc" : "FTMC_Xception" } 

from pydoc import locate
def classFromConfigId(configId):
    print(configId)
    configIdSplit = configId.split("_")
    modelName = configIdClassDict[configIdSplit[0]] + "_" + configIdSplit[1]

    
    # module = __import__("../"+modelName)
    # class_ = getattr(module,modelName)
    class_ = locate("code."+modelName+"."+modelName)
    return class_

configIdRE = re.compile(r".*-m(?P<basemodel>\w*)_(?P<version>\w*)_split.*")
def getConfigId(filename):
    print(filename)
    match = configIdRE.match(filename)
    if match == None:
        raise Exception("Does not contain a valid configId")
    return match.group("basemodel") + "_" + match.group("version")

def getSplitId(filename):
#TODO
    return False

def classFromFilename(fn):
    return classFromConfigId(getConfigId(fn))
