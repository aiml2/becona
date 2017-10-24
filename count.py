import sys,os
import math
import sys
sys.path.append("code/utils")
import utils

targetDir = sys.argv[1]

if len(sys.argv) < 3 :
        batchsize = 32 
else:
        batchsize = int(sys.argv[2])

print("Counting .jpg & .png only, using directory structure as hierarchical classes")
print("1) Provide directory to count, (2) optionally a batch size (default =32) for step calculation")

#TODO add groundtruth file maker?
#TXT with filename classnum pairs
#OR TXT with filename labelname pairs??

classes = {}
total = 0

for dirpath, dirs, files in os.walk(targetDir):
        print(dirpath)
        print(dirs)
        classes[dirpath] = 0
        for filename in os.listdir(dirpath):
                if filename.endswith(utils.getWhiteListFormatsTuple()):
                        classes[dirpath] += 1
                        #dirpath.split('/',-1)[-1]
#sumfor total and parent-class-totals

for key,value in classes.items():
        total += classes[key]

print(classes)

print("total="+str(total))

print("Suggested with (batchsize ="+str(batchsize)+") steps_per_epoch =="+ str(math.ceil(total/batchsize)))
