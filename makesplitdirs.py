import os
import sys
import random
sys.path.append("code/utils")
import utils
random.seed(42)
#rng = random.Random()
#rng.seed(42)

alldir="TaggedBatch2"
targetdir="B2split_"
nbofchunks=5

#TODO make deterministic with decent seed?

subdirpaths=[]

l = list(range(20))
print(l)
random.shuffle(l)
print(l)

for dirpath, dirs, files in os.walk(alldir):
        for subdir in dirs:
                print("subdir="+subdir)
                print("dirpath=="+dirpath)
                print("____")
                subdirpaths.append(dirpath+"/"+subdir)

print(subdirpaths)

filespersubdir = {subdir : [] for subdir in subdirpaths} 

print(filespersubdir)

for subdir in subdirpaths:
        for filename in os.listdir(subdir):
                if filename.endswith(utils.getWhiteListFormatsTuple()):
                        filespersubdir[subdir].append(subdir+"/"+filename)

def chunks(l, n):
    return (l[i::n] for i in range(n))

random.shuffle(l)
print(l)

#shuffle and split in chunks
for subdir, files in filespersubdir.items():
        print(subdir)
        if not(files == []):
                #sort for deterministic runs 
                files.sort()
                random.shuffle(files)
                #print(files)
                filespersubdir[subdir] = list(chunks(files,nbofchunks))


def mysymlink(newdir, oldname):
        #Make relative symlink
        upcount = oldname.count('/')
                #src , dest!!
        os.symlink("../"*upcount+oldname,newdir+"/"+oldname.split('/')[-1])


for n in range(nbofchunks):
        for subdir,files in filespersubdir.items():
                #Remove the top dir from subdir and add actual subdirs it to the new name
                newdirTrain=targetdir+str(n)+"_Train/"+subdir.split("/",1)[-1]
                newdirVal=targetdir+str(n)+"_Val/"+subdir.split("/",1)[-1]
                print(newdirTrain)
                if not(os.path.isdir(newdirTrain)):
                        os.makedirs(newdirTrain)
                if not(os.path.isdir(newdirVal)):
                        os.makedirs(newdirVal)
                
                if not(files ==[]):
                        #take e.g. 0 - 1 2 3 4 // 1 - 0 2 3 4 // ...
                        trainIndices = list(range(nbofchunks))
                        trainIndices.remove(n)
                        trainFiles = []
                        for ti in trainIndices:
                                trainFiles += files[ti]
                        for t in trainFiles:
                                mysymlink(newdirTrain,t)

                        valFiles = files[n] 
                        for v in valFiles:
                                mysymlink(newdirVal,v)
