from abc import ABC,abstractmethod

class AbstractFineTuneModelConfig(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def trainEra(self, eraInt, dataDirPrefix, cvIndex):
        pass

#    @property
#    def nbOfClasses(self):
#        raise NotImplementedError
#
#    @property
#    def baseModel(self):
#        raise NotImplementedError
#
#    @property
#    def model(self):
#        raise NotImplementedError
#
#    @property
#    def imgDatagen(self):
#        raise NotImplementedError
#
#    @property
#    def nbOfEras(self):
#        raise NotImplementedError
