import numpy as np

class DataLoader:
    def __init__(self,batchSize,inputShape,dataList):
        self.inputShape=inputShape
        self.batchSize=batchSize
        self.dataList=dataList

    def generator(self):
        pass
