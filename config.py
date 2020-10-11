class Config:
    def __init__(self):
        self.imageShape=(1024,1024,3)
        self.inputShape=(256,256,3)
        self.pretrainedModel=None
        self.classes=3
        self.model="PathoNet" #PathoNet,FRRN_A,FCRN_B,Deeplab_xception,Deeplab_mobilenet
        self.logPath="logs/"
        self.data_path=""
        self.loss="mse"
        self.optimizer="adam"
        self.lr=1e-2
        self.batchSize=16
        self.epoches=30
        self.validationSplit=0.2
        self.trainDataPath=""
        self.thresholds=[65,160,155]
        self.guaMaxValue=2550    