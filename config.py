class config:
    imageShape=(1024,1024,3)
    inputShape=(256,256,3)
    pretrainedModel=None
    classes=3
    model="PathoNet" #PathoNet,FRRN_A,FCRN_B,Deeplab_xception,Deeplab_mobilenet
    logPath="logs\\"
    checkpointsPath="checkpoints\\"
    data_path=""
    loss=""
    optimizer="adam"
    lr=1e-3
    batchSize=16
    epoches=30
    GaussianSize=9
    