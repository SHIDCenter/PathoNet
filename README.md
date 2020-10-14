# PathoNet: Deep learning assisted evaluation of Ki-67 and tumor infiltrating lymphocytes (TILs) as prognostic factors in breast cancer; A large dataset and baseline

<p align="center">    
  <img src="https://github.com/SHIDCenter/PathoNet/blob/master/doc/pipeline_LQ.jpg">
</p>


### Abstract
The nuclear protein Ki-67 and Tumor infiltrating lymphocytes (TILs) have been introduced as prognostic factors in predicting tumor progression and its treatment response. The value of Ki-67 index and TILs in approach to heterogeneous tumors such as Breast cancer (BC), known as the most common cancer in women worldwide, has been highlighted in literature. Due to the indeterminable and subjective nature of Ki-67 as well as TILs scoring, automated methods using machine learning, specifically approaches based on deep learning, have attracted attention. Yet, deep learning methods need considerable annotated data. In the absence of publicly available benchmarks for BC Ki-67 stained cell detection and further annotated classification of cells, we propose SHIDC-BC-Ki-67 as a dataset for aforementioned purpose. We also introduce a novel pipeline and a backend, namely PathoNet for Ki-67 immunostained cell detection and classification and simultaneous determination of intratumoral TILs score. Further, we show that despite facing challenges, our proposed backend, PathoNet, outperforms the state of the art methods proposed to date in the harmonic mean measure.

### PathoNet Backend

<p align="center">    
  <img  src="https://github.com/SHIDCenter/PathoNet/blob/master/doc/PathoNet-architecture_LQ.jpg">
</p>

## Contents

- [Requirments](#requirements)
- [Pretrained Models](#pretrained-models)
- [Datasets](#datasets)
- [Scripts](#scripts) 
- [Results](#results)


## Requirements
1. `pip install ./requirements.txt`

### Tested Config
- Cuda : 10.0
- Tensorflow : 10.13.1
- keras : 2.2.4
- Nvidia river version : 436.48


## Pretrained Models
You can download the pretrained models:
  1. [DeepLab MobileNet](https://drive.google.com/file/d/1Zx1pwWVK2TlvBC91yoL3vJdEyGGGpHHW/view?usp=sharing)
  2. [DeepLab Xception](https://drive.google.com/file/d/1qJA1_CbjUPdv_pvo4pVVIHPxkPPWdSJ-/view?usp=sharing)
  3. [FCRNA](https://drive.google.com/file/d/12CBfmZNTPdA9A40CTsjcIp6l2ICMvi12/view?usp=sharing)
  4. [FCRNB](https://drive.google.com/file/d/14RykzFBl4_usilZ8fSovkbJcQBOH-FyZ/view?usp=sharing)
  5. [PathoNet](https://drive.google.com/file/d/1SbFBRHVgvotpXkJwLgYT3KQJCgDcKFNB/view?usp=sharing)
  
## Datasets
You need to download SHIDC-B-Ki-67 dataset by following instructions [here](http://www.shidc.ir/). You can wether download the pretrianed models and skip the training or perfotrm training from scratch. Table below shows statistics of this dataset.

<p align="center">    
  <img src="https://github.com/SHIDCenter/PathoNet/blob/master/doc/1.PNG">
</p>

## Scripts
In the following section, usecases of the scripts are provided.

### Train 
To train a model:
  ```
  python trian.py -c <config file path>
  ```

### Evaluation
To evaluate file/files run the following command:
```
python evaluation.py -i <test data directory> -c <config file path>
```

### Demo
To run demo on file/files run the following command:
```
python demo.py -i <test data directory/file path> -o <output directory> -c <config file path>
```

### Preprocessing 
In order to perform preprocessing run the following command:
```
python preprocessing.py -i <input data directory> -o <output data directory> -s <output size(default=256,256,3)> -a <data augmentation flag(default=False)> -g <label gaussian size(default=9)>
```


## Results
In this section results obtained from this study is shown. Note that due to different initialization points, final result may vary a bit.

### Quantitative Results
<p align="center">    
  <img src="https://github.com/SHIDCenter/PathoNet/blob/master/doc/2.PNG">
</p>
<p align="center">    
  <img src="https://github.com/SHIDCenter/PathoNet/blob/master/doc/3.PNG">
</p>
<p align="center">    
  <img src="https://github.com/SHIDCenter/PathoNet/blob/master/doc/4.PNG">
</p>

### Qualitative Results

<p align="center">    
  <img  src="https://github.com/SHIDCenter/PathoNet/blob/master/doc/qual_res_LQ.jpg">
</p>

## TODO
- [ ] Add more comments
- [ ] Threshold tunning code
