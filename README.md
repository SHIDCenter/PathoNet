# PathoNet introduced as a deep neural network backend for evaluation of Ki-67 and tumor-infiltrating lymphocytes in breast cancer
This repository is code release for the study published [here](https://www.nature.com/articles/s41598-021-86912-w) on Scientific Reports Journal.

<p align="center">    
  <img src="https://github.com/SHIDCenter/PathoNet/blob/master/doc/pipeline_LQ.jpg">
</p>


### Abstract
The nuclear protein Ki-67 and Tumor infiltrating lymphocytes (TILs) have been introduced as prognostic factors in predicting both tumor progression and probable response to chemotherapy. The value of Ki-67 index and TILs in approach to heterogeneous tumors such as Breast cancer (BC) that is the most common cancer in women worldwide, has been highlighted in literature. Considering that estimation of both factors are dependent on professional pathologists’ observation and inter-individual variations may also exist, automated methods using machine learning, specifically approaches based on deep learning, have attracted attention. Yet, deep learning methods need considerable annotated data. In the absence of publicly available benchmarks for BC Ki-67 cell detection and further annotated classification of cells, In this study we propose SHIDC-BC-Ki-67 as a dataset for the aforementioned purpose. We also introduce a novel pipeline and backend, for estimation of Ki-67 expression and simultaneous determination of intratumoral TILs score in breast cancer cells. Further, we show that despite the challenges that our proposed model has encountered, our proposed backend, PathoNet, outperforms the state of the art methods proposed to date with regard to harmonic mean measure acquired.

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
1. `pip install -r ./requirements.txt`

### Tested Config
- Cuda : 10.0
- Tensorflow : 10.13.1
- keras : 2.2.4
- Nvidia Driver version : 436.48

## Pretrained Models
You can download the pretrained models :
  1. [DeepLab MobileNet](https://drive.google.com/file/d/1cGiM8LHYCycCCUrxNegWFAcaLTtXbfO5/view?usp=sharing)
  2. [DeepLab Xception](https://drive.google.com/file/d/1Mcn_85Sd2STYZQw5TWhhLaBEOcErPZoa/view?usp=sharing)
  3. [FCRNA](https://drive.google.com/file/d/1I48I_1xJvxH2Ug-C1qt-XRT_gpmqJIYd/view?usp=sharing)
  4. [FCRNB](https://drive.google.com/file/d/1h3alzYMF6SSCg7kNRQEsrrdtBJ3KiRQR/view?usp=sharing)
  5. [PathoNet](https://drive.google.com/file/d/13M6WpBsY_XtIKev_A6EK_Cj2LuBySM3K/view?usp=sharing)
  6. [U-Net](https://drive.google.com/file/d/1WOVw3vCBZkN9Nk58Il79gcnI7CoSsWfb/view?usp=sharing)
  
## Datasets
You need to download SHIDC-B-Ki-67-V1.0 dataset by following instructions [here](https://shiraz-hidc.com/service/ki-67-dataset/). You can wether download the pretrianed models and skip the training or perfotrm training from scratch. Table below shows statistics of this dataset.

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
  <img src="https://github.com/SHIDCenter/PathoNet/blob/master/doc/quant_1.png">
</p>
<p align="center">    
  <img src="https://github.com/SHIDCenter/PathoNet/blob/master/doc/quant_2.png">
</p>
<p align="center">    
  <img src="https://github.com/SHIDCenter/PathoNet/blob/master/doc/quant_3.png">
</p>

### Qualitative Results
Qualitative result samples on cropped images:
<p align="center">    
  <img  src="https://github.com/SHIDCenter/PathoNet/blob/master/doc/qual_res_LQ.jpg">
</p>

Below, you can see a sample of Raw image (4912x3684 pixels) and it's prediction using the proposed backend:
<p align="center">    
  <img  src="https://github.com/SHIDCenter/PathoNet/blob/master/doc/qual_2.png">
</p>

## Citation
If you found this work useful in your research, please give credits to the authors by citing:
```
@article{negahbani2021pathonet,
  title={PathoNet introduced as a deep neural network backend for evaluation of Ki-67 and tumor-infiltrating lymphocytes in breast cancer},
  author={Negahbani, Farzin and Sabzi, Rasool and Jahromi, Bita Pakniyat and Firouzabadi, Dena and Movahedi, Fateme and Shirazi, Mahsa Kohandel and Majidi, Shayan and Dehghanian, Amirreza},
  journal={Scientific reports},
  volume={11},
  number={1},
  pages={1--13},
  year={2021},
  publisher={Nature Publishing Group}
}
```
