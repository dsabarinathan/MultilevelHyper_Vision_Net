
# [MOIR´E IMAGE RESTORATION USING MULIT LEVEL HYPER VISION NET](https://arxiv.org/abs/2004.08541) 

A moire pattern in the images is resulting from high frequency patterns captured by the image sensor (colour filter array) that appear after demosaicing. These Moire patterns would appear in natural images of scenes with high frequency content. The Moire pattern can also vary intensely due to a minimal change in the camera direction/positioning. Thus the Moire pattern depreciates the quality of photographs. An important issue in demoireing pattern is that the Moireing patterns have dynamic structure with varying colors and forms. These challenges makes the demoireing more difficult than many other image restoration tasks. Inspired by these challenges in demoireing, a multilevel hyper vision net is proposed to remove the Moire pattern to improve the quality of the images. As a key aspect, in this network we involved residual channel attention block that can be used to extract and adaptively fuse hierarchical features from all the layers efficiently. The proposed algorithms has been tested with the NTIRE 2020 challenge dataset and thus achieved 36.85 and 0.98 Peak to Signal Noise Ratio (PSNR) and Structural Similarity (SSIM) Index respectively.

## MULIT LEVEL HYPER VISION NET diagram 
![alt text](https://github.com/sabaridsn/MultilevelHyper_Vision_Net/blob/master/Demoireing%20.jpg)

## Environment

1. Python 3.6.1
2. Anaconda 5.0.1
3. Ubuntu 16.04 or Windows10

## How to setup the environment

### Step1 

Unzip the downloaded folder


### Step2

Open the powershell or terminal


### Step3

```
$cd yourpathtoLightWeightModel

$pwd
> ~/MultilevelHyper_Vision_Net-

$pip install --upgrade -r requirements.txt

```
## How to test the model on your own imgaes
```
$python test.py --testImagePath=yourpathtoimages
```

## !Results:
 Our model secured the 10th place in the  NTIRE 2020 Challenge on [Image Demoireing](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Yuan_NTIRE_2020_Challenge_on_Image_Demoireing_Methods_and_Results_CVPRW_2020_paper.pdf). 
![alt text](https://github.com/sabaridsn/MultilevelHyper_Vision_Net/blob/master/results.png)
