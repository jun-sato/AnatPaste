
# AnatPaste
  
We proposed a self-supervised learning based anomaly detection model in chest radiographs.  
This is an official PyTorch implementation of [AnatPaste: Anatomy-aware Self-Supervised Learning for Anomaly Detection in chest radiographs](https://arxiv.org/abs/2205.04282v1).   
This repository is mainly based on [this repository](https://github.com/Runinho/pytorch-cutpaste)

 
# Features
 
* AnatMix designed augmentations such that abnormal areas are localized within an organ.
* AnatMix anomaly detection model is one-class classifier model based on ResNet-18.
 
# Requirement
 
We used the following library.

* pytorch and torchvision
* sklearn
* pandas
* tqdm
* tensorboard


# Installation

Our training environments are listed in environment.yml.  
please use the following command.
```bash
git clone https://github.com/jun-sato/AnatPaste
cd AnatPaste
conda env create -f=env_name.yml
```
 

## Dataset
After downloading certain dataset and splitting into train,validation and test, you should make the directories as follows,
```
dataset  
 |-----train  
 |        |---normal  
 |  
 |-----valid  
 |        |---normal  
 |        |---abnormal  
 |-----test  
 |        |---normal  
 |        |---abnormal  
```
We use three datasets. We can download these datasets as follows;
* [Zhanglab dataset](!https://github.com/coyotespike/zhanglab-chest-xrays)
* [Chexpert dataset](!https://stanfordmlgroup.github.io/competitions/chexpert/)
* [RSNA dataset](!https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data)


# Usage
 


* データセットの説明。どの位置に置き、どのような形式にしておけば良いか？
* 主要なファイルの説明。それぞれどのような関数があり、何ができるか？
* コード実行の手順を記載。どうしたら目的の成果(モデルの学習や成果物の保存など)が得られるか。
 
```bash
cd AnatPaste
python run_training.py --variant anatmix --type zhanglab  --seed 0 --no-pretrained --cuda 0 --batch_size 64 
```

 
# Note
 
You must specify the name and absolute path of normal directory in dataset.py and run_training.py and eval.py.The name of abnormal directory is anything you like.
In this github directory, we use Zhanglab, Chexpert, and RSNA dataset, so their normal directory name  good and No Finding, respectively.
 
# Author
 
作成情報を列挙する
 
* Junya Sato
* 2022/7/
* 更新情報
 
# License
ライセンスを明示する。研究室内での使用限定ならその旨を記載。
 
AnatPaste is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
  
AnatPaste is Private.
