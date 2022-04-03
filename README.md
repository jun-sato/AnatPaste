# Implementation of AnatMix

This is an official PyTorch reimplementation of [AnatMix: Self-Supervised Learning for Anomaly Detection and Localization](https://arxiv.org/abs/2104.0401) and in no way affiliated with the original authors. This repository is mainly based on [this repository](https://github.com/Runinho/pytorch-cutpaste)

## Setup
Our training environments are listed in environment.yml
This yml suppose cudatoolkit=11.3 and python=3.9. Please install corresponding Pytorch version.

## Dataset
After downloading certain dataset and splitting into train,validation and test, you should make the directories as follows,

dataset  
 <span>　</span>|-----train  
 <span>　</span>|          <span>　</span>|---normal  
 <span>　</span>|  
 <span>　</span>|-----valid  
 <span>　</span>|          <span>　</span>|---normal  
 <span>　</span>|          <span>　</span>|---abnormal  
 <span>　</span>|-----test  
 <span>　</span>|          <span>　</span>|---normal  
 <span>　</span>|          <span>　</span>|---abnormal  

You must specify the name of normal directory in datasetl.py.The name of abnormal directory is anything you like.
In this github directory, we use Zhanglab dataset and Chexpert dataset, so their normal directory name  good and No Finding, respectively.

## Run Training

python run_training.py --variant anatmix --type zhanglab  --seed 0 --no-pretrained --cuda 0 --batch_size 64 

and the performance in validation and test set is written as tfrecords in logdirs.

## Flowchart
![flowchart](doc/imgs/flowchart.png)

## Sample imgs

![sample imgs](doc/imgs/anatmix.png)

## Results
<img src="doc/imgs/zhanglab.png" width="40%"> <img src="doc/imgs/chexpert.png" width="40%">
