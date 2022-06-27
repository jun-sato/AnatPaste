
# AnatMix
  
We proposed a self-supervised learning based anomaly detection model in chest radiographs.
This is an official PyTorch implementation of [AnatPaste: Anatomy-aware Self-Supervised Learning for Anomaly Detection in chest radiographs](https://arxiv.org/abs/2205.04282v1). This repository is mainly based on [this repository](https://github.com/Runinho/pytorch-cutpaste)

 
# Features
 
* AnatMix designed augmentations such that abnormal areas are localized within an organ.
* AnatMix anomaly detection model is one-class classifier model based on ResNet-18.
 
# Requirement
 
We used the following library.

* python=3.9
* cudatoolkit=11.3

 
# Installation

Our training environments are listed in environment.yml.  
please use the following command.
```
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



# Usage
 
"hoge"の基本的な使い方を説明する。

**丁寧かつ簡潔に、初めて見た人でも理解できる様に**

* データセットの説明。どの位置に置き、どのような形式にしておけば良いか？
* 主要なファイルの説明。それぞれどのような関数があり、何ができるか？
* コード実行の手順を記載。どうしたら目的の成果(モデルの学習や成果物の保存など)が得られるか。
 
```bash
git clone https://github.com/hoge/~
cd examples
python demo.py
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
 
"hoge" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
  
"hoge" is Private.
