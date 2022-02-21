# Implementation of AnatMix

This is an official PyTorch reimplementation of [AnatMix: Self-Supervised Learning for Anomaly Detection and Localization](https://arxiv.org/abs/2104.0401) and in no way affiliated with the original authors. This repository largly rely on [this repository](https://github.com/Runinho/pytorch-cutpaste)

## Setup
Our training environments are listed in environment.yml
This yml suppose cudatoolkit=11.3 and python=3.9. Please install corresponding Pytorch version.

## Run Training

python run_training.py --variant anatmix --type zhanglab  --seed 0 --no-pretrained --cuda 0 --batch_size 64 

and the performance in validation and test set is written as tfrecords in logdirs.

