# Advancing Self-Supervised Learning for Building Change Detection and Damage Assessment: Unified Denoising Autoencoder and Contrastive Learning Framework

The SSL pretraining stage code is modified from MAE(https://github.com/facebookresearch/mae).

## 1. Environment Setup
  conda env create -f environment.yml

## 2. SSL pretraining
run main_pretrain_dae.py

## 3. Building Segmentation task
run main_eval_dae_finetune_seg_edge.py

## 4. Building Segmentation & Damage Assessment multi-task
run main_eval_dae_finetune_multitask_edge.py
