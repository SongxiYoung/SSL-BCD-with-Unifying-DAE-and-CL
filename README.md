# Advancing Self-Supervised Learning for Building Change Detection and Damage Assessment: Unified Denoising Autoencoder and Contrastive Learning Framework

The SSL pretraining stage code is modified from MAE(https://github.com/facebookresearch/mae).

## 1. Environment Setup
```
conda env create -f mae.yml
```

## 2. SSL pretraining
```
python main_pretrain_dae.py
```

## 3. Building Segmentation task
```
python main_eval_dae_finetune_seg_edge.py
```

## 4. Building Segmentation & Damage Assessment multi-task
```
python main_eval_dae_finetune_multitask_edge.py
```

## Citations:
```
@inproceedings{peng2021spatiotemporal,
  title={Spatiotemporal contrastive representation learning for building damage classification},
  author={Peng, Bo and Huang, Qunying and Rao, Jinmeng},
  booktitle={2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS},
  pages={8562--8565},
  year={2021},
  organization={IEEE}
}

@article{yang2025advancing,
  title={Advancing Self-Supervised Learning for Building Change Detection and Damage Assessment: Unified Denoising Autoencoder and Contrastive Learning Framework},
  author={Yang, Songxi and Peng, Bo and Sui, Tang and Wu, Meiliu and Huang, Qunying},
  journal={Remote Sensing},
  volume={17},
  number={15},
  pages={2717},
  year={2025},
  publisher={MDPI}
}
```
