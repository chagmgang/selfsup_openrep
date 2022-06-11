## Downstream task in DOTA 2.0 Dataset by mmdetection
---

### Introduction
This project is created to evaluate how well the backbone generated in the selfsup project can generate a good represenstation feature. This project use `mmrotate` as fundamental structure, and custom backbone and dataset class is constructed to evaluate the backbone.

### Dataset
The DOTA2.0 data set, which can be said to be the most representative data set for the rotated object detection task in the remote sensing field, is used.

The dataset can be downloaded from https://captain-whu.github.io/DOTA/dataset.html, and the dataset is prepared by the code provided by mmrotate https://github.com/open-mmlab/mmrotate/tree/main/tools.

### setup
```
pip3 install -r requirements.txt
```

### How to run
```
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 train.py {config_file} --launcher pytorch
```

### Resnet50
|     Initialization    | Pretraining | Pretraining data |  *mAP* |  harbor  |   swimming-pool  | roundabout | bridge | baseball-diamond | ground-track-field | soccer-ball-field | storage-tank | basketball-court | large-vehicle | plane | tennis-court | ship | helicopter | helipad | container-crane | airport | small-vehicle |
|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|
| Random Initialization |      -      |         -        | *0.442*   | 0.555 | 0.478 | 0.514 | 0.327 | 0.610 | 0.448 | 0.246 | 0.565 | 0.335 | 0.688 | 0.803 | 0.815 | 0.777 | 0.272 | 0.000 | 0.000 | 0.096 | 0.426 |
| Supervised Imagenet |      -      |         -        | *0.578*  | 0.665 | 0.561 | 0.597 | 0.428 | 0.751 | 0.716 | 0.685 | 0.616 | 0.669 | 0.727 | 0.813 | 0.904 | 0.804 | 0.527 | 0.000 | 0.061 | 0.373 | 0.501 | 
