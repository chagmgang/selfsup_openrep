## Downstream task in DOTA 2.0 Dataset by mmdetection
---

### Introduction
This project is created to evaluate how well the backbone generated in the selfsup project can generate a good represenstation feature. This project use `mmsegmentation` as fundamental structure, and custom backbone and dataset class is constructed to evaluate the backbone.

### Dataset
The inria data set, which can be said to be the most representative data set for the semantic segmentation task in the remote sensing field, is used.

The dataset can be downloaded from https://project.inria.fr/aerialimagelabeling/.

### Dataset Prepaation

The dataset has to be downloaded from [inria dataset link](https://project.inria.fr/aerialimagelabeling/). The inria dataset does not provide train/valid split criterion in public. The dataset is randomly divided to train/valid and recorded in train.txt and valid.txt. Also the dataset with tif extension has to be preprocessed to mmsegmentation style dataset.

```
python3 img_split.py
```

### setup
```
pip3 install -r requirements.txt
```

### How to run
```
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 train.py {config_file} --launcher pytorch
```

### Resnet50
|     Initialization    | Pretraining | Pretraining data |  *mAP* |  background  |   building  |
|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|
| Random Initialization |      -      |         -        | *65.1*   | 88.68 | 41.52 |
| Supervised Imagenet |      -      |         -        | *77.43*  | 92.58 | 62.27 |