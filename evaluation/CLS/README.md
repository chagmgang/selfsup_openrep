## Fine tuning on STL10 to evaluate the method of pretraining performance
---

### Introduction
---
This small project is to provide the table of performance comparison. Not for best performance on the STL10 dataset, this repository is simply to check whether the self-supervised learning algorithm works.

### How to run
---
```
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 train.py {config_file} --launcher pytorch
```

### Resnet50
|     Initialization    | Pretraining | Pretraining data |   top1  |   top5  |
|:---------------------:|:-----------:|:----------------:|:-------:|:-------:|
| Random Initialization |      -      |         -        | 58.0250 | 95.7250 |
| Random Initialization |  Simclr     | train+unlabeled  |         |         |
|  Supervised Imagenet  |      -      |         -        | 91.2125 | 99.4500 |
|  Supervised Imagenet  |  Simclr     | train+unlabeled  |         |         |