## Fine tuning on STL10 to evaluate the method of pretraining performance
---

### Introduction
This small project is to provide the table of performance comparison. Not for best performance on the STL10 dataset, this repository is simply to check whether the self-supervised learning algorithm works.

### setup
```
pip3 install -r requirements.txt
```

### How to run
```
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 train.py {config_file} --launcher pytorch
```

```
python3 knn.py --config-file {config_file}
```

### Performance
|     Initialization    | Backbone     | Pretraining | Pretraining data |   top1  |   top5  |   knn(20) top 1  |   knn(20) top 5  |
|:---------------------:|:------------:|:-----------:|:----------------:|:-------:|:-------:|:----------------:|:----------------:|
| Random Initialization | Resnet50     |      -      |         -        | 58.0250 | 95.7250 |         -        |         -        |
| Random Initialization | Resnet50     |  Simclr     | train+unlabeled  | 89.0500 | 99.6250 |      60.6750     |      92.6000     |
|  Supervised Imagenet  | Resnet50     |      -      |         -        | 91.2125 | 99.4500 |       6.2000     |      40.2125     |
