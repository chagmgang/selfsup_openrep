## Evaluate the method of pretraining performance
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