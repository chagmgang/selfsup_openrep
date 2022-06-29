## Beyond ImageNet in Remote Sensing : Looking for a better backbone in the field of remote sensing field beyond ImageNet.
---

### Introduction
---
`selfsup_openrep` is a self-supervised learning method framework for domain representation.

This project is to provide remote sensing imagery weight file for better performance in remote sensing electro-optical imagery. In addition, the simple code for representation learning is provided.

### Implementation
- [x] [Simclr : A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- [ ] [BYOL : Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733)
- [x] [MOCO V3 : An Empirical Study of Training Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.02057)
- [x] [Dino : Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)

### Backbone

* [x] [Resnet](https://arxiv.org/abs/1512.03385)
* [x] [Swin transformer](https://arxiv.org/abs/2103.14030)
* [x] [Convnext](https://arxiv.org/abs/2201.03545)

### Performance

#### Prove that the algorithm works properly
* This table shows that the only algorithm works properly with `STL10` Dataset.

|     Initialization    | Backbone     | Pretraining | Pretraining data |   finetune top1  |   finetune top5  |   knn(20) top 1  |   knn(20) top 5  |
|:---------------------:|:------------:|:-----------:|:----------------:|:----------------:|:----------------:|:----------------:|:----------------:|
| Random Initialization | Resnet50     |      -      |         -        |      58.0250     |     95.7250      |         -        |         -        |
| Random Initialization | Resnet50     |  Simclr     | train+unlabeled  |      89.0500     |     99.6250      |      60.6750     |      92.6000     |
|  Supervised Imagenet  | Resnet50     |      -      |         -        |      91.2125     |     99.4500      |       6.2000     |      40.2125     |


#### Downstream task

* More detailed performance in rotated object detection in link, https://github.com/chagmgang/selfsup_openrep/blob/main/evaluation/DET/README.md.
* More detailed performance in semantic segmentation in link, https://github.com/chagmgang/selfsup_openrep/blob/main/evaluation/SEG/README.md.

|     Initialization    | Backbone | Pretraining | Pretraining data |  Dota2.0 |  Inria |
|:---------------------:|:--------:|:-----------:|:----------------:|:--------:|:------:|
| Random Initialization | Resnet50 |      -      |         -        | 0.442    | 65.1.  |
| Supervised Imagenet   | Resnet50 |      -      |         -        | 0.578    | 77.43  |

### Reference
* https://github.com/open-mmlab/mmcv
* https://github.com/open-mmlab/mmselfsup
* https://github.com/facebookresearch/dino
* https://github.com/open-mmlab/mmclassification
* https://github.com/open-mmlab/mmdetection
* https://github.com/open-mmlab/mmsegmentation
* https://github.com/open-mmlab/mmrotate
* https://captain-whu.github.io/DOTA/dataset.html
* https://project.inria.fr/aerialimagelabeling/
* https://github.com/rwightman/pytorch-image-models
* https://github.com/facebookresearch/moco-v3
