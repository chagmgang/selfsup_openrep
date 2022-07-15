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

### Pretraining Dataset

* [spacenet6](https://spacenet.ai/sn6-challenge/)
* [dior](https://arxiv.org/abs/2110.01931)
* [fair1m v2.0](http://gaofen-challenge.com/benchmark)
* [nia building segmentation](https://github.com/SIAnalytics/buildingdetection2020)
* [nia object detection](https://github.com/SIAnalytics/roas)
* [inria](https://project.inria.fr/aerialimagelabeling/)
* [dota 2.0](https://captain-whu.github.io/DOTA/dataset.html)
* [rareplane](https://www.cosmiqworks.org/rareplanes/)

### Performance

#### Prove that the algorithm works properly
* This table shows that the only algorithm works properly with imagenet-1k dataset.
* knn is applied with 20 samples

|      Backbone     | resolution | Pretraining | F-T top1  |   F-T top5  | weight |
|:------------:|:-----------:|:-----------:|:----------------:|:----------------:|:----------------:|
|  Resnet50     |  224x224 |    -      |         -        |      -     | - |
|  Resnet50     |  224x224 |    Simclr      |         72.2440        |      89.9420     | [model](https://drive.google.com/file/d/15P7Ss_2Bhdbeb1jRGTxfHr5xaVLw-pbH/view?usp=sharing)/[config](https://drive.google.com/file/d/1SgKtAH6pa3sJlLM0rlyZ3eM_XvETM-an/view?usp=sharing) |
|  ViT-S/16     |  224x224 |    -      |         -        |      -     | - |
|  ViT-S/16     | 224x224 |     Moco V3      |         71.0860        |      88.9180     | [model](https://drive.google.com/file/d/1HBrTnz6BvNGcLhzzALdf-ljVB6QWiwrl/view?usp=sharing)/[config](https://drive.google.com/file/d/1CG3miiQVbP6o2Qx6w9rWQpN0sGwcSw5n/view?usp=sharing) |
|  ViT-S/16     |  224x224 |    DINO      |         -        |      -     | - |


#### Downstream task

* More detailed performance in rotated object detection in link, https://github.com/chagmgang/selfsup_openrep/blob/main/evaluation/DET/README.md.
* More detailed performance in semantic segmentation in link, https://github.com/chagmgang/selfsup_openrep/blob/main/evaluation/SEG/README.md.

|     Initialization    | Backbone | Pretraining | Pretraining data |  Dota2.0 |  Inria | weight |
|:---------------------:|:--------:|:-----------:|:----------------:|:--------:|:------:|:------:|
| Random Initialization | Resnet50 |      -      |         -        | 0.442    | 65.10  |    -   |
| Random Initialization | Resnet50 |    Simclr   |  remote sensing  | 0.554    | 88.23  |  [model](https://drive.google.com/file/d/18q47LNTfSbZ506-9ov9wbyTONns33ZuH/view?usp=sharing)/[log](https://drive.google.com/file/d/1OC6lPrwnhG0DKVnZmMpiMXX6gD2asYEr/view?usp=sharing)      |
| Supervised Imagenet   | Resnet50 |      -      |         -        | 0.578    | 77.43  |    -   |

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
* https://gcheng-nwpu.github.io/
* https://spacenet.ai/sn6-challenge/
* http://gaofen-challenge.com/benchmark
* https://www.cosmiqworks.org/rareplanes/
