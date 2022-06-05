data_root = '/nas/k8s/dev/mlops/dataset-artifacts/group_patches'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=[
        dict(
            type='BaseDataset',
            pipelines=[
                dict(type='LoadImageFromFile'),
                dict(type='RandomChannelShift'),
                dict(
                    type='Resize', img_size=(224, 224),
                    ratio_range=(0.8, 1.2)),
                dict(type='RandomCrop', crop_size=(224, 224)),
                dict(type='PhotoMetricDistortion', prob=0.5),
                dict(type='RandomFlip', prob=0.5, direction='horizontal'),
                dict(type='RandomFlip', prob=0.5, direction='vertical'),
                dict(type='RandomRotate', prob=1.0, degree=(-45, 45)),
                dict(type='Pad', size_divisor=224),
                dict(
                    type='Normalize',
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=True,
                ),
                dict(type='Collect'),
            ],
            img_dir='inria/austin1',
            data_root=data_root,
            img_suffix='.png',
        ),
    ],
)
