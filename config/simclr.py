data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    train=[
        dict(
            type='SimclrDataset',
            pipelines=[
                dict(type='LoadImageFromFile'),
                dict(type='RandomChannelShift'),
                dict(type='GaussianBlur'),
                dict(type='Resize', img_size=(96, 96), ratio_range=(0.8, 1.2)),
                dict(type='RandomCrop', crop_size=(96, 96)),
                dict(type='Solarization', prob=0.2),
                dict(type='PhotoMetricDistortion', prob=0.5),
                dict(type='RandomFlip', prob=0.5, direction='horizontal'),
                dict(type='RandomFlip', prob=0.5, direction='vertical'),
                dict(type='RandomRotate', prob=1.0, degree=(-45, 45)),
                dict(type='Pad', size_divisor=96),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True,
                ),
                dict(type='Collect'),
            ],
            img_dir='stl10',
            data_root='/nas/k8s/dev/mlops/chagmgang/dataset',
            img_suffix='.png',
        ),
    ],
)

model = dict(
    type='Simclr',
    backbone=dict(
        type='ResNet50',
        pretrained=False,
    ),
    projection=dict(
        type='BaseProjection',
        input_dim=2048,
        hidden_dim=2048,
        last_dim=2048,
    ),
)

runner = dict(
    type='BaseRunner',
    max_epochs=1600,
)

optimizer = dict(type='LARS', lr=0.01, weight_decay=6.25e-3)
scheduler = dict(type='cosine_with_warmup', num_warmup_steps=0)
checkpoint = dict()

logger = [
    dict(type='PrintLogger', interval=50),
    dict(
        type='MlflowLogger',
        experiment_name='SSL',
        run_name='resnet50-stl10',
        interval=50,
    ),
]
