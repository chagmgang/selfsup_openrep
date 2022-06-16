# runtime
# checkpoint saving
checkpoint_config = dict(interval=40)

# yapf:disable
log_config = dict(
    interval=25,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='CustomMlflowLoggerHook',
            exp_name='SSL',
            run_name='remote-resnet50-simclr',
        ),
    ])
# yapf:enable

# runtime settings
dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
persistent_workers = True

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# schedule
# optimizer
optimizer = dict(type='SGD', lr=2.4, weight_decay=1e-6, momentum=0.9)
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=1e-4,  # cannot be 0
    warmup_by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=800)

# data

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GaussianBlur'),
    dict(type='Resize', img_size=(224, 224), ratio_range=(0.8, 1.5)),
    dict(type='RandomCrop', crop_size=(224, 224)),
    dict(type='Solarization', prob=0.2),
    dict(type='PhotoMetricDistortion', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=1.0, degree=(-45, 45)),
    dict(type='Pad', size_divisor=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),
    dict(type='Collect'),
]

data = dict(
    samples_per_gpu=384,
    workers_per_gpu=16,
    train=[
        dict(
            type='ListSimclrDataset',
            pipelines=train_pipeline,
            txt_file='/nas/k8s/dev/mlops/chagmgang/dataset/' + txt_file,
        ) for txt_file in [
            'dior.txt',
            'dota.txt',
            'fair1m_train_part1.txt',
            'fair1m_train_part2.txt',
            'fair1m_validation.txt',
            'inria.txt',
            'nia.txt',
            'nia_building.txt',
            'rareplane.txt',
            'spacenet6.txt',
        ]
    ])

# models
model = dict(
    type='Simclr',
    backbone=dict(
        type='ResNet50',
        pretrained=True,
        weight=None,
    ),
    projection=dict(
        type='BaseProjection',
        input_dim=2048,
        hidden_dim=2048,
        last_dim=2048,
    ),
)
