# runtime
# checkpoint saving
checkpoint_config = dict(interval=10)

# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
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
optimizer = dict(type='LARS', lr=4.8, weight_decay=1e-6, momentum=0.9)
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1e-4,  # cannot be 0
    warmup_by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)

# data
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type='SimclrDataset',
        data_root='/nas/k8s/dev/mlops/chagmgang/dataset',
        img_dir='inria/valid/images',
        img_suffix='.png',
        pipelines=[
            dict(type='LoadImageFromFile'),
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
                mean=[0.0, 0.0, 0.0],
                std=[255.0, 255.0, 255.0],
                to_rgb=True,
            ),
            dict(type='Collect'),
        ],
    ))

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
