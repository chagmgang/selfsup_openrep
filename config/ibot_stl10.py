# runtime
# checkpoint saving
checkpoint_config = dict(interval=40)

# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable

# custom hook
custom_hooks = [
    dict(type='MomentumUpdateHook'),
    dict(type='iBOTTemperatureUpdateHook'),
    dict(
        type='LinearWeightDecayUpdateHook',
        start_weight_decay=0.04,
        end_weight_decay=0.4),
]

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
    warmup_iters=1,
    warmup_ratio=1e-4,  # cannot be 0
    warmup_by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=1600)

# data
global_pipelines = [
    dict(type='Resize', img_size=(224, 224), ratio_range=(0.5, 1.0)),
    dict(type='RandomCrop', crop_size=(224, 224)),
    dict(type='Pad', size_divisor=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=(-45, 45)),
    dict(type='PhotoMetricDistortion', prob=0.5),
    dict(type='GaussianBlur'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),
    dict(type='Collect'),
]

local_pipelines = [
    dict(type='Resize', img_size=(96, 96), ratio_range=(0.2, 1.0)),
    dict(type='RandomCrop', crop_size=(96, 96)),
    dict(type='Pad', size_divisor=96),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=(-45, 45)),
    dict(type='PhotoMetricDistortion', prob=0.5),
    dict(type='GaussianBlur'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),
    dict(type='Collect'),
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type='DinoDataset',
        datasource=dict(
            type='BaseDataSource',
            ann_file='stl10.txt',
        ),
        global_pipelines=global_pipelines,
        local_pipelines=local_pipelines,
        global_ncrop=2,
        local_ncrop=10,
    ),
)

# models
model = dict(
    type='iBOT',
    backbone=dict(
        type='TinyiBOTViT',
        img_size=224,
        patch_size=16,
    ),
    projection=dict(
        type='DinoHead',
        in_dim=192,
        out_dim=8196,
    ),
)
