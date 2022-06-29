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
            run_name='stl10-vit-mocov3',
        ),
    ])
# yapf:enable

# custom hooks
custom_hooks = [
    dict(type='MomentumUpdateHook'),
]

# runtime settings
dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
persistent_workers = True
fp16 = dict(loss_scale='dynamic')

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# schedule
# optimizer
optimizer = dict(type='AdamW', lr=1.5e-4, weight_decay=0.1, betas=(0.9, 0.999))
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=40,
    warmup_ratio=1e-4,  # cannot be 0
    warmup_by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=40000)

# data
train_pipeline = [
    dict(type='Resize', img_size=(96, 96), ratio_range=(0.5, 1.0)),
    dict(type='RandomCrop', crop_size=(96, 96)),
    dict(type='Pad', size_divisor=96),
    dict(type='PhotoMetricDistortion', prob=0.5),
    dict(type='RandomChannelShift'),
    dict(type='GaussianBlur'),
    dict(type='Solarization', prob=0.2),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),
    dict(type='Collect'),
]

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    train=dict(
        type='SimclrDataset',
        datasource=dict(
            type='BaseDataSource',
            ann_file='stl10.txt',
        ),
        pipelines=train_pipeline,
    ),
)

# models
model = dict(
    type='MocoV3',
    backbone=dict(
        type='SmallViT',
        image_size=96,
        patch_size=16,
    ),
    projection=dict(
        type='BatchNormProjection',
        num_layers=3,
        input_dim=384,
        hidden_dim=4096,
        last_dim=256,
    ),
    prediction=dict(
        type='BatchNormProjection',
        num_layers=2,
        input_dim=256,
        hidden_dim=4096,
        last_dim=256,
    ),
    start_momentum=0.99,
    temperature=0.2,
)
