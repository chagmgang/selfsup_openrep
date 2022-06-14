train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='unchanged'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(1.0, 1.0)),
    dict(type='RandomCrop', crop_size=(1024, 1024)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=(-45, 45)),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=1024),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='unchanged'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=1.0,
        flip=False,
        transforms=[
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
    ),
]

train_dataset = [
    dict(
        type='CustomDataset',
        data_root='/nas/k8s/dev/mlops/chagmgang/dataset/inria/train',
        img_dir='images',
        ann_dir='gt',
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=train_pipeline,
    ),
]

data = dict(
    dist=True,
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CustomDataset',
        data_root='dataset',
        img_dir='images',
        ann_dir='gt',
        classes=['background', 'building'],
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=train_pipeline,
    ),
    val=dict(
        type='CustomDataset',
        data_root='dataset',
        img_dir='images',
        ann_dir='gt',
        classes=['background', 'building'],
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
    ),
    test=dict(),
)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SelfSupBackbone',
        model_name='ResNet50',
        pretrained=True,
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        ignore_index=255,
        loss_decode=[
            dict(type='CrossEntropyLoss'),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)),
)

optimizer = dict(type='SGD', lr=1e-3, weight_decay=1e-5)
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU')
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
