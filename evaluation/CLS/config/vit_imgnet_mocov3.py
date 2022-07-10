dataset_type = 'ImageNet'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label']),
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='/nas/Dataset/ILSVRC2012/train',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_prefix='/nas/Dataset/ILSVRC2012/val',
        pipeline=test_pipeline,
    ),
)

evaluation = dict(interval=1, metric='accuracy')

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SelfSupViT',
        model_name='SmallMocoV3ViT',
        img_size=224,
        patch_size=16,
        weight='/tmp/tmpxoldr3ps/checkpoint/epoch_300.pth',
        unfreeze_patch=True,
    ),
    neck=dict(type='GlobalAveragePooling', ),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=384,
        loss=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
        topk=(1, 5),
    ),
)

optimizer = dict(
    type='AdamW',
    lr=5e-4,
    weight_decay=0.03,
)

optimizer_config = dict(grad_clip=dict(max_norm=1.0))

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=10000,
    warmup_ratio=1e-4,
)

runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=100, hooks=[
        dict(type='TextLoggerHook'),
    ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
