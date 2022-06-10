model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SelfSupBackbone',
        model_name='ResNet50',
        pretrained=None,
        weight='/nas/k8s/dev/mlops/chagmgang/checkpoint/test_selfsup/40000.pth',
    ),
    neck=dict(type='GlobalAveragePooling', ),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=2048,
        loss=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
        topk=(1, 5),
    ),
)

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type='TrainTorchvisionSTL10',
        data_prefix=None,
        pipeline=[
            dict(type='RandomResizedCrop', size=96),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label']),
        ],
    ),
    val=dict(
        type='TestTorchvisionSTL10',
        data_prefix=None,
        pipeline=[
            dict(type='Resize', size=(96, -1)),
            dict(type='CenterCrop', crop_size=96),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ],
    ),
)

evaluation = dict(interval=1000, metric='accuracy')
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    by_epoch=False,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333,
    step=[3000, 5000],
)
runner = dict(type='IterBasedRunner', max_iters=8000)
checkpoint_config = dict(interval=2000)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ],
)
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
load_from = None
workflow = [('train', 1)]
