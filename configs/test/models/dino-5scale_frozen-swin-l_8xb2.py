_base_ = [
    './datasets/aircraft_detection.py',
    '../dino/models/dino-5scale_swin-l_8xb2-36e.py'
]

custom_imports = dict(imports=['hod.datasets', 'hod.evaluation', 'hod.models'], allow_failed_imports=False)

# learning policy
max_epochs = 36
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

load_from = "https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"
# load_from = "work_dirs/dino-5scale_swin-l_8xb2/epoch_12.pth"
resume = False

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True
)

model = dict(
    backbone=dict(
        frozen_stages=4,
    ),
    bbox_head=dict(
        type='EmbeddingDINOHead',
        num_classes=81,
    ),
)


train_dataloader = dict(
    batch_size=1,
)
val_dataloader = dict(
    batch_size=1,
)

test_dataloader = dict(
    batch_size=10,
)

# Modify metric related settings
val_evaluator = dict(
    type='HierarchicalCocoMetric',
    )
test_evaluator = dict(
    type='HierarchicalCocoMetric',
    )
