_base_ = [
    './datasets/aircraft_detection.py',
    '../dino/models/dino-4scale_r50_improved_8xb2-12e.py'
]

custom_imports = dict(imports=['hod.evaluation', 'hod.models'], allow_failed_imports=False)

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
        milestones=[30],
        gamma=0.1)
]

load_from = "https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_improved_8xb2-12e_coco/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth"
# load_from = "work_dirs/frozen-4scale_r50_improved_8xb2/epoch_19.pth"
resume = False

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True
)

model = dict(
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
    taxonomy=_base_.metainfo['taxonomy'],
    )
test_evaluator = dict(
    type='HierarchicalCocoMetric',
    taxonomy=_base_.metainfo['taxonomy'],
    )
