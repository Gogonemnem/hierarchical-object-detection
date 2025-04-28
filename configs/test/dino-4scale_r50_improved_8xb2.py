_base_ = [
    '../_base_/datasets/military_aircrafts_detection.py', '../dino/dino-4scale_r50_8xb2-36e_coco.py'
]

custom_imports = dict(imports=['hod.evaluation', 'hod.models'], allow_failed_imports=False)

# learning policy
max_epochs = 36
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

load_from = "https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_improved_8xb2-12e_coco/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth"
# load_from = "work_dirs/dino-4scale_r50_improved_8xb2/epoch_12.pth"
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
    batch_size=3,
    )
val_dataloader = dict(
    batch_size=10,
    )
test_dataloader = dict(
    batch_size=6,
    )
