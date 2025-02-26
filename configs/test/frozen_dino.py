_base_ = [
    '../dino/dino-4scale_r50_8xb2-36e_coco.py'
]

# learning policy
max_epochs = 13
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

load_from = "https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_improved_8xb2-12e_coco/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth"
resume = True

model = dict(
    backbone=dict(
        frozen_stages=4,
    ),
    bbox_head=dict(
        num_classes=20,
    ),
)

# Modify dataset related settings
data_root = 'data/dior/'
metainfo = {
    'classes': (
        "airplane", "airport", "baseballfield", "basketballcourt", "bridge",
        "chimney", "dam", "Expressway-Service-area", "Expressway-toll-station",
        "groundtrackfield", "golffield", "harbor", "overpass", "ship", "stadium", 
        "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill"
    ),
    'palette': [
        (220, 20, 60), (0, 128, 0), (0, 0, 128), (255, 165, 0),
        (255, 69, 0), (75, 0, 130), (255, 140, 0), (255, 105, 180),
        (30, 144, 255), (255, 215, 0), (0, 191, 255), (0, 250, 154),
        (85, 107, 47), (100, 149, 237), (72, 61, 139), (255, 99, 71),
        (127, 255, 0), (0, 255, 127), (64, 224, 208), (240, 128, 128)
    ]
}

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/dior_train.json',
        data_prefix=dict(img='')
        )
    )
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/dior_val.json',
        data_prefix=dict(img='')
        )
    )
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'annotations/dior_val.json')
test_evaluator = val_evaluator

