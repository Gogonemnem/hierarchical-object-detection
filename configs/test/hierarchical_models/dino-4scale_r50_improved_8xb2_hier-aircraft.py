_base_ = [
    '../datasets/aircraft_detection.py',
    '../../dino/models/dino-4scale_r50_improved_8xb2-12e.py'
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
data_root = 'data/aircraft/hierarchical/'
model = dict(
    bbox_head=dict(
        type='EmbeddingDINOHead',
        num_classes=116,
        ann_file=data_root + 'aircraft_test.json',
        cls_curvature=-1.0,
        cls_cone_beta=0.1,
        cls_init_norm_upper_offset=0.5,
        # loss_embed=dict(
        #              type='EntailmentConeLoss',
        #              loss_weight=1.0),
        loss_cls=dict(
            type='HierarchicalFocalLoss',
            ann_file=data_root + 'aircraft_test.json',
            decay=2,)
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(
                    type='HierarchicalFocalLossCost',
                    weight=2.0,
                    ann_file=data_root + 'aircraft_test.json',
                    decay=2,
                ),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
)

train_dataloader = dict(
    batch_size=2,
    dataset = dict(
        ann_file='hierarchical/aircraft_train.json',
    )
)
val_dataloader = dict(
    batch_size=10,
    dataset = dict(
        ann_file='hierarchical/aircraft_validation.json',
    )
)
test_dataloader = dict(
    batch_size=1,
    dataset = dict(
        ann_file='hierarchical/aircraft_test.json',
    )
)

# Modify metric related settings
val_evaluator = dict(
    type='HierarchicalCocoMetric',
    ann_file=data_root + 'aircraft_validation.json',
    )
test_evaluator = dict(
    type='HierarchicalCocoMetric',
    ann_file=data_root + 'aircraft_test.json',
    format_only=False,
    )
