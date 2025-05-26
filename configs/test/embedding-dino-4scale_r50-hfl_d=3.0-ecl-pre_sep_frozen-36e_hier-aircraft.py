_base_ = [
    '../embedding_dino/hierarchical_models/dino-4scale_r50_improved_8xb2_hier-aircraft.py',
]
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
custom_imports = dict(imports=['hod.datasets', 'hod.evaluation', 'hod.models'], allow_failed_imports=False)

# load_from = "work_dirs/frozen-4scale_r50_improved_8xb2/epoch_19.pth"

data_root = 'data/aircraft/hierarchical/'
model = dict(
    bbox_head=dict(
        type='EmbeddingDINOHead',
        cls_curvature=0.0,
        share_cls_layer=False,
        freeze_cls_embeddings=True,
        loss_embed=dict(
            type='EntailmentConeLoss',
        ),
        loss_cls=dict(
            type='HierarchicalFocalLoss',
            ann_file=data_root + 'aircraft_test.json',
            decay=3.0,
        )
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
                    decay=3.0
                ),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
)

# Prototype Pre-training Configuration
prototype_pretrain_cfg = dict(
    enable=True,          # Set to False to disable pre-training
    epochs=1000,          # Number of epochs for pre-training
    force_pretrain=False  # Set to True to always re-run pre-training even if a checkpoint exists
)
