_base_ = [
    '../hierarchical_models/dino-4scale_r50_improved_8xb2_hier-aircraft.py',
]

custom_imports = dict(imports=['hod.datasets', 'hod.evaluation', 'hod.models'], allow_failed_imports=False)

# load_from = "work_dirs/frozen-4scale_r50_improved_8xb2/epoch_19.pth"

data_root = 'data/aircraft/hierarchical/'
model = dict(
    bbox_head=dict(
        type='EmbeddingDINOHead',
        cls_curvature=-1.0,
        share_cls_layer=True,
        freeze_cls_embeddings=False,
        loss_embed=dict(
            type='HierarchicalContrastiveLoss',
            aggregate_per='depth',
            decay=3,
            loss_weight=1.0,
            ),
        loss_cls=dict(
            type='HierarchicalFocalLoss',
            decay=3,)
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
                    decay=3,
                ),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
)

train_dataloader = dict(
    batch_size=2,
)
val_dataloader = dict(
    batch_size=10,
)
test_dataloader = dict(
    batch_size=1,
)
