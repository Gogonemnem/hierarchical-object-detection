_base_ = [
    './dino-5scale_swin-l.py'
]

# data_root = path to the hierarchical dataset
cls_decay = 3

model = dict(
    bbox_head=dict(
        type='EmbeddingDINOHead',
        # num_classes=# classes in the hierarchical dataset including the parents
        # ann_file=path to the annotation file for hierarchical focal loss,
        outputs_per_query=1,  # -1 for all classes per query, or k for top-k per query
        cls_curvature=0.0,
        share_cls_layer=False,
        cls_config=dict(
            use_bias=True,
            use_temperature=True,
            init_norm_upper_offset=0.5,
            freeze_embeddings=False,
        ),
        # num_classes=# classes in the hierarchical dataset including the parents
        # ann_file=path to the annotation file for hierarchical focal loss,
        loss_cls=dict(
            type='HierarchicalFocalLoss',
            # ann_file=path to the annotation file for hierarchical focal loss,
            decay=cls_decay,
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
                    # ann_file= path to the annotation file for hierarchical focal loss,
                    decay=cls_decay,
                ),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
)

# Prototype Pre-training Configuration
prototype_pretrain_cfg = dict(
    enable=False,  # Set to False to disable pre-training
    epochs=100,       # Number of epochs for pre-training
    force_pretrain=False  # Set to True to always re-run pre-training even if a checkpoint exists
)
