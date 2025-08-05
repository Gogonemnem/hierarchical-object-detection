_base_ = [
    './rtmdet_tiny.py'
]

# data_root = path to the hierarchical dataset
cls_decay = 3
model = dict(
    bbox_head=dict(
        type='EmbeddingRTMDetSepBNHead',
        # num_classes=# classes in the hierarchical dataset including the parents
        # ann_file=path to the annotation file for hierarchical focal loss,
        # cls_curvature=0.0,
        # cls_config=dict(
        #     use_bias=True,
        #     use_temperature=True,
        #     init_norm_upper_offset=0.5,
        #     freeze_embeddings=False,
        # ),
        loss_cls=dict(
            type='HierarchicalQualityFocalLoss',
            # ann_file=path to the annotation file for hierarchical focal loss,
            decay=cls_decay,
        )
    ),
)

# Prototype Pre-training Configuration
prototype_pretrain_cfg = dict(
    enable=False,  # Set to False to disable pre-training
    epochs=100,       # Number of epochs for pre-training
    force_pretrain=False  # Set to True to always re-run pre-training even if a checkpoint exists
)
