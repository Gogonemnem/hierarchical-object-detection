# Swin-L backbone + Hierarchical Focal Loss (func sweep, decay=10).
#
# Base layout is a SINGLE model base (dino-5scale_swin-l.py) plus the dataset
# and schedule bases. The hierarchical overrides (EmbeddingDINOHead,
# HierarchicalFocalLoss, hierarchical Hungarian assigner) are inlined below
# instead of living in a sibling base config, to avoid the mmengine
# "Duplicate key is not allowed among bases" error that occurs when two
# direct bases both inherit default_runtime.py through their dino chains.
_base_ = [
    '../../_base_/dino-5scale_swin-l.py',
    '../../datasets/aircraft_hierarchy_function.py',
    '../../schedules/schedule_swin-l_8xb2-36e.py'
]

custom_imports = dict(imports=['hod.datasets', 'hod.evaluation', 'hod.models'], allow_failed_imports=False)

model = dict(
    bbox_head=dict(
        # Embedding head for the learned class prototypes (sep / euclidean).
        type='EmbeddingDINOHead',
        # num_classes=# classes in the hierarchical dataset including the parents
        num_classes=145,
        # ann_file=path to the annotation file for hierarchical focal loss
        ann_file=_base_.test_evaluator.ann_file,
        outputs_per_query=1,  # -1 for all classes per query, or k for top-k per query
        cls_curvature=0.0,
        share_cls_layer=False,
        cls_config=dict(
            use_bias=True,
            use_temperature=True,
            init_norm_upper_offset=0.5,
            freeze_embeddings=False,
        ),
        # loss_cls: hierarchical focal loss (decay=10 for this sweep member)
        loss_cls=dict(
            type='HierarchicalFocalLoss',
            ann_file=_base_.test_evaluator.ann_file,
            decay=10,
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
                    ann_file=_base_.test_evaluator.ann_file,
                    decay=10,
                ),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
)

# Prototype Pre-training Configuration
prototype_pretrain_cfg = dict(
    enable=False,
    epochs=100,
    force_pretrain=False
)
