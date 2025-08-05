_base_ = [
    './rtmdet_tiny.py'
]

model = dict(
    bbox_head=dict(
        type='EmbeddingRTMDetSepBNHead',
        # num_classes=# classes in the dataset excluding the parents
        # ann_file=path to the annotation file for hierarchical focal loss,
        cls_curvature=0.0,
        cls_config=dict(
            use_bias=True,
            use_temperature=True,
            init_norm_upper_offset=0.5,
            freeze_embeddings=False,
        ),
    ),
)
