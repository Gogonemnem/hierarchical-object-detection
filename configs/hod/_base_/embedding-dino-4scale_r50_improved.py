_base_ = [
    './dino-4scale_r50_improved.py'
]

# custom_imports = dict(imports=['hod.datasets', 'hod.evaluation', 'hod.models'], allow_failed_imports=False)

model = dict(
    bbox_head=dict(
        type='EmbeddingDINOHead',
        # num_classes=# classes in the dataset excluding the parents
        # ann_file=path to the annotation file for hierarchical focal loss,
        cls_curvature=0.0,
        share_cls_layer=False,
        cls_config=dict(
            use_bias=True,
            use_temperature=True,
            init_norm_upper_offset=0.5,
            freeze_embeddings=False,
        ),
    ),
)
