_base_ = [
    '../embedding_dino/hierarchical_models/dino-4scale_r50_improved_8xb2_hier-aircraft.py',
]
# learning policy
max_epochs = 1
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

# Define shared hyperparameter samplers
hpo_shared_params = dict(
    # Shared decay for various hierarchical losses/costs
    shared_decay=dict(type='RandInt', lower=1, upper=5),

    # Shared loss_weight for the chosen loss_embed option
    shared_embed_loss_weight=dict(type='LogUniform', lower=0.1, upper=5.0)
)

# load_from = "work_dirs/frozen-4scale_r50_improved_8xb2/epoch_19.pth"

data_root = 'data/aircraft/hierarchical/'
model = dict(
    bbox_head=dict(
        type='EmbeddingDINOHead',
        cls_curvature=dict(
            _delete_=True,
            # type='Uniform', lower=-2.0, upper=-0.5),
            type='Choice', options=[-1.0, 0.0]),
        share_cls_layer=dict(
            _delete_=True,
            type='Choice', options=[True, False]),
        freeze_cls_embeddings=dict(
            _delete_=True,
            type='Choice', options=[True, False]),
        loss_embed=dict(
            _delete_=True,
            type='Choice',
            options=[
                None,
                dict(
                    type='EntailmentConeLoss',
                    beta=dict(type='LogUniform', lower=0.01, upper=1.0),
                    init_norm_upper_offset=0.5,
                    loss_weight="hpo_ref:shared_embed_loss_weight"
                ),
                dict(
                    type='HierarchicalContrastiveLoss',
                    aggregate_per=dict(type='Choice', options=['node', 'depth', None]),
                    decay="hpo_ref:shared_decay",
                    loss_weight="hpo_ref:shared_embed_loss_weight"
                ),
            ]
        ),
        loss_cls=dict(
            type='HierarchicalFocalLoss',
            ann_file=data_root + 'aircraft_test.json',
            decay="hpo_ref:shared_decay"
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
                    decay="hpo_ref:shared_decay"
                ),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
)

train_dataloader = dict(
    batch_size=1,
)
val_dataloader = dict(
    batch_size=1,
)
test_dataloader = dict(
    batch_size=1,
)

# Prototype Pre-training Configuration
prototype_pretrain_cfg = dict(
    enable=dict(
        _delete_=True,
        type='Choice', options=[True, False]),  # Set to False to disable pre-training
    epochs=100,           # Number of epochs for pre-training
    force_pretrain=False  # Set to True to always re-run pre-training even if a checkpoint exists
)
