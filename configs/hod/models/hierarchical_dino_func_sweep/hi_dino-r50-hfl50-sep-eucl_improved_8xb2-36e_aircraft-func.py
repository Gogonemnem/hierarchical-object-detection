_base_ = [
    '../../_base_/hierarchical-dino-4scale_r50_improved.py',
    '../../datasets/aircraft_hierarchy_function.py',
    '../../schedules/schedule_r50_improved_8xb2-36e.py'
]

custom_imports = dict(imports=['hod.datasets', 'hod.evaluation', 'hod.models'], allow_failed_imports=False)

model = dict(
    bbox_head=dict(
        num_classes=111,
        loss_cls=dict(
            ann_file=_base_.test_evaluator.ann_file,
            decay=50,
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
                    decay=50,
                ),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
)
