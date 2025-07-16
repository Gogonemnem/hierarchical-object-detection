_base_ = [
    '../../_base_/hierarchical-rtmdet_tiny.py',
    '../../datasets/rtmdet_aircraft_hierarchy_function.py',
    '../../schedules/schedule_8xb32-300e.py'
]

custom_imports = dict(imports=['hod.datasets', 'hod.evaluation', 'hod.models'], allow_failed_imports=False)

model = dict(
    bbox_head=dict(
        num_classes=111,
        ann_file=_base_.test_evaluator.ann_file,
        loss_cls=dict(
            ann_file=_base_.test_evaluator.ann_file,
            decay=10,
        )
    )
)
