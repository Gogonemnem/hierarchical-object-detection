_base_ = [
    './aircraft_detection.py',
]

# Modify metric related settings
val_dataloader = dict(
    dataset = dict(
        ann_file='aircraft_hierarchy_function_validation.json',
        # leaf classes obtain same ids
    )
)
test_dataloader = dict(
    dataset = dict(
        ann_file='aircraft_hierarchy_function_test.json',
    )
)
val_evaluator = dict(
    type='HierarchicalCocoMetric',
    ann_file=_base_.data_root + 'aircraft_hierarchy_function_validation.json',
    )
test_evaluator = dict(
    type='HierarchicalCocoMetric',
    ann_file=_base_.data_root + 'aircraft_hierarchy_function_test.json',
    )
