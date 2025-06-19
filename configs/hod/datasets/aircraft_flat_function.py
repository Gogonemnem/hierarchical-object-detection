_base_ = [
    './aircraft_detection.py',
]

# Modify metric related settings
val_evaluator = dict(
    type='HierarchicalCocoMetric',
    ann_file=_base_.data_root + 'aircraft_hierarchy_function_validation.json',
    )
test_evaluator = dict(
    type='HierarchicalCocoMetric',
    ann_file=_base_.data_root + 'aircraft_hierarchy_function_test.json',
    )
