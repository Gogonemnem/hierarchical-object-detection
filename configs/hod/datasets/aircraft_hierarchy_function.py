_base_ = [
    './aircraft_detection.py',
]

# custom_imports = dict(imports=['hod.datasets', 'hod.evaluation'], allow_failed_imports=False)

train_dataloader = dict(
    dataset = dict(
        ann_file='hierarchy_function/aircraft_train.json',
    )
)
val_dataloader = dict(
    dataset = dict(
        ann_file='hierarchy_function/aircraft_validation.json',
    )
)
test_dataloader = dict(
    dataset = dict(
        ann_file='hierarchy_function/aircraft_test.json',
    )
)

# Modify metric related settings
hierarchical_data_root = 'data/aircraft/hierarchy_function/'
val_evaluator = dict(
    ann_file=hierarchical_data_root + 'aircraft_validation.json',
    )
test_evaluator = dict(
    ann_file=hierarchical_data_root + 'aircraft_test.json',
    )
