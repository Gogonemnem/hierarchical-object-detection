_base_ = [
    './aircraft_flat_function.py',
]

train_dataloader = dict(
    dataset = dict(
        ann_file='aircraft_hierarchy_function_train.json',
    )
)
val_dataloader = dict(
    dataset = dict(
        ann_file='aircraft_hierarchy_function_validation.json',
    )
)
test_dataloader = dict(
    dataset = dict(
        ann_file='aircraft_hierarchy_function_test.json',
    )
)
