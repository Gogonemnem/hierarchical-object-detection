_base_ = [
    './aircraft_flat_area.py',
]

train_dataloader = dict(
    dataset = dict(
        ann_file='aircraft_hierarchy_area_train.json',
    )
)
val_dataloader = dict(
    dataset = dict(
        ann_file='aircraft_hierarchy_area_validation.json',
    )
)
test_dataloader = dict(
    dataset = dict(
        ann_file='aircraft_hierarchy_area_test.json',
    )
)
