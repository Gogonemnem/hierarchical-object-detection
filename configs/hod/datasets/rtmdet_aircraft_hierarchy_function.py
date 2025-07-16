_base_ = [
    './rtmdet_aircraft_flat_function.py',
]

train_dataloader = dict(
    dataset = dict(
        ann_file='aircraft_hierarchy_function_train.json',
    )
)
