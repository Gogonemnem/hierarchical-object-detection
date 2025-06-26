_base_ = [
    './aircraft_flat_function.py',
]

# Modify metric related settings
train_dataloader = dict(
    dataset = dict(
        ann_file='aircraft_flat_function_train_excluded_F16.Mig29.B1.C130.TB2.AH64.json',
    )
)
