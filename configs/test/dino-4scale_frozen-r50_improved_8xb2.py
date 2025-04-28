_base_ = [
    './dino-4scale_r50_improved_8xb2.py'
]

model = dict(
    backbone=dict(
        frozen_stages=4,
    ),
)
