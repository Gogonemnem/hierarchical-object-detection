_base_ = [
    './dino-5scale_swin-l_8xb2'
]

model = dict(
    backbone=dict(
        frozen_stages=4,
    ),
)
