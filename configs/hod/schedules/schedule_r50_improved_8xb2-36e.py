# training schedule for 36e
_base_ = [
    './schedule_r50_8xb2-36e.py',
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.0002),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
