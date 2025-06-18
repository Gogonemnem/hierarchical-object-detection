# training schedule for 36e
_base_ = [
    './schedule_r50_8xb2-36e.py',
]

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]
