_base_ = [
    '../../projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_coco.py'
]

model = dict(
    backbone=dict(
        frozen_stages=4
    )
)

train_dataloader = dict(
    batch_size=1
)
