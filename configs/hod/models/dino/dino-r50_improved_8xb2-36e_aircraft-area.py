_base_ = [
    '../../_base_/dino-4scale_r50_improved.py',
    '../../datasets/aircraft_flat_area.py',
    '../../schedules/schedule_r50_improved_8xb2-36e.py'
]

custom_imports = dict(imports=['hod.datasets', 'hod.evaluation', 'hod.models'], allow_failed_imports=False)

model = dict(
    bbox_head=dict(
        num_classes=81,
    )
)
