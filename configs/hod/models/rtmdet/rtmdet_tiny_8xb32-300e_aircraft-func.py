_base_ = [
    '../../_base_/rtmdet_tiny.py',
    '../../datasets/rtmdet_aircraft_flat_function.py',
    '../../schedules/schedule_8xb32-300e.py'
]

custom_imports = dict(imports=['hod.datasets', 'hod.evaluation', 'hod.models'], allow_failed_imports=False)

model = dict(
    bbox_head=dict(
        num_classes=81,
    )
)
