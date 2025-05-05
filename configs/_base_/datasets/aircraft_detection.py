_base_ =[
    './coco_detection.py',
]

# dataset settings
dataset_type = 'AircraftDataset'
data_root = 'data/aircraft/'

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # metainfo=metainfo,
        metainfo=dict(),
        ann_file='aircraft_train.json',
        data_prefix=dict(img='')))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # metainfo=metainfo,
        metainfo=dict(),
        ann_file='aircraft_validation.json',
        data_prefix=dict(img='')))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'aircraft_validation.json',
    metric='bbox',
    format_only=False,)


test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # metainfo=metainfo,
        metainfo=dict(),
        ann_file='aircraft_test.json',
        data_prefix=dict(img='')))

test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=True,
    ann_file=data_root + 'aircraft_test.json',
    outfile_prefix='./outputs/aircraft_detection/test',)
