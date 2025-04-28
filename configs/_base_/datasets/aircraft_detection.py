_base_ =[
    './coco_detection.py',
]
# dataset settings
# dataset_type = 'AircraftDataset'
data_root = 'data/aircraft/'

backend_args = None

classes = (
        "A10", "A400M", "AG600", "AH64", "AV8B", "An124", "An22", "An225",
        "An72", "B1", "B2", "B21", "B52", "Be200", "C130", "C17", "C2",
        "C390", "C5", "CH47", "CL415", "E2", "E7", "EF2000", "EMB314",
        "F117", "F14", "F15", "F16", "F18", "F22", "F35", "F4", "H6",
        "Il76", "J10", "J20", "J35", "JAS39", "JF17", "JH7", "KAAN", "KC135",
        "KF21", "KJ600", "Ka27", "Ka52", "MQ9", "Mi24", "Mi26", "Mi28",
        "Mi8", "Mig29", "Mig31", "Mirage2000", "P3", "RQ4", "Rafale", "SR71",
        "Su24", "Su25", "Su34", "Su57", "TB001", "TB2", "Tornado", "Tu160",
        "Tu22M", "Tu95", "U2", "UH60", "US2", "V22", "V280", "Vulcan", "WZ7",
        "XB70", "Y20", "YF23", "Z10", "Z19"
    )
taxonomy = {
    "Military Aircraft": {
        "Fixed-Wing Aircraft": {
            "Combat Aircraft": {
                "Fighters and Multirole": {
                    "American Fighters": {
                        "F117": {}, "F14": {}, "F15": {}, "F16": {}, "F18": {}, "F22": {}, "F35": {}, "F4": {}, "YF23": {}
                    },
                    "Russian Fighters": {
                        "Mig29": {}, "Mig31": {}, "Su57": {}
                    },
                    "Western Fighters": {
                        "JAS39": {}, "Mirage2000": {}, "Rafale": {}, "EF2000": {}, "Tornado": {}
                    },
                    "Asian Fighters": {
                        "J10": {}, "J20": {}, "J35": {}, "JF17": {}, "JH7": {}, "KF21": {}, "KAAN": {}
                    }
                },
                "Attack Aircraft": {
                    "A10": {}, "AV8B": {}, "EMB314": {}, "Su25": {}
                },
                "Bombers": {
                    "American Bombers": {
                        "B1": {}, "B2": {}, "B21": {}, "B52": {}, "XB70": {}
                    },
                    "Russian Bombers": {
                        "Tu160": {}, "Tu22M": {}, "Tu95": {}, "Su24": {}, "Su34": {}
                    },
                    "Chinese Bombers": {
                        "H6": {}
                    },
                    "British Bombers": {
                        "Vulcan": {}
                    }
                },
                "Reconnaissance and Surveillance": {
                    "Airborne Early Warning": {
                        "E2": {}, "E7": {}, "KJ600": {}
                    },
                    "High Altitude Reconnaissance": {
                        "SR71": {}, "U2": {}
                    },
                    "Maritime Patrol": {
                        "P3": {}, "WZ7": {}
                    }
                }
            },
            "Transport and Utility": {
                "Cargo Transports": {
                    "American Cargo Transports": {
                        "C130": {}, "C17": {}, "C2": {}, "C5": {}, "KC135": {}
                    },
                    "European Cargo Transports": {
                        "A400M": {}
                    },
                    "Latin American Cargo Transports": {
                        "C390": {}
                    },
                    "Russian Cargo Transports": {
                        "An124": {}, "An22": {}, "An225": {}, "An72": {}, "Il76": {}
                    },
                    "Chinese Cargo Transports": {
                        "Y20": {}
                    }
                },
                "Amphibious Aircraft": {
                    "AG600": {}, "Be200": {}, "CL415": {}, "US2": {}
                }
            }
        },
        "Rotorcraft": {
            "Attack Helicopters": {
                "AH64": {}, "Ka52": {}, "Mi24": {}, "Mi28": {}, "Z10": {}
            },
            "Utility and Transport Helicopters": {
                "Land-Based Helicopters": {
                    "CH47": {}, "Mi8": {}, "Mi26": {}, "UH60": {}
                },
                "Naval and Specialized Helicopters": {
                    "Ka27": {}, "Z19": {}
                }
            }
        },
        "Unmanned Aerial Vehicles": {
            "Combat and Reconnaissance Drones": {
                "MQ9": {}, "RQ4": {}
            },
            "Tactical UAVs": {
                "TB001": {}, "TB2": {}
            }
        },
        "Tiltrotor Aircraft": {
            "V22": {}, "V280": {}
        }
    }
}

palette = [
    (31, 119, 180),
    (174, 199, 232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (199, 199, 199),
    (188, 189, 34),
    (219, 219, 141),
    (23, 190, 207),
    (158, 218, 229),
    (31, 119, 180),
    (174, 199, 232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (199, 199, 199),
    (188, 189, 34),
    (219, 219, 141),
    (23, 190, 207),
    (158, 218, 229),
    (31, 119, 180),
    (174, 199, 232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (199, 199, 199),
    (188, 189, 34),
    (219, 219, 141),
    (23, 190, 207),
    (158, 218, 229),
    (31, 119, 180),
    (174, 199, 232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (199, 199, 199),
    (188, 189, 34),
    (219, 219, 141),
    (23, 190, 207),
    (158, 218, 229),
    (31, 119, 180)
]

metainfo = {
    'classes': classes,
    'taxonomy': taxonomy,
    'palette': palette
}

# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PackDetInputs')
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
#     # If you don't have a gt annotation, delete the pipeline
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]
# train_dataloader = dict(
#     batch_size=2,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     batch_sampler=dict(type='AspectRatioBatchSampler'),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='aircraft_train.json',
#         data_prefix=dict(img=''),
#         filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         pipeline=train_pipeline,
#         backend_args=backend_args))
train_dataloader = dict(
    dataset=dict(
        # type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='aircraft_train.json',
        data_prefix=dict(img='')))
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='aircraft_validation.json',
#         data_prefix=dict(img=''),
#         test_mode=True,
#         pipeline=test_pipeline,
#         backend_args=backend_args))
val_dataloader = dict(
    dataset=dict(
        # type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='aircraft_validation.json',
        data_prefix=dict(img='')))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'aircraft_validation.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'aircraft_test.json',
#         data_prefix=dict(img=''),
#         test_mode=True,
#         pipeline=test_pipeline,
#         backend_args=backend_args))
test_dataloader = dict(
    dataset=dict(
        # type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='aircraft_test.json',
        data_prefix=dict(img='')))
test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=True,
    ann_file=data_root + 'aircraft_test.json',
    outfile_prefix='./outputs/aircraft_detection/test',
    backend_args=backend_args)
