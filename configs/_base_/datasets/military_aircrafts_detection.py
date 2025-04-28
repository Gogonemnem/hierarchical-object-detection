# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/aircraft/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
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
import matplotlib.pyplot as plt

def generate_palette(n):
    cmap = plt.get_cmap("tab20")  # 20-color map for high contrast
    colors = []
    for i in range(n):
        color = cmap(i % 20)  # cycle if more than 20
        rgb = tuple(int(255 * c) for c in color[:3])
        colors.append(rgb)
    return colors
metainfo = {
    'classes': classes,
    'taxonomy': taxonomy,
    'palette': generate_palette(81),
}
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='aircraft_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='aircraft_validation.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader
test_dataloader.update(dict(
    dataset=dict(
        ann_file=data_root + 'aircraft_test.json',
        data_prefix=dict(img=''),
    )
))

val_evaluator = dict(
    type='HierarchicalCocoMetric',
    taxonomy=metainfo['taxonomy'],
    ann_file=data_root + 'aircraft_validation.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator
test_evaluator.update(dict(
    format_only=True,
    ann_file=data_root + 'aircraft_test.json',
    outfile_prefix='./outputs/results/test',
))
