_base_ = [
    '../dino/dino-4scale_r50_8xb2-36e_coco.py'
]

# learning policy
# max_epochs = 13
max_epochs = 24
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

# load_from = "https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_improved_8xb2-12e_coco/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth"
load_from = "/home/gonem/Documents/Projects/hierarchical-object-detection/work_dirs/frozen_dino/epoch_19.pth"
resume = True

model = dict(
    backbone=dict(
        frozen_stages=4,
    ),
    bbox_head=dict(
        num_classes=81,
    ),
)

# Modify dataset related settings
data_root = 'data/aircraft/'
metainfo = {
    'classes': (
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
    ),
    'taxonomy': {
        "Military Aircraft": {
            "Fixed-Wing Aircraft": {
                "Combat Aircraft": {
                    "Fighters and Multirole": {
                        "American Fighters": ["F-4", "F-14", "F-15", "F-16", "F/A-18", "F-117", "F-22", "F-35", "YF-23"],
                        "Russian Fighters": ["MiG-29", "MiG-31", "Su-57"],
                        "Western Fighters": ["JAS-39", "Mirage2000", "Rafale", "EF-2000", "Tornado"],
                        "Asian Fighters": ["J-10", "J-20", "J-35", "JF-17", "JH-7", "KF-21", "KAAN"]
                    },
                    "Attack Aircraft": ["A-10", "AV-8B", "EMB-314", "Su-25"],
                    "Bombers": {
                        "American Bombers": ["B-1", "B-2", "B-21", "B-52", "XB-70"],
                        "Russian Bombers": ["Tu-160", "Tu-22M", "Tu-95", "Su-24", "Su-34"],
                        "Chinese Bombers": ["H-6"],
                        "British Bombers": ["Vulcan"]
                    },
                    "Reconnaissance and Surveillance": {
                        "Airborne Early Warning": ["E-2", "E-7"],
                        "High Altitude Reconnaissance": ["SR-71", "U-2"],
                        "Maritime Patrol": ["P-3", "KJ-600", "WZ-7"]
                    }
                },
                "Transport and Utility": {
                    "Cargo Transports": {
                        "American Cargo Transports": ["C-130", "C-17", "C-2", "C-5", "KC-135"],
                        "European Cargo Transports": ["A-400M"],
                        "Latin American Cargo Transports": ["C-390"],
                        "Russian Cargo Transports": ["An-124", "An-22", "An-225", "An-72", "Il-76"],
                        "Chinese Cargo Transports": ["Y-20"]
                    },
                    "Amphibious Aircraft": ["US-2", "AG-600", "Be-200", "CL-415"]
                }
            },
            "Rotorcraft": {
                "Attack Helicopters": ["AH-64", "Ka-52", "Mi-24", "Mi-28", "Z-10"],
                "Utility and Transport Helicopters": {
                    "Land-Based Helicopters": ["CH-47", "Mi-8", "UH-60"],
                    "Naval and Specialized Helicopters": ["Ka-27", "Z-19"]
                }
            },
            "Unmanned Aerial Vehicles": {
                "Combat and Reconnaissance Drones": ["MQ-9", "RQ-4"],
                "Tactical UAVs": ["TB-001", "TB-2"]
            },
            "Tiltrotor Aircraft": ["V-22", "V-280"]
        }
    },
}
# metainfo = {
#     'classes': (
#         "airplane", "airport", "baseballfield", "basketballcourt", "bridge",
#         "chimney", "dam", "Expressway-Service-area", "Expressway-toll-station",
#         "groundtrackfield", "golffield", "harbor", "overpass", "ship", "stadium", 
#         "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill"
#     ),
#     'palette': [
#         (220, 20, 60), (0, 128, 0), (0, 0, 128), (255, 165, 0),
#         (255, 69, 0), (75, 0, 130), (255, 140, 0), (255, 105, 180),
#         (30, 144, 255), (255, 215, 0), (0, 191, 255), (0, 250, 154),
#         (85, 107, 47), (100, 149, 237), (72, 61, 139), (255, 99, 71),
#         (127, 255, 0), (0, 255, 127), (64, 224, 208), (240, 128, 128)
#     ]
# }

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='aircraft_train.json',
        data_prefix=dict(img='')
        )
    )
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='aircraft_validation.json',
        data_prefix=dict(img='')
        )
    )

test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='aircraft_test.json',
        data_prefix=dict(img='')
        )
    )

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'aircraft_validation.json')
test_evaluator = dict(ann_file=data_root + 'aircraft_test.json')

