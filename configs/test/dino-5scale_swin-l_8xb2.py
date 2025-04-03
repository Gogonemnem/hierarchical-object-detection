_base_ = [
    '../dino/dino-5scale_swin-l_8xb2-36e_coco.py'
]

custom_imports = dict(imports=['hod.evaluation'], allow_failed_imports=False)

# learning policy
# max_epochs = 13
max_epochs = 48
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

load_from = "https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"
# load_from = "/home/gonem/Documents/Projects/hierarchical-object-detection/work_dirs/dino/epoch_24.pth"
resume = True

model = dict(
    # backbone=dict(
    #     frozen_stages=4,
    # ),
    bbox_head=dict(
        num_classes=81,
    ),
)


import matplotlib.pyplot as plt

def generate_palette(n):
    cmap = plt.get_cmap("tab20")  # 20-color map for high contrast
    colors = []
    for i in range(n):
        color = cmap(i % 20)  # cycle if more than 20
        rgb = tuple(int(255 * c) for c in color[:3])
        colors.append(rgb)
    return colors

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
                        "American Fighters": ["F117", "F14", "F15", "F16", "F18", "F22", "F35", "F4", "YF23"],
                        "Russian Fighters": ["Mig29", "Mig31", "Su57"],
                        "Western Fighters": ["JAS39", "Mirage2000", "Rafale", "EF2000", "Tornado"],
                        "Asian Fighters": ["J10", "J20", "J35", "JF17", "JH7", "KF21", "KAAN"]
                    },
                    "Attack Aircraft": ["A10", "AV8B", "EMB314", "Su25"],
                    "Bombers": {
                        "American Bombers": ["B1", "B2", "B21", "B52", "XB70"],
                        "Russian Bombers": ["Tu160", "Tu22M", "Tu95", "Su24", "Su34"],
                        "Chinese Bombers": ["H6"],
                        "British Bombers": ["Vulcan"]
                    },
                    "Reconnaissance and Surveillance": {
                        "Airborne Early Warning": ["E2", "E7", "KJ600"],
                        "High Altitude Reconnaissance": ["SR71", "U2"],
                        "Maritime Patrol": ["P3", "WZ7"]
                    }
                },
                "Transport and Utility": {
                    "Cargo Transports": {
                        "American Cargo Transports": ["C130", "C17", "C2", "C5", "KC135"],
                        "European Cargo Transports": ["A400M"],
                        "Latin American Cargo Transports": ["C390"],
                        "Russian Cargo Transports": ["An124", "An22", "An225", "An72", "Il76"],
                        "Chinese Cargo Transports": ["Y20"]
                    },
                    "Amphibious Aircraft": ["AG600", "Be200", "CL415", "US2"]
                }
            },
            "Rotorcraft": {
                "Attack Helicopters": ["AH64", "Ka52", "Mi24", "Mi28", "Z10"],
                "Utility and Transport Helicopters": {
                    "Land-Based Helicopters": ["CH47", "Mi8", "Mi26", "UH60"],
                    "Naval and Specialized Helicopters": ["Ka27", "Z19"]
                }
            },
            "Unmanned Aerial Vehicles": {
                "Combat and Reconnaissance Drones": ["MQ9", "RQ4"],
                "Tactical UAVs": ["TB001", "TB2"]
            },
            "Tiltrotor Aircraft": ["V22", "V280"]
        }
    },
    'palette': generate_palette(81),
}

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
    batch_size=6,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='aircraft_test.json',
        data_prefix=dict(img='')
        )
    )

# Modify metric related settings
val_evaluator = dict(
    type='CocoMetric', # 'PRFMetric',
    ann_file=data_root + 'aircraft_validation.json' # for CocoMetric
    )
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'aircraft_test.json', # for CocoMetric
    outfile_prefix='./outputs/results/test'
    )

