import os.path as osp
import csv
import argparse
from collections import defaultdict

from mmengine.fileio import dump
from mmengine.utils import mkdir_or_exist

def parse_csv(csv_file):
    """Parse the CSV file into a list of annotation dictionaries."""
    rows = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row['width'] = int(row['width'])
            row['height'] = int(row['height'])
            row['xmin'] = int(row['xmin'])
            row['ymin'] = int(row['ymin'])
            row['xmax'] = int(row['xmax'])
            row['ymax'] = int(row['ymax'])
            rows.append(row)
    return rows

def group_annotations_by_image(rows):
    """
    Group annotations by split and filename.
    Returns a dict of dicts:
      { split: { filename: {'width': int, 'height': int, 'annotations': [row, ...]} } }
    """
    split_to_images = defaultdict(dict)
    for row in rows:
        split = row['split']
        fname = 'dataset/' + row['filename'] + '.jpg'
        if fname not in split_to_images[split]:
            split_to_images[split][fname] = {
                'width': row['width'],
                'height': row['height'],
                'annotations': []
            }
        split_to_images[split][fname]['annotations'].append(row)
    return split_to_images

def flatten_taxonomy(taxonomy, parent=None, out=None):
    """Walk the nested taxonomy, collect (node_name, parent_name) pairs."""
    if out is None:
        out = []
    for node, children in taxonomy.items():
        out.append((node, parent or "none"))
        # recurse
        flatten_taxonomy(children, node, out)
    return out

def cvt_to_coco_json(split_to_images, taxonomy, subtract_one=True, use_all_nodes=False):
    """
    Convert grouped annotations into COCO format.
    If subtract_one is True, subtract 1 from bounding box coordinates.
    Returns a dictionary mapping each split (e.g., 'train') to its COCO JSON dict.
    """
    flat = flatten_taxonomy(taxonomy)
    supers = {n: p for n, p in flat}

    # Create a canonical category mapping.
    # Start with leaf nodes from the data, sorted alphabetically.
    categories_set = set()
    for images in split_to_images.values():
        for info in images.values():
            for ann in info['annotations']:
                categories_set.add(ann['class'])
    categories_list = sorted(list(categories_set))
    label_ids = {cat: idx for idx, cat in enumerate(categories_list)}

    # If using all nodes, append parent/intermediate nodes from the taxonomy
    # that are not already present. This ensures leaf nodes always have the
    # same, lower-numbered IDs.
    if use_all_nodes:
        all_taxonomy_nodes = [n for n, p in flat]
        for node_name in all_taxonomy_nodes:
            if node_name not in label_ids:
                label_ids[node_name] = len(categories_list)
                categories_list.append(node_name)

    # Create the final categories list for the COCO JSON.
    coco_cats = [{
        "id": label_ids[name],
        "name": name,
        "supercategory": supers.get(name, "none")
    } for name in categories_list]

    # Prepare COCO JSONs per split
    coco_files = {}
    for split, images in split_to_images.items():
        coco = {
            'images': [],
            'annotations': [],
            'categories': coco_cats,
            'type': 'instance',
            # Add taxonomy metadata here:
            'taxonomy': taxonomy
        }
        image_id = 0
        ann_id = 0
        for fname, info in images.items():
            # Create image item
            image_item = {
                'id': image_id,
                'file_name': fname,
                'width': info['width'],
                'height': info['height']
            }
            coco['images'].append(image_item)
            # Process each annotation for the image
            for ann in info['annotations']:
                xmin = ann['xmin'] - 1 if subtract_one else ann['xmin']
                ymin = ann['ymin'] - 1 if subtract_one else ann['ymin']
                xmax = ann['xmax'] - 1 if subtract_one else ann['xmax']
                ymax = ann['ymax'] - 1 if subtract_one else ann['ymax']
                bbox_width = xmax - xmin
                bbox_height = ymax - ymin
                bbox = [xmin, ymin, bbox_width, bbox_height]
                area = bbox_width * bbox_height
                # Create a simple rectangle segmentation
                segmentation = [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]]
                coco_ann = {
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': label_ids.get(ann['class']),
                    'bbox': bbox,
                    'area': area,
                    'segmentation': segmentation,
                    'iscrowd': 0,
                    'ignore': 0
                }
                if coco_ann['category_id'] is not None:
                    coco['annotations'].append(coco_ann)
                ann_id += 1
            image_id += 1
        coco_files[split] = coco
    return coco_files

def main():
    parser = argparse.ArgumentParser(
        description='Convert aircraft CSV annotations to COCO JSON format')
    parser.add_argument(
        '--csv-file',
        type=str,
        required=True,
        help='Path to the CSV file (e.g., labels_with_split.csv)')
    parser.add_argument(
        '--out-dir',
        type=str,
        required=True,
        help='Directory to save the output COCO JSON files')
    parser.add_argument(
        '--subtract-one',
        action='store_true',
        help='Subtract 1 from bounding box coordinates (default: False)')
    args = parser.parse_args()

    # Parse CSV and group annotations by image and split.
    rows = parse_csv(args.csv_file)
    split_to_images = group_annotations_by_image(rows)

    mkdir_or_exist(args.out_dir)

    taxonomy_function = {
        "Military Aircraft": {
            "Fixed-Wing": {
                "Combat": {
                    "Fighters": {
                        "US Fighters": {
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
                        "US Bombers": {
                            "B1": {}, "B2": {}, "B21": {}, "B52": {}, "XB70": {}
                        },
                        "Russian Bombers": {
                            "Tu160": {}, "Tu22M": {}, "Tu95": {}, "Su24": {}, "Su34": {}
                        },
                        "H6": {},
                        "Vulcan": {}
                    },
                    "Surveillance": {
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
                "Utility": {
                    "Cargo Transports": {
                        "American Cargo": {
                            "C130": {}, "C17": {}, "C2": {}, "C5": {}, "KC135": {}
                        },
                        "Russian Cargo": {
                            "An124": {}, "An22": {}, "An225": {}, "An72": {}, "Il76": {}
                        },
                        "A400M": {},
                        "C390": {},
                        "Y20": {}
                    },
                    "Amphibious Aircraft": {
                        "AG600": {}, "Be200": {}, "CL415": {}, "US2": {}
                    }
                },
                "UAV": {
                    "Combat and Reconnaissance Drones": {
                        "MQ9": {}, "RQ4": {}
                    },
                    "Tactical UAVs": {
                        "TB001": {}, "TB2": {}
                    }
                },
            },
            "Rotorcraft": {
                "Attack Helicopters": {
                    "AH64": {}, "Ka52": {}, "Mi24": {}, "Mi28": {}, "Z10": {}
                },
                "Utility Helicopters": {
                    "Land-Based Helicopters": {
                        "CH47": {}, "Mi8": {}, "Mi26": {}, "UH60": {}
                    },
                    "Specialized Helicopters": {
                        "Ka27": {}, "Z19": {}
                    }
                }
            },
            "Tiltrotor": {
                "V22": {}, "V280": {}
            }
        }
    }

    taxonomy_area = {
        "Military Aircraft": {
            "US Aircraft": {
                "Fixed-Wing": {
                    "Combat": {
                        "Fighters": {"F117": {}, "F14": {}, "F15": {}, "F16": {}, "F18": {}, "F22": {}, "F35": {}, "F4": {}, "YF23": {}},
                        "Attack Aircraft": {"A10": {}, "AV8B": {}},
                        "Bombers": {"B1": {}, "B2": {}, "B21": {}, "B52": {}, "XB70": {}},
                        "Surveillance": {
                            "E2": {},
                            "High Altitude Reconnaissance": {"SR71": {}, "U2": {}},
                            "P3": {}
                        }
                    },
                    "Cargo Transports": {"C130": {}, "C17": {}, "C2": {}, "C5": {}, "KC135": {}},
                    "US2": {},
                    "UAV": {"MQ9": {}, "RQ4": {}}
                },
                "Rotorcraft": {
                    "AH64": {},
                    "Utility Helicopters": {"CH47": {}, "UH60": {}}
                },
                "Tiltrotor": {"V22": {}, "V280": {}}
            },
            "Russian Aircraft": {
                "Fixed-Wing": {
                    "Combat": {
                        "Fighters": {"Mig29": {}, "Mig31": {}, "Su57": {}},
                        "Su25": {},
                        "Bombers": {"Tu160": {}, "Tu22M": {}, "Tu95": {}, "Su24": {}, "Su34": {}}
                    },
                    "Cargo Transports": {"An124": {}, "An22": {}, "An225": {}, "An72": {}, "Il76": {}},
                    "Be200": {}
                },
                "Rotorcraft": {
                    "Attack Helicopters": {"Ka52": {}, "Mi24": {}, "Mi28": {}},
                    "Utility Helicopters": {"Mi8": {}, "Mi26": {}, "Ka27": {}}
                }
            },
            "European Aircraft": {
                "Fixed-Wing": {
                    "Fighters": {"JAS39": {}, "Mirage2000": {}, "Rafale": {}, "EF2000": {}, "Tornado": {}},
                    "Vulcan": {},
                    "Utility": {
                        "A400M": {},
                        "CL415": {}
                    },
                    "TB2": {}
                }
            },
            "Asian Aircraft": {
                "Fixed-Wing": {
                    "Combat": {
                        "Fighters": {"J10": {}, "J20": {}, "J35": {}, "JF17": {}, "JH7": {}, "KF21": {}, "KAAN": {}},
                        "Surveillance": {
                            "KJ600": {},
                            "WZ7": {}
                        }
                    },
                    "Utility": {
                        "Y20": {},
                        "AG600": {}
                    },
                    "TB001": {}
                },
                "Rotorcraft": {
                    "Z10": {},
                    "Z19": {}
                }
            },
            "Latin American Aircraft": {
                "Fixed-Wing": {
                    "EMB314": {},
                    "C390": {}
                }
            }
        }
    }

    # Generate and save flat version (for function hierarchy eval)
    coco_files_flat_func = cvt_to_coco_json(
        split_to_images, {}, subtract_one=args.subtract_one, use_all_nodes=False)
    for split, coco in coco_files_flat_func.items():
        out_file = osp.join(args.out_dir, f'aircraft_{split}.json')
        dump(coco, out_file)
        print(f'Saved {split} COCO file to {out_file}')

    # # Generate and save flat version (for function hierarchy eval)
    # coco_files_flat_func = cvt_to_coco_json(
    #     split_to_images, taxonomy_function, subtract_one=args.subtract_one, use_all_nodes=False)
    # for split, coco in coco_files_flat_func.items():
    #     out_file = osp.join(args.out_dir, f'aircraft_flat_function_{split}.json')
    #     dump(coco, out_file)
    #     print(f'Saved {split} COCO file to {out_file}')

    # # Generate and save flat version (for area hierarchy eval)
    # coco_files_flat_area = cvt_to_coco_json(
    #     split_to_images, taxonomy_area, subtract_one=args.subtract_one, use_all_nodes=False)
    # for split, coco in coco_files_flat_area.items():
    #     out_file = osp.join(args.out_dir, f'aircraft_flat_area_{split}.json')
    #     dump(coco, out_file)
    #     print(f'Saved {split} COCO file to {out_file}')

    # Generate and save hierarchical version (by function)
    coco_files_hier_func = cvt_to_coco_json(
        split_to_images, taxonomy_function, subtract_one=args.subtract_one, use_all_nodes=True)
    for split, coco in coco_files_hier_func.items():
        out_file = osp.join(args.out_dir,
                                f'aircraft_hierarchy_function_{split}.json')
        dump(coco, out_file)
        print(f'Saved {split} COCO file to {out_file}')

    # Generate and save hierarchical version (by area)
    coco_files_hier_area = cvt_to_coco_json(
        split_to_images, taxonomy_area, subtract_one=args.subtract_one, use_all_nodes=True)
    for split, coco in coco_files_hier_area.items():
        out_file = osp.join(args.out_dir,
                                f'aircraft_hierarchy_area_{split}.json')
        dump(coco, out_file)
        print(f'Saved {split} COCO file to {out_file}')


if __name__ == '__main__':
    main()
