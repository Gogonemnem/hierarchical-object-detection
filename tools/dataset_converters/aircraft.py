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

def cvt_to_coco_json(split_to_images, subtract_one=True):
    """
    Convert grouped annotations into COCO format.
    If subtract_one is True, subtract 1 from bounding box coordinates.
    Returns a dictionary mapping each split (e.g., 'train') to its COCO JSON dict.
    """
    # Define your taxonomy (only once, as metadata)
    taxonomy = {
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
    }

    # Gather unique categories from the CSV
    categories_set = set()
    for images in split_to_images.values():
        for info in images.values():
            for ann in info['annotations']:
                categories_set.add(ann['class'])
    categories_list = sorted(list(categories_set))
    label_ids = {cat: idx for idx, cat in enumerate(categories_list)}

    # Prepare COCO JSONs per split
    coco_files = {}
    for split, images in split_to_images.items():
        coco = {
            'images': [],
            'annotations': [],
            'categories': [],
            'type': 'instance',
            # Add taxonomy metadata here:
            'taxonomy': taxonomy
        }
        # Build the categories list (adjust 'supercategory' if needed)
        for cat, idx in label_ids.items():
            coco['categories'].append({
                'supercategory': 'none',  # or set a high-level parent if you wish
                'id': idx + 1,
                'name': cat
            })
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
                    'category_id': label_ids[ann['class']],
                    'bbox': bbox,
                    'area': area,
                    'segmentation': segmentation,
                    'iscrowd': 0,
                    'ignore': 0
                }
                coco['annotations'].append(coco_ann)
                ann_id += 1
            image_id += 1
        coco_files[split] = coco
    return coco_files

def main():
    parser = argparse.ArgumentParser(
        description='Convert aircraft CSV annotations to COCO JSON format'
    )
    parser.add_argument(
        '--csv_file',
        type=str,
        required=True,
        help='Path to the CSV file (e.g., labels_with_split.csv)'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='Directory to save the output COCO JSON files'
    )
    parser.add_argument(
        '--subtract_one',
        action='store_true',
        help='Subtract 1 from bounding box coordinates (default: False)'
    )
    args = parser.parse_args()

    # Parse CSV and group annotations by image and split.
    rows = parse_csv(args.csv_file)
    split_to_images = group_annotations_by_image(rows)
    coco_files = cvt_to_coco_json(split_to_images, subtract_one=args.subtract_one)

    mkdir_or_exist(args.out_dir)
    for split, coco in coco_files.items():
        out_file = osp.join(args.out_dir, f'aircraft_{split}.json')
        dump(coco, out_file)
        print(f'Saved {split} COCO file to {out_file}')

if __name__ == '__main__':
    main()
