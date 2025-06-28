import os.path as osp
import csv
import argparse
from collections import defaultdict

from mmengine.fileio import dump
from mmengine.utils import mkdir_or_exist


TAXONOMY_FUNCTION = {
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

TAXONOMY_AREA = {
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


def cvt_to_coco_json(split_to_images, taxonomy, subtract_one=True, use_all_nodes=False, label_ids_to_use=None):
    """
    Convert grouped annotations into COCO format.
    If subtract_one is True, subtract 1 from bounding box coordinates.
    Returns a dictionary mapping each split (e.g., 'train') to its COCO JSON dict.
    """
    flat = flatten_taxonomy(taxonomy)
    supers = {n: p for n, p in flat}

    # Create a canonical category mapping.
    if label_ids_to_use:
        label_ids = label_ids_to_use
        categories_list = sorted(label_ids.keys(), key=lambda k: label_ids[k])
    else:
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
            'info': {},
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


def expand_excluded_nodes(taxonomy, nodes_to_exclude):
    """
    Expand a list of nodes to include all leaf nodes if an internal node is specified.
    """
    # Helper to get all leaves under a specific starting dictionary
    def _collect_leaves(sub_taxonomy):
        leaves = set()
        for node, children in sub_taxonomy.items():
            if not children:
                leaves.add(node)
            else:
                leaves.update(_collect_leaves(children))
        return leaves

    # Helper to find a node in the taxonomy and return its sub-tree
    def _find_node_subtree(full_taxonomy, target_node):
        if target_node in full_taxonomy:
            return full_taxonomy[target_node]
        for _, children in full_taxonomy.items():
            if isinstance(children, dict):
                found = _find_node_subtree(children, target_node)
                if found is not None:
                    return found
        return None

    expanded_set = set()
    all_leaf_nodes = _collect_leaves(taxonomy)

    for node_name in nodes_to_exclude:
        if node_name in all_leaf_nodes:
            expanded_set.add(node_name)
        else:
            # It's potentially an internal node. Find its subtree.
            subtree = _find_node_subtree(taxonomy, node_name)
            if subtree is not None:
                leaves_of_node = _collect_leaves({node_name: subtree})
                expanded_set.update(leaves_of_node)
            else:
                print(f"Warning: Node '{node_name}' not found in taxonomy. It will be treated as a leaf node.")
                expanded_set.add(node_name)

    return list(expanded_set)


def generate_and_save_coco(split_to_images, taxonomy, use_all_nodes,
                           file_template, out_dir, subtract_one,
                           categories_to_use=None, label_ids_to_use=None):
    """Generates and saves COCO JSON files for the given splits."""
    coco_files = cvt_to_coco_json(
        split_to_images,
        taxonomy,
        subtract_one=subtract_one,
        use_all_nodes=use_all_nodes,
        label_ids_to_use=label_ids_to_use)

    # Overwrite categories for zero-shot evaluation.
    if categories_to_use:
        for split in coco_files:
            coco_files[split]['categories'] = categories_to_use

    for split, coco in coco_files.items():
        out_file = osp.join(out_dir, file_template.format(split=split))
        dump(coco, out_file)
        print(f'Saved {split} COCO file to {out_file}')

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
    parser.add_argument(
        '--exclude-nodes-from-train',
        nargs='*',
        default=[],
        help='List of leaf or internal node names to exclude from the training set. '
             'Images containing these nodes will be removed from the train split.')
    args = parser.parse_args()

    # Parse CSV and group annotations by image and split.
    rows = parse_csv(args.csv_file)
    split_to_images = group_annotations_by_image(rows)
    mkdir_or_exist(args.out_dir)

    # Get canonical node order from the functional taxonomy for deterministic filenames.
    func_taxonomy_nodes_flat = flatten_taxonomy(TAXONOMY_FUNCTION)
    node_order = {node[0]: i for i, node in enumerate(func_taxonomy_nodes_flat)}

    # --- Generate and save standard dataset versions ---
    coco_files_flat = generate_and_save_coco(
        split_to_images, {}, False, 'aircraft_{split}.json',
        args.out_dir, args.subtract_one)

    coco_files_func = generate_and_save_coco(
        split_to_images, TAXONOMY_FUNCTION, True, 'aircraft_hierarchy_function_{split}.json',
        args.out_dir, args.subtract_one)

    coco_files_area = generate_and_save_coco(
        split_to_images, TAXONOMY_AREA, True, 'aircraft_hierarchy_area_{split}.json',
        args.out_dir, args.subtract_one)

    # --- Generate and save versions with excluded training nodes ---
    if args.exclude_nodes_from_train:
        print(f"Expanding nodes to exclude from training set: {args.exclude_nodes_from_train}")
        nodes_to_exclude = expand_excluded_nodes(TAXONOMY_FUNCTION, args.exclude_nodes_from_train)
        print(f"Final list of excluded leaf nodes: {nodes_to_exclude}")

        nodes_to_exclude_set = set(nodes_to_exclude)
        train_images = split_to_images.get('train', {})
        if not train_images:
            print("No 'train' split found to filter.")
        else:
            # Filter training images
            filtered_train_images = {}
            for fname, info in train_images.items():
                if not any(ann['class'] in nodes_to_exclude_set for ann in info['annotations']):
                    filtered_train_images[fname] = info

            original_count = len(train_images)
            filtered_count = len(filtered_train_images)
            print(f"Original training images: {original_count}. "
                  f"Filtered training images: {filtered_count}. "
                  f"Removed {original_count - filtered_count} images.")

            filtered_split_to_images = {'train': filtered_train_images}
            # Sort the excluded nodes based on their order in the taxonomy for a deterministic filename.
            sorted_excluded_nodes = sorted(
                args.exclude_nodes_from_train,
                key=lambda n: node_order.get(n, float('inf')))
            suffix = '_excluded_' + '.'.join(sorted_excluded_nodes).replace(' ', '-')

            # Get full category lists from the original datasets
            flat_cats = coco_files_flat.get('train', {}).get('categories')
            func_cats = coco_files_func.get('train', {}).get('categories')
            area_cats = coco_files_area.get('train', {}).get('categories')

            # Generate and save the filtered datasets
            generate_and_save_coco(
                filtered_split_to_images, {}, False, f'aircraft_{{split}}{suffix}.json',
                args.out_dir, args.subtract_one, categories_to_use=flat_cats,
                label_ids_to_use={cat['name']: cat['id'] for cat in flat_cats})

            generate_and_save_coco(
                filtered_split_to_images, TAXONOMY_FUNCTION, True, f'aircraft_hierarchy_function_{{split}}{suffix}.json',
                args.out_dir, args.subtract_one, categories_to_use=func_cats,
                label_ids_to_use={cat['name']: cat['id'] for cat in func_cats})

            generate_and_save_coco(
                filtered_split_to_images, TAXONOMY_AREA, True, f'aircraft_hierarchy_area_{{split}}{suffix}.json',
                args.out_dir, args.subtract_one, categories_to_use=area_cats,
                label_ids_to_use={cat['name']: cat['id'] for cat in area_cats})


if __name__ == '__main__':
    main()
