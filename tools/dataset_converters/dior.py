import os.path as osp
import xml.etree.ElementTree as ET

import numpy as np
from mmengine.fileio import dump, list_from_file
from mmengine.utils import mkdir_or_exist, track_progress


from pascal_voc import parse_args, cvt_to_coco_json

from hod.evaluation import dior_classes

label_ids = {name: i for i, name in enumerate(dior_classes())}

def parse_xml(args):
    xml_path, img_path = args
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        
        # Check if the class is in the current class list
        if name not in label_ids:
            print(f"Warning: Class '{name}' not in class list. Skipping.")
            continue
        
        label = label_ids[name]
        
        # Check if 'difficult' exists, otherwise default to 0
        difficult_tag = obj.find('difficult')
        difficult = int(difficult_tag.text) if difficult_tag is not None else 0
        
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        if difficult:
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
        else:
            bboxes.append(bbox)
            labels.append(label)

    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0,))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)

    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0,))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore)

    annotation = {
        'filename': img_path,
        'width': w,
        'height': h,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation


def cvt_annotations(devkit_path, split, out_file):
    """ Convert DIOR annotations to COCO format or pickle format. """

    # Check which images directory to use
    if split == 'test':
        img_dir = 'JPEGImages-test'
    else:
        img_dir = 'JPEGImages-trainval'
    
    # ImageSets txt file
    filelist = osp.join(devkit_path, f'ImageSets/Main/{split}.txt')
    if not osp.isfile(filelist):
        print(f'filelist does not exist: {filelist}, skip dior {split}')
        return

    # Get image names
    img_names = list_from_file(filelist)
    xml_paths = [
        osp.join(devkit_path, f'Annotations/Horizontal Bounding Boxes/{img_name}.xml')
        for img_name in img_names
    ]
    img_paths = [
        osp.join(img_dir, f'{img_name}.jpg') for img_name in img_names
    ]
    part_annotations = track_progress(parse_xml,
                                      list(zip(xml_paths, img_paths)))

    annotations = []
    annotations.extend(part_annotations)
    if out_file.endswith('json'):
        annotations = cvt_to_coco_json(annotations)
    dump(annotations, out_file)
    return annotations


def cvt_to_coco_json(annotations):
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
        annotation_item = dict()
        annotation_item['segmentation'] = []

        seg = []
        # bbox[] is x1,y1,x2,y2
        # left_top
        seg.append(int(bbox[0]))
        seg.append(int(bbox[1]))
        # left_bottom
        seg.append(int(bbox[0]))
        seg.append(int(bbox[3]))
        # right_bottom
        seg.append(int(bbox[2]))
        seg.append(int(bbox[3]))
        # right_top
        seg.append(int(bbox[2]))
        seg.append(int(bbox[1]))

        annotation_item['segmentation'].append(seg)

        xywh = np.array(
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item['area'] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 1
        else:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 0
        annotation_item['image_id'] = int(image_id)
        annotation_item['bbox'] = xywh.astype(int).tolist()
        annotation_item['category_id'] = int(category_id)
        annotation_item['id'] = int(annotation_id)
        coco['annotations'].append(annotation_item)
        return annotation_id + 1

    for category_id, name in enumerate(dior_classes()):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for ann_dict in annotations:
        file_name = ann_dict['filename']
        ann = ann_dict['ann']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(ann_dict['height'])
        image_item['width'] = int(ann_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        bboxes = ann['bboxes'][:, :4]
        labels = ann['labels']
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=0)

        bboxes_ignore = ann['bboxes_ignore'][:, :4]
        labels_ignore = ann['labels_ignore']
        for bbox_id in range(len(bboxes_ignore)):
            bbox = bboxes_ignore[bbox_id]
            label = labels_ignore[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=1)

        image_id += 1

    return coco


def main():
    args = parse_args()
    devkit_path = args.devkit_path
    out_dir = args.out_dir if args.out_dir else devkit_path
    mkdir_or_exist(out_dir)

    # DIOR has its own folder structure
    splits = ['train', 'val', 'trainval', 'test']

    # Confirm the required directories are present
    if not osp.isdir(osp.join(devkit_path, 'Annotations', 'Horizontal Bounding Boxes')):
        raise IOError(f'The devkit path {devkit_path} must contain an "Annotations/Horizontal Bounding Boxes" folder')
    if not osp.isdir(osp.join(devkit_path, 'JPEGImages-trainval')) and not osp.isdir(osp.join(devkit_path, 'JPEGImages-test')):
        raise IOError(f'The devkit path {devkit_path} must contain both "JPEGImages-trainval" and "JPEGImages-test" folders')
    if not osp.isdir(osp.join(devkit_path, 'ImageSets/Main')):
        raise IOError(f'The devkit path {devkit_path} must contain "ImageSets/Main" folder with train/test splits')

    # Set output format
    out_fmt = f'.{args.out_format}'
    if args.out_format == 'coco':
        out_fmt = '.json'

    # Loop through DIOR-specific splits
    for split in splits:
        dataset_name = f'dior_{split}'
        print(f'Processing {dataset_name} ...')
        
        # Convert DIOR annotations
        cvt_annotations(devkit_path, split,
                             osp.join(out_dir, dataset_name + out_fmt))
    
    print('Done!')


if __name__ == '__main__':
    main()
