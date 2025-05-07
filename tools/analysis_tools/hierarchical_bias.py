import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from mmcv.ops import nms
from mmengine import Config, DictAction
from mmengine.fileio import load
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

from mmdet.evaluation import bbox_overlaps
from mmdet.registry import DATASETS
from mmdet.utils import replace_cfg_vals, update_data_root

from hod.utils.tree import HierarchyTree
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from detection results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test .pkl result')
    parser.add_argument(
        'save_dir', help='directory where confusion matrix will be saved')
    parser.add_argument(
        '--show', action='store_true', help='show confusion matrix')
    parser.add_argument(
        '--color-theme',
        default='plasma',
        help='theme of the matrix color map')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='score threshold to filter detection bboxes')
    parser.add_argument(
        '--tp-iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold to be considered as matched')
    parser.add_argument(
        '--nms-iou-thr',
        type=float,
        default=None,
        help='nms IoU threshold, only applied when users want to change the'
        'nms IoU threshold.')
    parser.add_argument(
        '--mode',
        type=str,
        default='hierarchy',
        choices=['leaf', 'hierarchy', 'aggregate'],
        help='Mode to plot the confusion matrix: "leaf" (only leaf nodes), '
             '"hierarchy" (leaf nodes with hierarchical boundaries drawn), or '
             '"aggregate" (aggregate and plot confusion matrices at every hierarchy level).'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def calculate_confusion_matrix(dataset,
                               results,
                               score_thr=0,
                               nms_iou_thr=None,
                               tp_iou_thr=0.5):
    """Calculate the confusion matrix.

    Args:
        dataset (Dataset): Test or val dataset.
        results (list[ndarray]): A list of detection results in each image.
        score_thr (float|optional): Score threshold to filter bboxes.
            Default: 0.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
        tp_iou_thr (float|optional): IoU threshold to be considered as matched.
            Default: 0.5.
    """
    num_classes = len(dataset.metainfo['classes'])
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
    assert len(dataset) == len(results)
    prog_bar = ProgressBar(len(results))
    for idx, per_img_res in enumerate(results):
        res_bboxes = per_img_res['pred_instances']
        gts = dataset.get_data_info(idx)['instances']
        analyze_per_img_dets(confusion_matrix, gts, res_bboxes, score_thr,
                             tp_iou_thr, nms_iou_thr)
        prog_bar.update()
    return confusion_matrix


def analyze_per_img_dets(confusion_matrix,
                         gts,
                         result,
                         score_thr=0,
                         tp_iou_thr=0.5,
                         nms_iou_thr=None):
    """Analyze detection results on each image.

    Args:
        confusion_matrix (ndarray): The confusion matrix,
            has shape (num_classes + 1, num_classes + 1).
        gt_bboxes (ndarray): Ground truth bboxes, has shape (num_gt, 4).
        gt_labels (ndarray): Ground truth labels, has shape (num_gt).
        result (ndarray): Detection results, has shape
            (num_classes, num_bboxes, 5).
        score_thr (float): Score threshold to filter bboxes.
            Default: 0.
        tp_iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
    """
    true_positives = np.zeros(len(gts))
    gt_bboxes = []
    gt_labels = []
    for gt in gts:
        gt_bboxes.append(gt['bbox'])
        gt_labels.append(gt['bbox_label'])

    gt_bboxes = np.array(gt_bboxes)
    gt_labels = np.array(gt_labels)

    unique_label = np.unique(result['labels'].numpy())

    for det_label in unique_label:
        mask = (result['labels'] == det_label)
        det_bboxes = result['bboxes'][mask].numpy()
        det_scores = result['scores'][mask].numpy()

        if nms_iou_thr:
            det_bboxes, _ = nms(
                det_bboxes, det_scores, nms_iou_thr, score_threshold=score_thr)
        ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
        for i, score in enumerate(det_scores):
            det_match = 0
            if score >= score_thr:
                for j, gt_label in enumerate(gt_labels):
                    if ious[i, j] >= tp_iou_thr:
                        det_match += 1
                        if gt_label == det_label:
                            true_positives[j] += 1  # TP
                        confusion_matrix[gt_label, det_label] += 1
                if det_match == 0:  # BG FP
                    confusion_matrix[-1, det_label] += 1
    for num_tp, gt_label in zip(true_positives, gt_labels):
        if num_tp == 0:  # FN
            confusion_matrix[gt_label, -1] += 1


def calculate_hierarchy_bias(dataset, confusion_matrix,):
    """Calculate the hierarchy bias.

    Args:
        dataset (Dataset): Test or val dataset.
        confusion_matrix (ndarray): The confusion matrix,
            has shape (num_classes + 1, num_classes + 1).
    """
    assert 'taxonomy' in dataset.metainfo, \
        "No taxonomy available for bias calculation. Exiting."
    # Build the taxonomy tree.
    taxonomy = dataset.metainfo['taxonomy']
    tree = HierarchyTree(taxonomy)
    leaf_nodes = tree.get_leaf_nodes()

    classes_list = dataset.metainfo['classes']
    class_to_idx = {name: i for i, name in enumerate(classes_list)}

    # Initialize metrics as a defaultdict that creates defaultdict(int) for new keys
    stats = defaultdict(dict)

    # Specifically initialize the 'Total' key as a regular dictionary
    # with its predefined structure and zero values.
    stats['Total'] = {
        'total_gt': 0,
        'tp': 0,
        'parent_tp': 0,
        'ancestor_tp': 0,
        'distance': 0,
        'fn': 0,
    }

    # Sort leaf_node_names for consistent output, if desired
    sorted_leaf_nodes = sorted(list(set(leaf_nodes)))

    for leaf in sorted_leaf_nodes:
        if leaf.name not in class_to_idx:
            # This leaf from taxonomy is not in the dataset's main class list, skip
            # Or handle as an error/warning if all taxonomy leaves should be in classes_list
            # print(f"Warning: Leaf node '{leaf_name}' from taxonomy not in dataset classes_list.")
            continue
        
        idx = class_to_idx[leaf.name]
        gts = confusion_matrix[idx]

        total_gt = gts.sum()
        stats[leaf.name]['total_gt'] = total_gt
        stats['Total']['total_gt'] += total_gt

        fn = gts[-1]
        stats[leaf.name]['fn'] = fn
        stats['Total']['fn'] += fn

        tp = gts[idx]
        stats[leaf.name]['tp'] = tp
        stats['Total']['tp'] += tp

        parent = leaf.parent
        parent_idx = class_to_idx[parent.name]
        parent_tp = gts[parent_idx]
        stats[leaf.name]['parent_tp'] = parent_tp
        stats['Total']['parent_tp'] += parent_tp

        ancestors = parent.ancestors()
        ancestor_idx = [class_to_idx[ancestor] for ancestor in ancestors]
        ancestors_tp = gts[ancestor_idx].sum()
        stats[leaf.name]['ancestor_tp'] = ancestors_tp
        stats['Total']['ancestor_tp'] += ancestors_tp

        # Calculate weights: len(ancestor_idx)+1, len(ancestor_idx), ..., 2
        ancestor_weights = np.arange(len(ancestor_idx), 0, -1) + 1
        
        # Calculate the weighted sum of ancestor TPs
        # gts[ancestor_idx] is 1D, so .transpose is redundant but kept from original
        # If ancestor_idx is empty, gts[ancestor_idx] and ancestor_weights will be empty arrays,
        # and their dot product will correctly be 0.0.
        weighted_ancestor_tp = gts[ancestor_idx].transpose() @ ancestor_weights
        total_distance = parent_tp + weighted_ancestor_tp
        stats[leaf.name]['distance'] = total_distance
        stats['Total']['distance'] += total_distance
        
        stats[leaf.name].update(get_additional_stats(stats[leaf.name]))

    stats['Total'].update(get_additional_stats(stats['Total']))
    print_hierarchy_bias(stats)

    return stats


def get_additional_stats(stats):
    parent_percentage = stats['parent_tp'] / stats['total_gt'] * 100 if stats['total_gt'] > 0 else 0
    ancestor_percentage = stats['ancestor_tp'] / stats['total_gt'] * 100 if stats['total_gt'] > 0 else 0
    total_tp = stats['tp'] + stats['parent_tp'] + stats['ancestor_tp']
    avg_distance = stats['distance'] / total_tp if total_tp > 0 else 0
    other_class = stats['total_gt'] - total_tp - stats['fn']
    
    return {
        'parent_percentage': parent_percentage,
        'ancestor_percentage': ancestor_percentage,
        'avg_distance': avg_distance,
        'other_class': other_class,
    }

def print_hierarchy_bias(stats):
    """Print the hierarchy bias statistics.

    Args:
        stats (dict): The hierarchy bias statistics.
    """
    print("\nAncestor Prediction Bias Analysis (Leaf Nodes):")
    header = (f"{'GT Leaf Class':<15} | {'Total GT':>8} | {'Total TP':>8} | {'Parent Pred':>11} | "
              f"{'Ancestor Pred':>13} | {'% Parent':>8} | {'% Ancestor':>10} | {'Total Distance':>14} |"
              f"{'Avg Distance':>13}")
    print(header)
    print("-" * len(header))

    for node_name, stats_per_class in stats.items():
        if node_name == 'Total':
            continue
        print_single_class_stats(node_name, stats_per_class)

    print("-" * len(header))

    print_single_class_stats('Total', stats['Total'])


def print_single_class_stats(node_name, stats):
    """Print the statistics for a single class.

    Args:
        stats_per_class (dict): The statistics for a single class.
    """
    print(f"{node_name:<15} | {stats['total_gt']:>8} | {stats['tp']:>8} | {stats['parent_tp']:>11} | "
                f"{stats['ancestor_tp']:>13} | {stats['parent_percentage']:>7.1f}% | {stats['ancestor_percentage']:>9.1f}% | "
                f"{stats['distance']:>14.1f} | {stats['avg_distance']:>12.1f}")


def plot_stacked_percentage_bar_chart(stats, save_dir, show=True, title_suffix=''):
    """
    Plots a stacked percentage bar chart for ancestor bias.
    """
    if not stats:
        print("No data available for stacked bar chart.")
        return
    
    leaf_names = stats.keys()
    total_gts = np.array([s['total_gt'] for s in stats.values()])
    leaf_counts = np.array([s['tp'] for s in stats.values()])
    parent_counts = np.array([s['parent_tp'] for s in stats.values()])
    ancestor_counts = np.array([s['ancestor_tp'] for s in stats.values()])
    fn_counts = np.array([s['fn'] for s in stats.values()])
    other_class_counts = np.array([s['other_class'] for s in stats.values()])

    # Calculate percentages, handle division by zero for total_gts
    correct_perc = np.zeros_like(total_gts, dtype=float)
    parent_perc = np.zeros_like(total_gts, dtype=float)
    other_ancestor_perc = np.zeros_like(total_gts, dtype=float)
    fn_perc = np.zeros_like(total_gts, dtype=float)
    other_misclass_perc = np.zeros_like(total_gts, dtype=float)

    valid_gt_mask = total_gts > 0
    correct_perc[valid_gt_mask] = (leaf_counts[valid_gt_mask] / total_gts[valid_gt_mask]) * 100
    parent_perc[valid_gt_mask] = (parent_counts[valid_gt_mask] / total_gts[valid_gt_mask]) * 100
    other_ancestor_perc[valid_gt_mask] = (ancestor_counts[valid_gt_mask] / total_gts[valid_gt_mask]) * 100
    fn_perc[valid_gt_mask] = (fn_counts[valid_gt_mask] / total_gts[valid_gt_mask]) * 100
    other_misclass_perc[valid_gt_mask] = (other_class_counts[valid_gt_mask] / total_gts[valid_gt_mask]) * 100
    
    num_leaves = len(leaf_names)
    fig_width = max(10, num_leaves * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 8), dpi=150)

    bar_width = 0.65

    ax.bar(leaf_names, correct_perc, bar_width, label='Correct as Leaf')
    ax.bar(leaf_names, parent_perc, bar_width, bottom=correct_perc, label='Predicted as Parent')
    ax.bar(leaf_names, other_ancestor_perc, bar_width, bottom=correct_perc + parent_perc, label='Predicted as Other Ancestor')
    ax.bar(leaf_names, other_misclass_perc, bar_width, bottom=correct_perc + parent_perc + other_ancestor_perc, label='Other Misclassification')
    ax.bar(leaf_names, fn_perc, bar_width, bottom=correct_perc + parent_perc + other_ancestor_perc + other_misclass_perc, label='Missed (FN)')

    ax.set_xlabel('Ground Truth Leaf Class')
    ax.set_ylabel('Percentage of Predictions (%)')
    ax.set_title(f'Prediction Distribution for Leaf Classes{title_suffix}')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3) # Adjust legend position
    plt.xticks(rotation=45, ha='right')
    ax.set_ylim(0, 105) # Give a little space above 100%
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.1, 1, 1]) # Adjust layout to make space for legend

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'hierarchical_bias_stacked_bar{title_suffix.replace(" ", "_")}.png')
        plt.savefig(save_path)
        print(f"Stacked bar chart saved to {save_path}")
    
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    results = load(args.prediction_path)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)

    confusion_matrix = calculate_confusion_matrix(dataset, results,
                                                  args.score_thr,
                                                  args.nms_iou_thr,
                                                  args.tp_iou_thr)
    
    stats = calculate_hierarchy_bias(dataset, confusion_matrix)


    plot_stacked_percentage_bar_chart(stats, args.save_dir, args.show)



if __name__ == '__main__':
    main()
