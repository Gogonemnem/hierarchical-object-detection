import argparse
import os
import re

import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


def positive_int(value):
    """Type function for argparse to ensure positive integers."""
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"{value} is not a valid positive integer")
    return ivalue


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
        '--no-hierarchy-labels',
        dest='show_hierarchy_labels',
        action='store_false',
        default=True,
        help='disable hierarchical family labels above the bars (default: enabled)')
    parser.add_argument(
        '--show-sample-overlay',
        action='store_true',
        help='overlay sample size information on the plot')
    parser.add_argument(
        '--use-log-scale',
        action='store_true',
        help='use log scale for sample overlay (only used with --show-sample-overlay)')
    parser.add_argument(
        '--max-hierarchy-levels',
        type=int,
        default=None,
        help='maximum number of hierarchy levels to show (default: show all levels)')
    parser.add_argument(
        '--aggregate-at-level',
        type=positive_int,
        default=None,
        help='aggregate classes at specific hierarchy level (0=most general, higher=more specific). '
             'When set, classes are grouped and averaged at this level for cleaner visualization. '
             'If the value exceeds available hierarchy levels, shows all individual leaf classes. '
             'Default: None (show all individual leaf classes)')
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
    # Efficiently extract GT data using list comprehensions
    gt_bboxes = np.array([gt['bbox'] for gt in gts])
    gt_labels = np.array([gt['bbox_label'] for gt in gts])
    true_positives = np.zeros(len(gts))

    # Get unique detection labels for efficient processing
    unique_labels = np.unique(result['labels'].numpy())

    for det_label in unique_labels:
        mask = (result['labels'] == det_label)
        det_bboxes = result['bboxes'][mask].numpy()
        det_scores = result['scores'][mask].numpy()

        # Apply NMS if threshold specified
        if nms_iou_thr:
            det_bboxes, _ = nms(det_bboxes, det_scores, nms_iou_thr, score_threshold=score_thr)
        
        # Calculate IoU overlaps once per detection label
        if len(det_bboxes) > 0 and len(gt_bboxes) > 0:
            ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
            
            # Process detections with vectorized operations where possible
            for i, score in enumerate(det_scores):
                if score >= score_thr:
                    # Find matches above IoU threshold
                    matches = ious[i] >= tp_iou_thr
                    if matches.any():
                        matched_gt_indices = np.where(matches)[0]
                        for j in matched_gt_indices:
                            if gt_labels[j] == det_label:
                                true_positives[j] += 1  # TP
                            confusion_matrix[gt_labels[j], det_label] += 1
                    else:
                        confusion_matrix[-1, det_label] += 1  # BG FP

    # Mark false negatives efficiently
    fn_mask = true_positives == 0
    for gt_label in gt_labels[fn_mask]:
        confusion_matrix[gt_label, -1] += 1


def calculate_hierarchical_prediction_distribution(dataset, confusion_matrix):
    """Calculate the hierarchical prediction distribution including siblings and cousins.

    Args:
        dataset (Dataset): Test or val dataset.
        confusion_matrix (ndarray): The confusion matrix,
            has shape (num_classes + 1, num_classes + 1).
    """
    assert 'taxonomy' in dataset.metainfo, "No taxonomy available for prediction distribution calculation. Exiting."
    
    # Initialize core data structures
    tree = HierarchyTree(dataset.metainfo['taxonomy'])
    class_to_idx = {name: i for i, name in enumerate(dataset.metainfo['classes'])}
    stats = defaultdict(dict)
    
    # Initialize totals with all required fields
    total_stats = stats['Total'] = {key: 0 for key in ['total_gt', 'tp', 'parent_tp', 'grandparent_tp', 
                                                       'sibling_tp', 'cousin_tp', 'ancestor_tp', 'distance', 'fn']}
    
    # Helper function for getting valid indices from class names
    def get_valid_indices(class_names):
        return [class_to_idx[name] for name in class_names if name in class_to_idx]
    
    # Process each leaf node efficiently
    for leaf in sorted(set(tree.get_leaf_nodes())):
        if leaf.name not in class_to_idx:
            continue
            
        # Pre-compute frequently used values
        idx = class_to_idx[leaf.name]
        gts = confusion_matrix[idx]
        leaf_stats = stats[leaf.name]
        
        # Basic stats (combined operations)
        basic_values = {
            'total_gt': gts.sum(),
            'fn': gts[-1], 
            'tp': gts[idx]
        }
        
        # Hierarchical relationship predictions
        parent_tp = gts[class_to_idx[leaf.parent.name]] if (leaf.parent and leaf.parent.name in class_to_idx) else 0
        grandparent = tree.get_grandparent(leaf.name)
        grandparent_tp = gts[class_to_idx[grandparent]] if (grandparent and grandparent in class_to_idx) else 0
        
        # Batch process sibling/cousin/ancestor relationships
        sibling_tp = gts[get_valid_indices(tree.get_siblings(leaf.name))].sum() if tree.get_siblings(leaf.name) else 0
        cousin_tp = gts[get_valid_indices(tree.get_cousins(leaf.name))].sum() if tree.get_cousins(leaf.name) else 0
        
        # Optimized ancestor processing (exclude parent/grandparent to avoid double counting)
        ancestors = set(tree.get_ancestors(leaf.name))
        if leaf.parent and leaf.parent.name in ancestors:
            ancestors.discard(leaf.parent.name)
        if grandparent:
            ancestors.discard(grandparent)
        ancestor_tp = gts[get_valid_indices(ancestors)].sum() if ancestors else 0
        
        # Calculate weighted distance efficiently
        all_ancestors = tree.get_ancestors(leaf.name)
        all_ancestor_indices = get_valid_indices(all_ancestors)
        distance = (gts[all_ancestor_indices] @ np.arange(len(all_ancestors), 
                   len(all_ancestors) - len(all_ancestor_indices), -1)) if all_ancestor_indices else 0
        
        # Combine all values and update both leaf and total stats
        hierarchical_values = {'parent_tp': parent_tp, 'grandparent_tp': grandparent_tp, 
                              'sibling_tp': sibling_tp, 'cousin_tp': cousin_tp, 
                              'ancestor_tp': ancestor_tp, 'distance': distance}
        
        all_values = {**basic_values, **hierarchical_values}
        leaf_stats.update(all_values)
        leaf_stats.update(get_additional_stats(leaf_stats))
        
        # Update totals efficiently
        for key, value in all_values.items():
            total_stats[key] += value
    
    total_stats.update(get_additional_stats(total_stats))
    print_hierarchical_prediction_distribution(stats)
    return stats


def get_additional_stats(stats):
    """Calculate additional derived statistics efficiently."""
    total_gt = stats['total_gt']
    
    # Calculate percentages efficiently with batch processing
    percentage_keys = ['parent_tp', 'grandparent_tp', 'sibling_tp', 'cousin_tp', 'ancestor_tp']
    percentages = {f"{key.replace('_tp', '')}_percentage": 
                   (stats[key] / total_gt * 100) if total_gt > 0 else 0 
                   for key in percentage_keys}
    
    # Calculate remaining derived stats
    total_tp = sum(stats[key] for key in ['tp'] + percentage_keys)
    
    return {
        **percentages,
        'avg_distance': stats['distance'] / total_tp if total_tp > 0 else 0,
        'other_class': total_gt - total_tp - stats['fn']
    }

def print_hierarchical_prediction_distribution(stats):
    """Print the hierarchical prediction distribution statistics.

    Args:
        stats (dict): The hierarchical prediction distribution statistics.
    """
    print("\nHierarchical Prediction Distribution Analysis (Leaf Nodes):")
    header = (f"{'GT Leaf Class':<15} | {'Total GT':>8} | {'TP':>6} | {'Parent':>8} | {'G.Parent':>9} | {'Sibling':>8} | "
              f"{'Cousin':>8} | {'Ancestor':>8} | {'%Parent':>8} | {'%G.Parent':>10} | {'%Sibling':>9} | {'%Cousin':>8} | "
              f"{'%Ancestor':>9} | {'FN':>6}")
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
        node_name (str): The name of the node/class.
        stats (dict): The statistics for a single class.
    """
    print(f"{node_name:<15} | {stats['total_gt']:>8} | {stats['tp']:>6} | {stats['parent_tp']:>8} | "
          f"{stats['grandparent_tp']:>9} | {stats['sibling_tp']:>8} | {stats['cousin_tp']:>8} | {stats['ancestor_tp']:>8} | "
          f"{stats['parent_percentage']:>7.1f}% | {stats['grandparent_percentage']:>9.1f}% | {stats['sibling_percentage']:>8.1f}% | "
          f"{stats['cousin_percentage']:>7.1f}% | {stats['ancestor_percentage']:>8.1f}% | {stats['fn']:>6}")


def _add_hierarchy_annotations(ax, hierarchy_info, leaf_names):
    """Add multi-level hierarchy labels and separators to the plot efficiently."""
    if not hierarchy_info or not hierarchy_info.get('levels'):
        return 105  # Return default top margin
    
    levels = hierarchy_info['levels']
    base_y = 101
    
    # Pre-define color schemes and styles
    level_colors = ['lightgray', 'lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    family_colors = ['#f0f0f0', '#e8f4f8', '#f0f8e8', '#fff8e8', '#f8e8e8']
    line_styles = ['-', '--', '-.', ':']
    separator_colors = ['black', 'darkred', 'darkblue', 'darkgreen', 'purple']
    
    # Add background shading for family groups (most specific level only)
    if levels:
        most_specific = levels[-1]
        for i, (boundary, size) in enumerate(zip(most_specific['boundaries'], most_specific['sizes'])):
            if size > 1:  # Only shade groups with multiple classes
                ax.axvspan(boundary - 0.5, boundary + size - 0.5, 
                          alpha=0.25, color=family_colors[i % len(family_colors)], zorder=0)
    
    # Efficiently collect separator positions and label positions in single pass
    separator_positions = {}  # position -> (level_idx, style_info)
    label_positions = {}      # label -> list of (level_idx, boundary, size)
    
    for level_idx, level_data in enumerate(levels):
        boundaries, labels, sizes = level_data['boundaries'], level_data['labels'], level_data['sizes']
        
        # Process separators
        for i, (boundary, size) in enumerate(zip(boundaries[:-1], sizes[:-1])):
            separator_x = boundary + size - 0.5
            if separator_x < len(leaf_names) - 1:  # Don't add separator after last group
                if separator_x not in separator_positions or level_idx < separator_positions[separator_x][0]:
                    separator_positions[separator_x] = (level_idx, {
                        'alpha': 0.8 - (level_idx * 0.1),
                        'linewidth': 2.5 - (level_idx * 0.3),
                        'linestyle': line_styles[level_idx % len(line_styles)],
                        'color': separator_colors[level_idx % len(separator_colors)]
                    })
        
        # Process labels for duplicate detection
        for boundary, label, size in zip(boundaries, labels, sizes):
            if size >= 2:  # Only consider labels for groups with 2+ items
                if label not in label_positions:
                    label_positions[label] = []
                label_positions[label].append((level_idx, boundary, size))
    
    # Draw separators
    for separator_x, (_, style) in separator_positions.items():
        ax.axvline(x=separator_x, zorder=10, **style)
    
    # Determine which labels to show (highest level only for duplicates)
    labels_to_show = {}
    for label, positions in label_positions.items():
        best_level_idx, best_boundary, best_size = min(positions, key=lambda x: x[0])
        labels_to_show[(best_level_idx, best_boundary, best_size)] = label
    
    # Draw labels efficiently
    for level_idx, level_data in enumerate(levels):
        inverted_level_idx = len(levels) - 1 - level_idx
        y_position = base_y + (inverted_level_idx * 4.0)
        color = level_colors[level_idx % len(level_colors)]
        font_size = max(7, 9 - level_idx)
        
        for boundary, label, size in zip(level_data['boundaries'], level_data['labels'], level_data['sizes']):
            if size >= 2 and labels_to_show.get((level_idx, boundary, size)) == label:
                center_x = boundary + (size - 1) / 2
                ax.text(center_x, y_position, label, ha='center', va='bottom',
                       fontweight='bold' if level_idx == 0 else 'normal',
                       fontsize=font_size, rotation=25,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.9,
                                edgecolor='gray', linewidth=0.5))
    
    return base_y + (len(levels) * 4.0) + 2.0


def _abbreviate_label(label, max_length=8):
    """Abbreviate long labels to prevent overlap."""
    if len(label) <= max_length:
        return label
    
    # Common abbreviation strategies
    abbreviations = {
        'Fighter': 'Fight',
        'Bomber': 'Bomb',
        'Transport': 'Trans',
        'Helicopter': 'Heli',
        'Aircraft': 'AC',
        'Military': 'Mil',
        'Commercial': 'Comm',
        'Private': 'Priv'
    }
    
    # Try known abbreviations first
    for full, abbrev in abbreviations.items():
        if full in label:
            return label.replace(full, abbrev)[:max_length]
    
    # Fallback: truncate with ellipsis
    return label[:max_length-1] + '…' if len(label) > max_length else label


def _format_x_axis_labels(ax, leaf_names, hierarchy_info, stats, show_sample_overlay=False):
    """Format x-axis labels for better readability with low-sample annotations."""
    # Note: hierarchy_info and show_sample_overlay are kept for interface compatibility
    _ = hierarchy_info, show_sample_overlay  # Suppress unused parameter warnings
    # Create abbreviated labels for leaf classes with sample size annotations
    abbreviated_labels = []
    for name in leaf_names:
        if name == 'Total':
            abbreviated_labels.append('**TOTAL**')
        else:
            # Simple abbreviation strategy - take first few chars and numbers
            # You can customize this based on your class naming convention
            if len(name) > 12:
                # Keep important parts like numbers and first letters
                # Extract alphanumeric parts
                parts = re.findall(r'[A-Z0-9]+', name.upper())
                if parts:
                    abbreviated = ''.join(parts[:2])  # Take first 2 parts
                    if len(abbreviated) > 8:
                        abbreviated = abbreviated[:8]
                    label = abbreviated
                else:
                    label = name[:8]
            else:
                label = name
            
            # Add asterisk for low-sample classes (< 10 samples)
            if name in stats and stats[name]['total_gt'] < 10:
                label += '*'
            
            abbreviated_labels.append(label)
    
    # Set the labels with better rotation (90 degrees for better readability)
    ax.set_xticks(range(len(leaf_names)))
    ax.set_xticklabels(abbreviated_labels, rotation=90, ha='center', fontsize=7)
    
    # Add footnote for low-sample annotation positioned to avoid legend overlap
    if any('*' in label for label in abbreviated_labels if label != '**TOTAL**'):
        # Position footnote below the main plot area to avoid legend overlap
        footnote_x = 0.02  # Left side of plot
        footnote_y = -0.15  # Below x-axis labels
        # Make footnote more visible and prominent
        ax.text(footnote_x, footnote_y, '* Classes with < 10 samples', transform=ax.transAxes, 
                fontsize=11, style='italic', alpha=1.0, weight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.98, 
                         edgecolor='red', linewidth=1.5))


def _organize_classes_by_hierarchy(stats, dataset, max_hierarchy_levels=None):
    """
    Organize classes by their hierarchical families using multi-level taxonomy grouping.
    
    Args:
        stats: Statistics dictionary containing class data
        dataset: Dataset object with taxonomy information
        max_hierarchy_levels: Maximum number of hierarchy levels to show (None = show all)
        
    Returns:
        tuple: (ordered_leaf_names, hierarchy_info) where hierarchy_info contains
               multi-level hierarchy boundaries and labels for visualization
    """
    # If no dataset provided, fall back to alphabetical ordering
    if dataset is None or 'taxonomy' not in dataset.metainfo:
        leaf_names = [name for name in stats.keys() if name != 'Total']
        return sorted(leaf_names), {}
    
    # Build hierarchy tree and extract leaf names
    tree = HierarchyTree(dataset.metainfo['taxonomy'])
    leaf_names_only = [name for name in stats.keys() if name != 'Total']
    
    # Pre-compute ancestors for all leaves to avoid repeated calculations
    leaf_ancestors = {leaf: tree.get_ancestors(leaf) if leaf in tree.class_to_node else [] 
                     for leaf in leaf_names_only}
    
    def get_ancestor_at_level(leaf_name, level):
        """Get ancestor at specific level (0=root, 1=mid-level, etc.)"""
        ancestors = leaf_ancestors.get(leaf_name, [])
        return ancestors[level] if level < len(ancestors) else (ancestors[-1] if ancestors else "Other")
    
    def get_groups_at_level(level):
        """Get groups at specific level in hierarchical order."""
        # Create groups using defaultdict for cleaner code
        groups = defaultdict(list)
        for leaf_name in leaf_names_only:
            groups[get_ancestor_at_level(leaf_name, level)].append(leaf_name)
        
        # Extract "Other" group if present
        other_group = groups.pop("Other", None)
        
        # Collect ordered groups by tree traversal
        ordered_groups = []
        def collect_at_level(node, current_level):
            if current_level == level and node.name in groups:
                ordered_groups.append(node.name)
            elif current_level < level:
                for child in node.children:
                    collect_at_level(child, current_level + 1)
        
        collect_at_level(tree.root, 0)
        
        # Add remaining groups and "Other" at end
        ordered_groups.extend(group for group in groups if group not in ordered_groups)
        if other_group:
            groups["Other"] = other_group
            ordered_groups.append("Other")
        
        return ordered_groups, dict(groups)

    # Determine maximum hierarchy depth
    max_depth = max((len(ancestors) for ancestors in leaf_ancestors.values()), default=0)
    
    # Determine finest level for leaf ordering
    finest_level = min(max_depth - 1, (max_hierarchy_levels or max_depth) - 1) if max_depth > 0 else 0
    
    # Get groups at finest level and handle empty groups fallback
    ordered_groups, groups = get_groups_at_level(finest_level)
    if not groups and finest_level > 0:
        finest_level -= 1
        ordered_groups, groups = get_groups_at_level(finest_level)
    
    # Sort leaves within each group by accuracy and create final order
    ordered_leaf_names = []
    for group_name in ordered_groups:
        if group_name not in groups:
            continue
        
        # Sort group leaves by accuracy (descending)
        group_leaves = groups[group_name]
        group_with_accuracy = [(leaf, stats[leaf]['tp'] / max(stats[leaf]['total_gt'], 1)) 
                              for leaf in group_leaves]
        group_with_accuracy.sort(key=lambda x: x[1], reverse=True)
        ordered_leaf_names.extend(leaf for leaf, _ in group_with_accuracy)
    
    # Create hierarchy levels with optimized grouping detection
    max_levels_to_show = min(max_depth, max_hierarchy_levels or max_depth)
    hierarchy_levels = []
    previous_grouping = None
    
    for level in range(max_levels_to_show):
        # Build level info by scanning ordered leaves
        level_info = {'boundaries': [], 'labels': [], 'sizes': []}
        current_grouping = []
        
        i = 0
        while i < len(ordered_leaf_names) and ordered_leaf_names[i] != 'Total':
            current_ancestor = get_ancestor_at_level(ordered_leaf_names[i], level)
            group_start = i
            
            # Find consecutive leaves with same ancestor
            while (i < len(ordered_leaf_names) and ordered_leaf_names[i] != 'Total' and
                   get_ancestor_at_level(ordered_leaf_names[i], level) == current_ancestor):
                i += 1
            
            group_size = i - group_start
            if group_size > 0:
                current_grouping.append((group_start, current_ancestor, group_size))
                level_info['boundaries'].append(group_start)
                level_info['labels'].append(current_ancestor)
                level_info['sizes'].append(group_size)
        
        # Skip duplicate levels
        if (previous_grouping and len(current_grouping) == len(previous_grouping) and
            all(curr[:2] == prev[:2] and curr[2] == prev[2] 
                for curr, prev in zip(current_grouping, previous_grouping))):
            continue
        
        previous_grouping = current_grouping
        hierarchy_levels.append(level_info)
    
    # Add Total at end if exists
    if 'Total' in stats:
        ordered_leaf_names.append('Total')
        for level_info in hierarchy_levels:
            level_info['boundaries'].append(len(ordered_leaf_names) - 1)
            level_info['labels'].append('Total')
            level_info['sizes'].append(1)
    
    return ordered_leaf_names, {'levels': hierarchy_levels}


def plot_stacked_percentage_bar_chart(stats, save_dir, show=True, title_suffix='', dataset=None, 
                                     show_hierarchy_labels=True, show_sample_overlay=False, use_log_scale=False, max_hierarchy_levels=None, aggregate_at_level=None):
    """
    Plots a stacked percentage bar chart for hierarchical prediction distribution with improved organization.
    
    Args:
        stats: Dictionary containing hierarchical prediction distribution statistics
        save_dir: Directory to save the plot
        show: Whether to display the plot
        title_suffix: Additional text for plot title
        dataset: Dataset object with taxonomy information
        show_hierarchy_labels: Whether to show hierarchy labels above bars
        show_sample_overlay: Whether to show sample size overlay
        use_log_scale: Whether to use log scale for sample overlay
        max_hierarchy_levels: Maximum number of hierarchy levels to show (None = show all)
        aggregate_at_level: Aggregate classes at specific hierarchy level (None = show individual classes)
    """
    if not stats:
        print("No data available for stacked bar chart.")
        return
    
    # Group and sort classes by hierarchical families
    leaf_names, hierarchy_info = _organize_classes_by_hierarchy(stats, dataset, max_hierarchy_levels)
    
    # Apply aggregation if requested
    if aggregate_at_level is not None and dataset is not None and 'taxonomy' in dataset.metainfo:
        taxonomy = dataset.metainfo['taxonomy']
        tree = HierarchyTree(taxonomy)
        
        # Find maximum hierarchy depth efficiently
        max_depth = max((len(tree.get_ancestors(name)) for name in stats.keys() 
                        if name != 'Total' and name in tree.class_to_node), default=0)
        
        # Only aggregate if the requested level is within available hierarchy depth
        if aggregate_at_level < max_depth:
            leaf_names, stats, hierarchy_info = _aggregate_stats_at_level(
                leaf_names, stats, dataset, aggregate_at_level, hierarchy_info)
            title_suffix += f' (Aggregated at Level {aggregate_at_level})'
        else:
            print(f"Warning: Requested aggregation level {aggregate_at_level} exceeds maximum hierarchy depth {max_depth-1}. Showing all individual leaf classes.")
    
    # Extract data arrays efficiently using list comprehension with zip
    data_keys = ['total_gt', 'tp', 'parent_tp', 'grandparent_tp', 'sibling_tp', 'cousin_tp', 'ancestor_tp', 'fn', 'other_class']
    data_arrays = {}
    for key in data_keys:
        data_arrays[key] = np.array([stats[name][key] for name in leaf_names])
    
    # Calculate percentages efficiently using vectorized operations
    total_gts = data_arrays['total_gt']
    valid_gt_mask = total_gts > 0
    
    # Create percentage arrays in one go
    percentage_keys = ['tp', 'parent_tp', 'grandparent_tp', 'sibling_tp', 'cousin_tp', 'ancestor_tp', 'fn', 'other_class']
    percentage_arrays = {}
    for key in percentage_keys:
        perc_array = np.zeros_like(total_gts, dtype=float)
        perc_array[valid_gt_mask] = (data_arrays[key][valid_gt_mask] / total_gts[valid_gt_mask]) * 100
        percentage_arrays[key] = perc_array
    
    # Calculate adaptive figure dimensions
    num_bars = len(leaf_names)
    is_aggregated = aggregate_at_level is not None
    
    # Determine bar widths and spacing
    base_bar_width = 0.75
    bar_config = {
        'aggregated': {'min_width': 6, 'max_width': 12, 'width_per_bar': 0.15, 'legend_space': 3.5},
        'individual': {'min_width': 12, 'max_width': 18, 'width_per_bar': 0.75, 'legend_space': 2.0}
    }
    config = bar_config['aggregated'] if is_aggregated else bar_config['individual']
    
    # Adjust legend space based on features
    legend_space = config['legend_space']
    if show_hierarchy_labels and hierarchy_info.get('levels'):
        legend_space += 1.5 if is_aggregated else 0.8
    if show_sample_overlay:
        legend_space += 1.0 if is_aggregated else 0.6
    
    # Calculate final dimensions
    calculated_width = num_bars * config['width_per_bar']
    base_width = max(config['min_width'], min(config['max_width'], calculated_width))
    fig_width = base_width + legend_space
    fig_height = 7
    
    # Cap width in aggregated view
    if is_aggregated:
        fig_width = min(fig_width, fig_height * 1)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

    # Define color scheme and stacking configuration
    colors = {
        'tp': '#2E8B57', 'parent_tp': '#66CDAA', 'grandparent_tp': '#90EE90',
        'sibling_tp': '#87CEEB', 'cousin_tp': '#DDA0DD', 'ancestor_tp': '#CD853F',
        'fn': '#FF8C00', 'other_class': '#DC143C'
    }
    
    labels = {
        'tp': 'Correct as Leaf', 'parent_tp': 'Predicted as Parent', 
        'grandparent_tp': 'Predicted as Grandparent', 'sibling_tp': 'Predicted as Sibling',
        'cousin_tp': 'Predicted as Cousin', 'ancestor_tp': 'Predicted as Other Ancestor',
        'fn': 'Missed (FN)', 'other_class': 'Other Misclassification'
    }
    
    # Calculate cumulative bottoms for stacking
    stack_order = ['tp', 'parent_tp', 'grandparent_tp', 'sibling_tp', 'cousin_tp', 'ancestor_tp', 'fn', 'other_class']
    bottoms = {stack_order[0]: np.zeros_like(total_gts, dtype=float)}
    for i in range(1, len(stack_order)):
        bottoms[stack_order[i]] = bottoms[stack_order[i-1]] + percentage_arrays[stack_order[i-1]]

    # Plot bars efficiently
    x_positions = range(len(leaf_names))
    bar_width = base_bar_width
    total_bar_width = base_bar_width * 1.05
    
    for i, (x, name) in enumerate(zip(x_positions, leaf_names)):
        width = total_bar_width if name == 'Total' else bar_width
        
        # Plot all segments for this bar
        for key in stack_order:
            label = labels[key] if i == 0 else ""
            ax.bar(x, percentage_arrays[key][i], width, bottom=bottoms[key][i],
                   label=label, color=colors[key], alpha=0.85)
        
        # Add outline for Total bar
        if name == 'Total':
            total_height = sum(percentage_arrays[key][i] for key in stack_order)
            outline_rect = patches.Rectangle((x - width/2, 0), width, total_height, 
                                           fill=False, edgecolor='black', linewidth=2, alpha=0.8)
            ax.add_patch(outline_rect)

    # Configure plot appearance
    ax.set_xlabel('Ground Truth Leaf Class', fontsize=12)
    ax.set_ylabel('Percentage of Predictions (%)', fontsize=12)
    ax.set_title(f'Hierarchical Prediction Distribution for Leaf Classes{title_suffix}', 
                fontsize=14, pad=20)
    
    # Add hierarchy annotations if requested
    if show_hierarchy_labels and hierarchy_info.get('levels') and aggregate_at_level is None:
        max_y_position = _add_hierarchy_annotations(ax, hierarchy_info, leaf_names)
        show_hierarchy_labels = True  # Confirm it was shown
    else:
        max_y_position = 105  # Basic spacing above 100%
        show_hierarchy_labels = False  # Ensure hierarchy labels are not shown when aggregating
    
    # Set y-axis and grid
    ax.set_ylim(0, max_y_position)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create sample size overlay if requested
    if show_sample_overlay:
        ax2 = ax.twinx()
        
        # Extract sample counts (excluding Total for better scaling)
        sample_data = [(i, total_gts[i]) for i, name in enumerate(leaf_names) if name != 'Total']
        if sample_data:
            x_positions_overlay, sample_counts = zip(*sample_data)
            
            # Apply log scale if requested
            if use_log_scale:
                sample_counts_plot = np.log10(np.maximum(sample_counts, 1))
                ax2.set_ylabel('Log₁₀(Sample Count)', fontsize=9, color='gray')
                format_string = 'Log₁₀'
            else:
                sample_counts_plot = sample_counts
                ax2.set_ylabel('Sample Count', fontsize=9, color='gray')
                format_string = 'Count'
            
            # Plot sample overlay
            _ = ax2.plot(x_positions_overlay, sample_counts_plot, 
                        color='steelblue', marker='o', markersize=3, 
                        linewidth=1.2, alpha=0.7, linestyle='-',
                        label=f'Sample {format_string}')
            
            # Style secondary axis
            ax2.tick_params(axis='y', labelcolor='steelblue', labelsize=8)
            ax2.spines['right'].set_color('steelblue')
            ax2.spines['right'].set_alpha(0.7)
    
    # Format x-axis labels
    _format_x_axis_labels(ax, leaf_names, hierarchy_info, stats, show_sample_overlay)
    
    # Handle legend efficiently
    handles, labels = ax.get_legend_handles_labels()
    main_handles = handles[:8]
    main_labels = labels[:8]
    reversed_handles = list(reversed(main_handles))
    reversed_labels = list(reversed(main_labels))
    
    # Add sample overlay to legend if present
    if show_sample_overlay and len(handles) > 8:
        reversed_handles.append(handles[-1])
        reversed_labels.append(labels[-1])

    # Place main legend with adaptive sizing
    legend_fontsize = 10 if is_aggregated else 9
    _ = fig.legend(reversed_handles, reversed_labels,
                  loc='center left', bbox_to_anchor=(1.02, 0.6),
                  frameon=True, fancybox=True, shadow=True, fontsize=legend_fontsize,
                  framealpha=0.9, title='Prediction Categories')

    # Add hierarchy level legend if needed
    if show_hierarchy_labels and hierarchy_info.get('levels'):
        from matplotlib.lines import Line2D
        line_styles = ['-', '--', '-.', ':']
        line_colors = ['black', 'darkred', 'darkblue', 'darkgreen', 'purple']
        level_count = len(hierarchy_info['levels'])
        
        line_elements = [Line2D([0], [0], color=line_colors[i % len(line_colors)], 
                               linestyle=line_styles[i % len(line_styles)], 
                               linewidth=2.5 - (i * 0.3), alpha=0.8 - (i * 0.1))
                        for i in range(level_count)]
        line_labels = [f'Level {i}' for i in range(level_count)]
        
        _ = fig.legend(line_elements, line_labels, loc='center left', bbox_to_anchor=(1.02, 0.3),
                      ncol=1, fontsize=8, frameon=True, fancybox=True, shadow=True,
                      framealpha=0.9, title='Hierarchy Separators', title_fontsize=9)

    # Final layout adjustments
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, right=0.98 if not is_aggregated else 0.85)

    # Save and show
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'hierarchical_prediction_distribution_stacked_bar{title_suffix.replace(" ", "_")}.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Stacked bar chart saved to {save_path}")
    if show:
        plt.show()

    plt.close(fig)

def _aggregate_stats_at_level(leaf_names, stats, dataset, aggregate_level, hierarchy_info):
    """
    Aggregate statistics at a specific hierarchy level for cleaner visualization.
    
    Args:
        leaf_names: List of leaf class names
        stats: Original statistics dictionary
        dataset: Dataset object with taxonomy
        aggregate_level: Level at which to aggregate (0=most general)
        hierarchy_info: Original hierarchy information
        
    Returns:
        tuple: (aggregated_leaf_names, aggregated_stats, updated_hierarchy_info)
    """
    tree = HierarchyTree(dataset.metainfo['taxonomy'])
    
    # Helper function to get ancestor at specific level
    def get_ancestor_at_level(leaf_name, level):
        if leaf_name == 'Total':
            return 'Total'
        if leaf_name not in tree.class_to_node:
            return f"Other_L{level}"
        
        ancestors = tree.get_ancestors(leaf_name)
        return ancestors[level] if level < len(ancestors) else (ancestors[-1] if ancestors else f"Other_L{level}")
    
    # Group classes by their ancestor efficiently using dictionary comprehension
    aggregated_groups = defaultdict(list)
    for leaf_name in leaf_names:
        ancestor = get_ancestor_at_level(leaf_name, aggregate_level)
        aggregated_groups[ancestor].append(leaf_name)
    
    # Define aggregation keys for statistics
    stat_keys = ['total_gt', 'tp', 'parent_tp', 'grandparent_tp', 'sibling_tp', 'cousin_tp', 'ancestor_tp', 'distance', 'fn']
    
    # Create aggregated statistics efficiently
    aggregated_stats = {}
    aggregated_leaf_names = []
    
    for group_name, group_members in aggregated_groups.items():
        if group_name == 'Total':
            # Keep Total as-is
            aggregated_stats['Total'] = stats['Total']
        else:
            # Aggregate statistics using vectorized sum operations
            group_stats = {key: sum(stats[member][key] for member in group_members if member in stats) 
                          for key in stat_keys}
            
            # Add derived statistics
            group_stats.update(get_additional_stats(group_stats))
            aggregated_stats[group_name] = group_stats
        
        aggregated_leaf_names.append(group_name)
    
    # Simplified hierarchy info update - only keep levels above aggregation level
    updated_hierarchy_info = {'levels': []}
    if hierarchy_info.get('levels') and aggregate_level > 0:
        # Create new hierarchy levels for aggregated structure
        for _ in range(min(aggregate_level, len(hierarchy_info['levels']))):
            # Build new level info based on aggregated groups
            non_total_groups = [name for name in aggregated_leaf_names if name != 'Total']
            if non_total_groups:
                new_level_info = {
                    'boundaries': list(range(len(non_total_groups))),
                    'labels': non_total_groups,
                    'sizes': [1] * len(non_total_groups)  # Each aggregated group is size 1
                }
                
                # Add Total if present
                if 'Total' in aggregated_leaf_names:
                    total_pos = len(non_total_groups)
                    new_level_info['boundaries'].append(total_pos)
                    new_level_info['labels'].append('Total')
                    new_level_info['sizes'].append(1)
                
                updated_hierarchy_info['levels'].append(new_level_info)
                break  # Only need one level for aggregated view
    
    return aggregated_leaf_names, aggregated_stats, updated_hierarchy_info


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
    
    stats = calculate_hierarchical_prediction_distribution(dataset, confusion_matrix)

    plot_stacked_percentage_bar_chart(
        stats, 
        args.save_dir, 
        args.show, 
        dataset=dataset,
        show_hierarchy_labels=args.show_hierarchy_labels,
        show_sample_overlay=args.show_sample_overlay,
        use_log_scale=args.use_log_scale,
        max_hierarchy_levels=args.max_hierarchy_levels,
        aggregate_at_level=args.aggregate_at_level
    )



if __name__ == '__main__':
    main()
