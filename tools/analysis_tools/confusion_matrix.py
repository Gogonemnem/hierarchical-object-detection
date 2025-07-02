import argparse
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from mmcv.ops import nms
from mmengine import Config, DictAction
from mmengine.fileio import load
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

from mmdet.evaluation import bbox_overlaps
from mmdet.registry import DATASETS
from mmdet.utils import replace_cfg_vals, update_data_root
from hod.utils.tree import HierarchyTree


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


def flatten_taxonomy_with_paths(taxonomy_tree):
    """
    Flatten a taxonomy dictionary using HierarchyTree.
    Returns a list of tuples: (leaf_name_str, full_path_list_of_names_from_root_to_parent)
    Example: [("F-16", ["Military Aircraft", "Fixed-Wing", "Fighters"])]
    """
    flat_list = []
    
    # Assuming tree.get_leaf_nodes() returns a list of HierarchyNode objects
    # Each node object should have a .name attribute (string)
    leaf_node_objects = taxonomy_tree.get_leaf_nodes()
    
    if not leaf_node_objects:
        return []
    
    # Sort node objects by their name attribute for consistent order
    # Assuming node_obj.name exists and is a string
    try:
        sorted_leaf_node_objects = sorted(leaf_node_objects, key=lambda node: node.name)
    except AttributeError:
        # Fallback if .name attribute is missing, try direct sort (might fail or be unsorted if complex objects)
        # Or if get_leaf_nodes() actually returns strings (error message suggested otherwise)
        sorted_leaf_node_objects = sorted(leaf_node_objects) 

    for node_obj in sorted_leaf_node_objects:
        leaf_name_str = ""
        if hasattr(node_obj, 'name'):
            leaf_name_str = str(node_obj.name) # Extract string name
        else:
            leaf_name_str = str(node_obj) # Assume node_obj is already a name or string-convertible
        
        # tree.get_ancestors(str_name) returns a list of ancestor name strings,
        # from immediate parent up to the root.
        ancestor_names_parent_to_root = taxonomy_tree.get_ancestors(leaf_name_str) 
        
        path_names_root_to_parent = []
        if ancestor_names_parent_to_root:
            path_names_root_to_parent = [str(name) for name in reversed(ancestor_names_parent_to_root)]
        
        flat_list.append((leaf_name_str, path_names_root_to_parent))
    return flat_list


def build_nested_boundaries(flat_tax):
    """
    Build group boundaries for every level in the path.
    Returns a list of boundary specs:
        [
          (level_idx, group_name, start_idx, end_idx),
          ...
        ]
    from outermost (level 0) to innermost (leaf parent).
    """

    # Sort flat_tax by the entire path so that siblings are adjacent
    flat_tax_sorted = sorted(flat_tax, key=lambda x: x[1])

    boundaries_per_level = []
    max_depth = 0
    if not flat_tax_sorted:
        return [], []
        
    for _, path in flat_tax_sorted:
        max_depth = max(max_depth, len(path))

    for level in range(max_depth):
        current_group_name = None
        start_idx = 0
        for idx, (leaf, path) in enumerate(flat_tax_sorted):
            group_at_this_level = path[level] if len(path) > level else "<NoGroup>"

            if current_group_name is None:
                current_group_name = group_at_this_level
                start_idx = idx
            elif current_group_name != group_at_this_level:
                boundaries_per_level.append((level, current_group_name, start_idx, idx - 1))
                current_group_name = group_at_this_level
                start_idx = idx

        if current_group_name is not None:
            boundaries_per_level.append((level, current_group_name, start_idx, len(flat_tax_sorted) - 1))
            
    return flat_tax_sorted, boundaries_per_level


def _get_node_group_at_agg_level(node_label: str, tree: HierarchyTree, agg_level: int) -> str:
    """
    Determines the group name for a given node label based on the aggregation level.

    Args:
        node_label: The name of the node (can be leaf or internal).
        tree: The HierarchyTree object representing the taxonomy.
        agg_level: The target depth for aggregation (0-indexed).
                   Nodes at this depth will form the groups.

    Returns:
        The name of the group class.
    """
    if node_label not in tree.class_to_node:
        # This should not happen if node_label comes from tree-aware processing
        return node_label # Fallback: node is its own group

    # path_ancestors is the list of ancestor names: [root, child_of_root, ..., parent_of_node]
    path_ancestors = tree.get_ancestors(node_label)
    
    # node_depth is the depth of the node itself. Root is at depth 0.
    # A node whose parent is the root has depth 1. path_ancestors = [root], len=1.
    # A node whose grandparent is the root has depth 2. path_ancestors = [root, parent], len=2.
    node_depth = len(path_ancestors) 

    if not path_ancestors and node_label == tree.root.name: # node_label is the root
        return node_label

    if agg_level < node_depth:
        # The node is strictly deeper than the aggregation level.
        # Group it by its ancestor at the specified aggregation depth.
        # e.g., agg_level=0 means group by root (path_ancestors[0])
        group_name = path_ancestors[agg_level]
        return group_name
    else: # agg_level >= node_depth
        # The node is at or shallower than the aggregation level.
        # It means this node should be its own distinct group at this level of detail.
        return node_label


def aggregate_confusion_matrix(
    original_cm: np.ndarray,
    tree: HierarchyTree,
    agg_level: int,
    node_labels_in_cm_order: list
) -> tuple:
    """
    Aggregates a confusion matrix based on a hierarchical taxonomy.

    Args:
        original_cm: The input confusion matrix. Assumed to have classes corresponding
                     to node_labels_in_cm_order, with the last row/col being 'background'.
                     These can be leaf or non-leaf nodes.
        tree: The HierarchyTree object.
        agg_level: The target depth for aggregation (0-indexed).
        node_labels_in_cm_order: Ordered list of node class names (leaf or non-leaf)
                                 as they appear in the original_cm (excluding 'background').

    Returns:
        A tuple containing:
        - aggregated_cm (np.ndarray): The new aggregated confusion matrix.
        - aggregated_labels (List[str]): The labels for the aggregated matrix (including 'background').
    """
    num_original_nodes = len(node_labels_in_cm_order)
    if original_cm.shape != (num_original_nodes + 1, num_original_nodes + 1):
        raise ValueError(
            f"Original CM shape {original_cm.shape} does not match "
            f"number of node labels {num_original_nodes} + background."
        )

    node_to_group_map = {
        node_label: _get_node_group_at_agg_level(node_label, tree, agg_level)
        for node_label in tree.all_classes()
    }

    # Define the labels for this level: nodes at this depth AND leaf nodes that are deeper.
    # Leaf nodes must be preserved at all aggregation levels once they appear.
    groups_at_this_level = []
    seen_groups = set()
    for leaf in node_labels_in_cm_order:
        group = node_to_group_map.get(leaf)
        if group and group not in seen_groups:
            group_node = tree.class_to_node.get(group)
            if group_node:
                # Include groups that are exactly at this level OR leaf nodes at deeper levels
                if (group_node.get_depth() == agg_level or group_node.is_leaf()):
                    groups_at_this_level.append(group)
                    seen_groups.add(group)

    aggregated_labels_no_bg = groups_at_this_level
    aggregated_labels_with_bg = aggregated_labels_no_bg + ['background']
    
    group_to_idx_map = {name: i for i, name in enumerate(aggregated_labels_no_bg)}
    num_aggregated_classes = len(aggregated_labels_no_bg)
    aggregated_cm = np.zeros((num_aggregated_classes + 1, num_aggregated_classes + 1), dtype=original_cm.dtype)

    # --- Main Aggregation Loop ---
    for i, gt_node_name in enumerate(node_labels_in_cm_order):
        gt_group = node_to_group_map.get(gt_node_name)
        if gt_group not in group_to_idx_map:
            continue
        gt_idx = group_to_idx_map[gt_group]

        for j, dt_node_name in enumerate(node_labels_in_cm_order):
            # Key Filtering Logic:
            # If a prediction is for a node at a HIGHER level (shallower depth),
            # it should ALWAYS go to background, regardless of ancestry relationships
            dt_node = tree.class_to_node.get(dt_node_name)
            if dt_node.get_depth() < agg_level and not dt_node.is_leaf():
                # All ancestor predictions which are not leaves go to background
                dt_idx = num_aggregated_classes
            else:
                # Otherwise, aggregate it up to the current level normally.
                dt_group = node_to_group_map.get(dt_node_name)
                if dt_group not in group_to_idx_map:
                    continue  # Skip if no group mapping exists
                dt_idx = group_to_idx_map[dt_group]
            
            aggregated_cm[gt_idx, dt_idx] += original_cm[i, j]

        # Handle original background predictions for the current GT group
        aggregated_cm[gt_idx, num_aggregated_classes] += original_cm[i, num_original_nodes]

    # --- Handle GT Background Row ---
    for j, dt_node_name in enumerate(node_labels_in_cm_order):
        dt_node = tree.class_to_node.get(dt_node_name)

        # For background GT, ancestor predictions should still be mapped to background
        # since there's no specific GT class to match against
        if dt_node.get_depth() < agg_level and not dt_node.is_leaf():
            new_pred_idx = num_aggregated_classes  # Remap to background
        else:
            dt_group = node_to_group_map.get(dt_node_name)
            if dt_group not in group_to_idx_map:
                continue  # Skip if no group mapping exists
            new_pred_idx = group_to_idx_map[dt_group]
        
        aggregated_cm[num_aggregated_classes, new_pred_idx] += original_cm[num_original_nodes, j]
    
    # Sum of original GT background to predicted background
    aggregated_cm[num_aggregated_classes, num_aggregated_classes] += original_cm[num_original_nodes, num_original_nodes]

    return aggregated_cm, aggregated_labels_with_bg


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
    # Hierarchical setup
    tree = None
    path_lookup = None
    class_names = dataset.metainfo['classes']
    if 'taxonomy' in dataset.metainfo:
        tree = HierarchyTree(dataset.metainfo['taxonomy'])
        path_lookup = {}
        for name in tree.class_to_node.keys():
            path_lookup[name] = set(tree.get_path(name))

    assert len(dataset) == len(results)
    prog_bar = ProgressBar(len(results))
    for idx, per_img_res in enumerate(results):
        res_bboxes = per_img_res['pred_instances']
        gts = dataset.get_data_info(idx)['instances']
        analyze_per_img_dets(confusion_matrix, gts, res_bboxes, score_thr,
                             tp_iou_thr, nms_iou_thr, class_names, tree, path_lookup)
        prog_bar.update()
    return confusion_matrix


def analyze_per_img_dets(confusion_matrix,
                         gts,
                         result,
                         score_thr=0,
                         tp_iou_thr=0.5,
                         nms_iou_thr=None,
                         class_names=None,
                         tree=None,
                         path_lookup=None):
    """Analyze detection results on each image using non-greedy hierarchical matching.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        gts (list): Ground truth instances.
        result (dict): Detection results from the model.
        score_thr (float): Score threshold to filter bboxes.
        tp_iou_thr (float): IoU threshold to be considered as matched.
        nms_iou_thr (float|optional): NMS IoU threshold.
        class_names (list): List of class names.
        tree (HierarchyTree): The hierarchy tree.
        path_lookup (dict): Lookup table for hierarchy paths.
    """
    # Use standard matching if no hierarchy is provided
    if tree is None or path_lookup is None:
        # This is the original greedy matching logic from this file
        true_positives = np.zeros(len(gts))
        gt_bboxes = np.array([gt['bbox'] for gt in gts]) if gts else np.empty((0, 4))
        gt_labels = np.array([gt['bbox_label'] for gt in gts]) if gts else np.empty((0,))

        det_labels_all = result['labels'].cpu().numpy() if hasattr(result['labels'], 'cpu') else np.array(result['labels'])
        det_bboxes_all = result['bboxes'].cpu().numpy() if hasattr(result['bboxes'], 'cpu') else np.array(result['bboxes'])
        det_scores_all = result['scores'].cpu().numpy() if hasattr(result['scores'], 'cpu') else np.array(result['scores'])

        unique_det_labels = np.unique(det_labels_all)
        num_classes = confusion_matrix.shape[0] - 1

        for det_label in unique_det_labels:
            if det_label >= num_classes:
                continue

            mask = (det_labels_all == det_label) & (det_scores_all >= score_thr)
            det_bboxes = det_bboxes_all[mask]

            if len(det_bboxes) == 0:
                continue

            if nms_iou_thr:
                # Placeholder for NMS if needed, current logic processes per class
                pass

            if len(gt_bboxes) == 0:
                for _ in range(len(det_bboxes)):
                    confusion_matrix[num_classes, det_label] += 1
                continue

            ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
            for i in range(len(det_bboxes)):
                matched_gt = -1
                best_iou = tp_iou_thr

                for j in range(len(gt_bboxes)):
                    if ious[i, j] >= best_iou:
                        best_iou = ious[i, j]
                        matched_gt = j
                
                if matched_gt != -1:
                    if true_positives[matched_gt] == 0:
                        true_positives[matched_gt] = 1
                        confusion_matrix[gt_labels[matched_gt], det_label] += 1
                    else:
                        confusion_matrix[num_classes, det_label] += 1
                else:
                    confusion_matrix[num_classes, det_label] += 1

        for i in range(len(gt_labels)):
            if true_positives[i] == 0:
                confusion_matrix[gt_labels[i], num_classes] += 1
        return

    # Hierarchical Matching Logic
    assert class_names is not None

    # 1. Data Preparation
    gt_bboxes = np.array([gt['bbox'] for gt in gts]) if gts else np.empty((0, 4))
    gt_labels = np.array([gt['bbox_label'] for gt in gts]) if gts else np.empty((0,))
    det_scores = result['scores'].numpy()
    score_mask = det_scores >= score_thr
    det_bboxes = result['bboxes'].numpy()[score_mask]
    det_labels = result['labels'].numpy()[score_mask]
    det_scores = det_scores[score_mask]

    if nms_iou_thr and len(det_bboxes) > 0:
        det_bboxes, keep = nms(det_bboxes, det_scores, nms_iou_thr)
        det_labels = det_labels[keep]
        det_scores = det_scores[keep]

    # Sort detections by score for processing order
    sort_inds = np.argsort(-det_scores)
    det_bboxes = det_bboxes[sort_inds]
    det_labels = det_labels[sort_inds]
    det_scores = det_scores[sort_inds]

    D = len(det_labels)
    G = len(gt_labels)
    num_classes = confusion_matrix.shape[0] - 1

    if G == 0:
        for d_label in det_labels:
            confusion_matrix[num_classes, d_label] += 1
        return

    if D == 0:
        for g_label in gt_labels:
            confusion_matrix[g_label, num_classes] += 1
        return

    # 2. Pre-calculate IoUs
    ious = bbox_overlaps(det_bboxes, gt_bboxes)

    # 3. Initialization for Non-Greedy Matching
    gtm_idx = -np.ones(G, dtype=int)
    gt_hf1 = np.zeros(G)
    d_matched = np.zeros(D, dtype=bool)
    dtIg = np.zeros(D, dtype=bool)

    detections_to_process = list(range(D))
    label_to_name = {i: name for i, name in enumerate(class_names)}

    # 4. Iterative Matching with Stealing
    while detections_to_process:
        dind = detections_to_process.pop(0)
        d_label_name = label_to_name.get(det_labels[dind])
        if not d_label_name:
            continue

        best_boost = 1e-9
        best_gt_match_idx = -1
        best_iou = -1.0
        best_unmatched_hf1 = -1.0
        best_unmatched_gt_idx = -1

        for gind in range(G):
            if ious[dind, gind] < tp_iou_thr:
                continue

            gt_label_name = label_to_name.get(gt_labels[gind])
            if not gt_label_name:
                continue

            metrics = hierarchical_prf_metric(path_lookup, d_label_name, gt_label_name, return_paths=True)
            potential_hf1 = metrics.get('hf1', 0.0) if isinstance(metrics, dict) else 0.0

            existing_hf1 = gt_hf1[gind]
            boost = potential_hf1 - existing_hf1

            if boost > best_boost or (boost == best_boost and ious[dind, gind] > best_iou):
                best_boost = boost
                best_gt_match_idx = gind
                best_iou = ious[dind, gind]

            if potential_hf1 > best_unmatched_hf1:
                best_unmatched_hf1 = potential_hf1
                best_unmatched_gt_idx = gind

        if best_gt_match_idx != -1:
            prev_d_idx = gtm_idx[best_gt_match_idx]
            if prev_d_idx != -1:
                d_matched[prev_d_idx] = False
                detections_to_process.insert(0, prev_d_idx)

            gtm_idx[best_gt_match_idx] = dind
            gt_hf1[best_gt_match_idx] = best_boost + gt_hf1[best_gt_match_idx]
            d_matched[dind] = True
        else:
            # 5. Ancestor-Ignoring Logic
            if best_unmatched_gt_idx != -1:
                final_match_d_idx = gtm_idx[best_unmatched_gt_idx]
                if final_match_d_idx != -1:
                    final_match_d_label = label_to_name.get(det_labels[final_match_d_idx])
                    if final_match_d_label and tree.is_descendant(final_match_d_label, d_label_name):
                        dtIg[dind] = True

    # 6. Update Confusion Matrix
    gt_matched_by_any_det = np.zeros(G, dtype=bool)
    for dind in range(D):
        if d_matched[dind]:
            matched_gind = np.where(gtm_idx == dind)[0]
            if len(matched_gind) > 0:
                gind = matched_gind[0]
                gt_label = gt_labels[gind]
                det_label = det_labels[dind]
                confusion_matrix[gt_label, det_label] += 1
                gt_matched_by_any_det[gind] = True
        elif not dtIg[dind]:
            confusion_matrix[num_classes, det_labels[dind]] += 1

    for gind in range(G):
        if not gt_matched_by_any_det[gind]:
            confusion_matrix[gt_labels[gind], num_classes] += 1


def hierarchical_prf_metric(
    path_lookup: dict,
    dt_label,
    gt_label,
    return_paths: bool = False
):
    """Compute hierarchical precision, recall, and F1 between predicted and GT labels.

    Optionally returns the raw paths for additional use in aggregation.
    """
    dt_path = path_lookup.get(dt_label, set())
    gt_path = path_lookup.get(gt_label, set())

    len_dt_path = len(dt_path)
    len_gt_path = len(gt_path)

    if len_dt_path == 0 or len_gt_path == 0:
        hf1 = 0.0
        len_overlap = 0
    else:
        overlap = dt_path & gt_path
        len_overlap = len(overlap)
        hprecision = len_overlap / len_dt_path
        hrecall = len_overlap / len_gt_path
        if hprecision + hrecall == 0:
            hf1 = 0.0
        else:
            hf1 = 2 * (hprecision * hrecall) / (hprecision + hrecall)

    if return_paths:
        return {
            'hf1': hf1,
            'len_overlap': len_overlap,
            'len_dt': len_dt_path,
            'len_gt': len_gt_path,
        }
    else:
        return hf1


def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_path=None,
                          show=True,
                          title='Normalized Confusion Matrix',
                          color_theme='plasma',
                          boundaries_info=None,
                          wrap_width=15):
    """
    Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_path (str|optional): If set, save the confusion matrix plot to the given path.
        show (bool): Whether to show the plot.
        title (str): Title of the plot.
        color_theme (str): Theme of the matrix color map.
        boundaries_info (list|None): Information for drawing boundaries.
        wrap_width (int): Maximum width for wrapping labels.
    """
    # Normalize the confusion matrix.
    per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    # Handle division by zero for labels with no instances
    confusion_matrix_normalized = np.zeros_like(confusion_matrix, dtype=np.float32)
    np.divide(confusion_matrix, per_label_sums, out=confusion_matrix_normalized, where=per_label_sums!=0)
    confusion_matrix_normalized *= 100

    num_classes = len(labels)
    # Compute dynamic figure size but set minimum dimensions.
    fig_width = max(8, 0.5 * num_classes)
    fig_height = max(8, 0.5 * num_classes * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix_normalized, cmap=cmap, vmin=0, vmax=100) # Ensure consistent color scale
    plt.colorbar(mappable=im, ax=ax, shrink=0.8) # Added shrink to colorbar

    title_font = {'weight': 'bold', 'size': 12}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 10}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # Draw locators and grid.
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(True, which='minor', linestyle='-')

    # Optionally draw hierarchical boundaries.
    if boundaries_info is not None and boundaries_info: # Check if list is not None and not empty
        boundaries_info_sorted = sorted(boundaries_info, key=lambda x: x[0]) # Sort by level_idx, ascending

        # Determine max_level_val for z-ordering (higher conceptual levels on top)
        max_level_val = max(b[0] for b in boundaries_info_sorted) 

        # New color list for better contrast with 'plasma'
        # Level 0 (highest conceptual) will be white, then lime, cyan, etc.
        boundary_colors = ['white', 'lime', 'cyan', 'magenta', 'orange']
        
        for level, group_name, start, end in boundaries_info_sorted:
            # Prominence: Higher conceptual levels (lower 'level') are thicker and more opaque
            lw = 2.5 - (level * 0.5) 
            alpha = 0.8 - (level * 0.15)
            
            # Ensure lw and alpha do not become too small or negative
            lw = max(0.5, lw)
            alpha = max(0.1, alpha)

            color_index = level % len(boundary_colors)
            c = boundary_colors[color_index]
            
            # Z-order: Higher conceptual levels (lower 'level') get higher zorder to be drawn on top
            current_zorder = (max_level_val - level) + 5 

            # Horizontal lines (for rows)
            ax.plot([-0.5, num_classes - 0.5], [start - 0.5, start - 0.5], color=c, linewidth=lw, alpha=alpha, zorder=current_zorder)
            ax.plot([-0.5, num_classes - 0.5], [end + 0.5, end + 0.5], color=c, linewidth=lw, alpha=alpha, zorder=current_zorder)
            # Vertical lines (for columns)
            ax.plot([start - 0.5, start - 0.5], [-0.5, num_classes - 0.5], color=c, linewidth=lw, alpha=alpha, zorder=current_zorder)
            ax.plot([end + 0.5, end + 0.5], [-0.5, num_classes - 0.5], color=c, linewidth=lw, alpha=alpha, zorder=current_zorder)
            
    # Wrap long labels.
    wrapped_labels = [textwrap.fill(label, width=wrap_width) for label in labels]
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(wrapped_labels)
    ax.set_yticklabels(wrapped_labels)
    
    # Adjust tick parameters.
    ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True, pad=10, labelsize=8) # Reduced label size
    ax.tick_params(axis='y', labelsize=8, pad=10) # Reduced label size
    plt.setp(ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # Draw the numeric values inside each cell.
    for i in range(num_classes):
        for j in range(num_classes):
            val = confusion_matrix_normalized[i, j]
            # Determine text color based on cell background lightness
            # Using a simple threshold on the normalized value (0-100)
            # Values closer to 0 are dark, values closer to 100 are light (depending on cmap)
            # This might need adjustment based on the specific cmap used.
            # For 'plasma', lower values are dark purple/blue, higher are yellow.
            # A more robust way would be to get the RGB of the cell color and calculate luminance.
            if color_theme in ['plasma', 'viridis', 'magma', 'cividis']:
                # These colormaps generally go from dark to light as value increases.
                text_color = 'white' if val < 50 else 'black' # If cell is dark, use white text.
            elif color_theme in ['binary', 'gray', 'gist_gray']:
                 text_color = 'white' if val < 50 else 'black' # Similar logic for grayscale
            else: # Default for other cmaps, or could be more specific
                text_color = 'white' if val > 50 else 'black' # If cell is light, use black text.
            ax.text(
                j, i, f'{val:.1f}', ha='center', va='center', color=text_color, fontsize=7) # Reduced fontsize

    ax.set_ylim(len(confusion_matrix_normalized) - 0.5, -0.5)  # matplotlib>3.1.1
    
    # Adjust margins so labels have room.
    # Default margins, hierarchical labels removed.
    fig.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)

    if save_path is not None:
        plt.savefig(
            save_path, format='png', bbox_inches='tight') # bbox_inches='tight' can help
    if show:
        plt.show()
    plt.close(fig) # Close the figure after saving/showing to free memory


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg = replace_cfg_vals(cfg)

    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    results = load(args.prediction_path)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    _ = dataset.metainfo

    confusion_matrix_raw = calculate_confusion_matrix(dataset, results,
                                                  args.score_thr,
                                                  args.nms_iou_thr,
                                                  args.tp_iou_thr)
    
    original_labels_no_bg = list(dataset.metainfo['classes'])
    num_classes_no_bg = len(original_labels_no_bg)
    original_label_to_idx = {label: i for i, label in enumerate(original_labels_no_bg)}

    # Initialize variables that will be set in the if/else block
    ordered_leaf_names = []
    boundaries_for_plot = None
    confusion_matrix_to_plot = None
    labels_to_plot = []
    # flat_tax_sorted needs to be defined for the aggregate mode later, even if taxonomy is not used for hierarchy plot
    flat_tax_sorted = [] 
    taxonomy_tree = None # Initialize to None

    if 'taxonomy' in dataset.metainfo and dataset.metainfo['taxonomy']:
        taxonomy_tree = HierarchyTree(dataset.metainfo['taxonomy'])
        all_taxonomy_leaves_with_paths = flatten_taxonomy_with_paths(taxonomy_tree)
        
        plottable_leaves_with_paths_unsorted = [
            item for item in all_taxonomy_leaves_with_paths if item[0] in original_label_to_idx
        ]

        if not plottable_leaves_with_paths_unsorted:
            print("Warning: No common leaf classes found between taxonomy and dataset. Plotting non-hierarchically.")
            labels_to_plot = original_labels_no_bg + ['background']
            confusion_matrix_to_plot = confusion_matrix_raw
            boundaries_for_plot = None
            ordered_leaf_names = original_labels_no_bg 
            # flat_tax_sorted remains empty as there's no valid taxonomy structure to use
        else:
            # Build boundaries and the final sorted order using ONLY these plottable leaves
            # build_nested_boundaries returns the sorted list as its first element
            flat_tax_sorted, boundaries_for_plot = build_nested_boundaries(plottable_leaves_with_paths_unsorted)
            # raise Exception(f"flat_tax_sorted: {flat_tax_sorted}, boundaries_for_plot: {boundaries_for_plot}")
            ordered_leaf_names = [item[0] for item in flat_tax_sorted] # Final order for plot

            dataset_classes_excluded = [
                name for name in original_labels_no_bg if name not in ordered_leaf_names
            ]
            if dataset_classes_excluded:
                print(f"Warning: The following dataset classes are not recognized leaf nodes in the provided taxonomy "
                      f"or are not part of the plottable set, and will be EXCLUDED from the hierarchical "
                      f"confusion matrix: {dataset_classes_excluded}")

            # Reorder the entire matrix, including the background row/column, in one step
            confusion_matrix_to_plot = confusion_matrix_raw

            labels_to_plot = ordered_leaf_names + ['background']

    else: # No taxonomy in dataset.metainfo or taxonomy is empty
        labels_to_plot = original_labels_no_bg + ['background']
        confusion_matrix_to_plot = confusion_matrix_raw
        boundaries_for_plot = None
        ordered_leaf_names = original_labels_no_bg 
        taxonomy_tree = None # No tree to use for aggregation
        # flat_tax_sorted remains empty
        print("No taxonomy information found in dataset.metainfo or taxonomy is empty. Plotting with original class order.")

    # Always use the aggregation logic. If no taxonomy, it plots a non-hierarchical matrix.
    if not taxonomy_tree: # Check if taxonomy_tree was initialized
        print("No taxonomy information available. Plotting a single non-hierarchical confusion matrix.")
        plot_confusion_matrix(
            confusion_matrix_to_plot, # Should be confusion_matrix_raw if no taxonomy
            labels_to_plot,           # Should be original_labels_no_bg + ['background']
            save_path=os.path.join(args.save_dir, 'confusion_matrix_flat.png'),
            show=args.show,
            title='Normalized Confusion Matrix (Flat)',
            color_theme=args.color_theme,
            boundaries_info=None # No boundaries for a flat matrix
        )
    else:
        # Aggregation logic
        max_agg_level = taxonomy_tree.root.get_height()

        ordered_labels_leaf_for_agg = ordered_leaf_names # These are the leaves in confusion_matrix_to_plot

        levels_to_aggregate = []
        print(f"Plotting all aggregation levels from 0 (root) to {max_agg_level} (leaves).")
        levels_to_aggregate = range(0, max_agg_level + 1)

        for level in levels_to_aggregate:
            if not ordered_labels_leaf_for_agg:
                # This case should be rare if taxonomy_tree is valid and there are plottable leaves
                print(f"No leaf labels to aggregate at level {level}. Skipping.")
                continue
            
            current_agg_matrix, current_agg_labels_with_bg = aggregate_confusion_matrix(
                confusion_matrix_to_plot, 
                taxonomy_tree, 
                level, 
                dataset.metainfo['classes']
            )

            plot_matrix = current_agg_matrix
            plot_labels_with_bg = current_agg_labels_with_bg
            boundaries_for_agg_plot = None
            current_agg_labels_no_bg = current_agg_labels_with_bg[:-1]

            title_suffix = f"Aggregated Level {level}"
            if level == max_agg_level:
                title_suffix = "Leaf Level (Full Hierarchy)"
            elif level == 0:
                title_suffix = "Root Level (Most Aggregated)"

            if current_agg_labels_no_bg and len(current_agg_labels_no_bg) > 1:
                items_for_boundary_calc = []
                for label_name in current_agg_labels_no_bg:
                    if label_name in taxonomy_tree.class_to_node:
                        path_to_parent = taxonomy_tree.get_ancestors(label_name)
                        items_for_boundary_calc.append((label_name, path_to_parent))
                    else:
                        items_for_boundary_calc.append((label_name, []))
                
                path_sorted_items, boundaries_for_agg_plot = build_nested_boundaries(items_for_boundary_calc)
                
                if boundaries_for_agg_plot:
                    final_plot_labels_no_bg = [item[0] for item in path_sorted_items]
                    if final_plot_labels_no_bg != current_agg_labels_no_bg:
                        N = len(current_agg_labels_no_bg)
                        current_cm_main_part = current_agg_matrix[:N, :N]
                        current_bg_row = current_agg_matrix[N, :N]
                        current_bg_col = current_agg_matrix[:N, N]
                        map_alpha_to_idx = {label: i for i, label in enumerate(current_agg_labels_no_bg)}
                        reorder_indices_for_path_sort = [map_alpha_to_idx[label] for label in final_plot_labels_no_bg]
                        reordered_cm_main_part = current_cm_main_part[np.ix_(reorder_indices_for_path_sort, reorder_indices_for_path_sort)]
                        reordered_bg_row = current_bg_row[reorder_indices_for_path_sort]
                        reordered_bg_col = current_bg_col[reorder_indices_for_path_sort]
                        new_plot_matrix = np.zeros_like(current_agg_matrix)
                        new_plot_matrix[:N, :N] = reordered_cm_main_part
                        new_plot_matrix[N, :N] = reordered_bg_row
                        new_plot_matrix[:N, N] = reordered_bg_col
                        new_plot_matrix[N, N] = current_agg_matrix[N, N]
                        plot_matrix = new_plot_matrix
                        plot_labels_with_bg = final_plot_labels_no_bg + ['background']
                else:
                    boundaries_for_agg_plot = None
            
            plot_confusion_matrix(
                plot_matrix,
                plot_labels_with_bg, 
                save_path=os.path.join(args.save_dir, f'confusion_matrix_level_{level}.png'),
                show=args.show,
                title=f'Normalized Confusion Matrix ({title_suffix})',
                color_theme=args.color_theme,
                boundaries_info=boundaries_for_agg_plot
            )


if __name__ == '__main__':
    main()
