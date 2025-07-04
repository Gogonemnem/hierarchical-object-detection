from typing import Optional
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
from tqdm import tqdm

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
    parser.add_argument(
        '--paper-size',
        action='store_true',
        help='Whether to use paper-sized plots. Defaults to False.'
    )
    args = parser.parse_args()
    return args


class HierarchyCache:
    """Pre-computed hierarchy information cache for efficient lookups."""
    
    def __init__(self, taxonomy_tree):
        self.tree = taxonomy_tree
        self._build_cache()
    
    def _build_cache(self):
        """Build all necessary caches in one pass through the tree."""
        # Core mappings
        self.node_to_depth = {}
        self.node_to_ancestors = {}
        self.depth_to_nodes = {}
        self.node_to_level_ancestor = {}
        
        # Process all nodes in one traversal
        def _traverse(node, depth=0, ancestors=None):
            if ancestors is None:
                ancestors = []
            
            name = node.name
            self.node_to_depth[name] = depth
            self.node_to_ancestors[name] = ancestors.copy()
            
            # Group nodes by depth
            if depth not in self.depth_to_nodes:
                self.depth_to_nodes[depth] = []
            self.depth_to_nodes[depth].append(name)
            
            # Pre-compute ancestor at each level
            if name not in self.node_to_level_ancestor:
                self.node_to_level_ancestor[name] = {}
            
            for level in range(depth + 1):
                if level < len(ancestors):
                    self.node_to_level_ancestor[name][level] = ancestors[level]
                else:
                    self.node_to_level_ancestor[name][level] = name
            
            # Recurse to children
            new_ancestors = ancestors + [name]
            for child in node.children:
                _traverse(child, depth + 1, new_ancestors)
        
        _traverse(self.tree.root)
        
        # Additional optimizations
        self.max_depth = max(self.depth_to_nodes.keys()) if self.depth_to_nodes else 0
        self.leaf_nodes = [name for name, node in self.tree.class_to_node.items() if node.is_leaf()]


def get_leaf_hierarchy_paths(taxonomy_tree):
    """
    Get sorted leaf nodes with their hierarchy paths.
    Returns list of tuples: (leaf_name, path_from_root)
    
    Optimized with hierarchy cache for efficient construction.
    """
    cache = HierarchyCache(taxonomy_tree)
    
    # Use cached data for fast construction
    leaf_paths = []
    for leaf_name in sorted(cache.leaf_nodes):
        ancestors = cache.node_to_ancestors[leaf_name]
        path_from_root = list(reversed(ancestors)) if ancestors else []
        leaf_paths.append((leaf_name, path_from_root))
    
    return leaf_paths


def compute_hierarchy_boundaries(sorted_items_with_paths):
    """
    Compute hierarchical group boundaries for visualization.
    Returns (sorted_items, boundaries) where boundaries contain (level, group_name, start, end).
    
    Optimized with pre-sorted input and single-pass boundary detection.
    """
    if not sorted_items_with_paths:
        return [], []
    
    # Sort by hierarchy path for proper grouping - single operation
    sorted_items = sorted(sorted_items_with_paths, key=lambda x: x[1])
    
    # Pre-calculate maximum depth to avoid repeated computation
    max_depth = max(len(path) for _, path in sorted_items)
    boundaries = []
    
    # Process all levels in one pass for efficiency
    for level in range(max_depth):
        current_group = None
        start_idx = 0
        
        for idx, (_, path) in enumerate(sorted_items):
            group_at_level = path[level] if level < len(path) else None
            
            if current_group != group_at_level:
                # Finalize previous group
                if current_group is not None:
                    boundaries.append((level, current_group, start_idx, idx - 1))
                # Initialize new group
                current_group = group_at_level
                start_idx = idx
        
        # Finalize last group at this level
        if current_group is not None:
            boundaries.append((level, current_group, start_idx, len(sorted_items) - 1))
    
    return sorted_items, boundaries


def get_aggregation_group_cached(node_label: str, cache: HierarchyCache, agg_level: int) -> str:
    """
    Fast aggregation group lookup using pre-computed cache.
    
    Args:
        node_label: Node name to aggregate
        cache: Pre-computed hierarchy cache
        agg_level: Target aggregation depth (0-indexed)
    
    Returns:
        Group name for aggregation
    """
    # Fast path for unknown nodes
    if node_label not in cache.node_to_level_ancestor:
        return node_label
    
    # Direct lookup from cache - O(1) operation
    return cache.node_to_level_ancestor[node_label].get(agg_level, node_label)


def aggregate_confusion_matrix(
    original_cm: np.ndarray,
    tree: HierarchyTree,
    agg_level: int,
    node_labels_in_cm_order: list,
    cache: Optional[HierarchyCache] = None
) -> tuple:
    """
    Aggregate confusion matrix based on hierarchical taxonomy at specified level.
    
    Uses vectorized operations with pre-computed hierarchy cache for optimal performance.
    
    Args:
        original_cm: Input confusion matrix with background as last row/col
        tree: Hierarchy tree
        agg_level: Target aggregation depth (0-indexed)
        node_labels_in_cm_order: Node class names (excluding 'background')
        cache: Pre-computed hierarchy cache (optional, will create if None)
    
    Returns:
        (aggregated_cm, aggregated_labels): Aggregated matrix and labels
    """
    num_original_nodes = len(node_labels_in_cm_order)
    if original_cm.shape != (num_original_nodes + 1, num_original_nodes + 1):
        raise ValueError(f"Matrix shape {original_cm.shape} doesn't match {num_original_nodes} + background")

    # Use provided cache or create new one
    if cache is None:
        cache = HierarchyCache(tree)
    
    # Compute aggregation groups for GT and predictions
    gt_groups = [get_aggregation_group_cached(node_label, cache, agg_level) 
                 for node_label in node_labels_in_cm_order]
    
    dt_groups = []
    for node_label in node_labels_in_cm_order:
        node_depth = cache.node_to_depth.get(node_label, float('inf'))
        
        # Filter ancestor predictions to background
        if node_depth < agg_level and not tree.class_to_node.get(node_label, tree.root).is_leaf():
            dt_groups.append('background')
        else:
            dt_groups.append(get_aggregation_group_cached(node_label, cache, agg_level))
    
    # Collect unique aggregation groups
    unique_groups = []
    seen = set()
    
    for group in gt_groups:
        if group not in seen and group in tree.class_to_node:
            group_node = tree.class_to_node[group]
            if group_node.get_depth() == agg_level or (group_node.is_leaf() and group_node.get_depth() < agg_level):
                unique_groups.append(group)
                seen.add(group)
    
    # Build aggregated matrix structure
    aggregated_labels = unique_groups + ['background']
    group_to_idx = {name: i for i, name in enumerate(unique_groups)}
    num_agg_classes = len(unique_groups)
    
    # Create index mappings for vectorized operations
    gt_indices = np.array([group_to_idx.get(group, -1) for group in gt_groups])
    dt_indices = np.array([
        group_to_idx.get(group, num_agg_classes if group == 'background' else -1) 
        for group in dt_groups
    ])
    
    # Initialize aggregated matrix
    aggregated_cm = np.zeros((num_agg_classes + 1, num_agg_classes + 1), dtype=original_cm.dtype)
    
    # Vectorized aggregation of main confusion matrix block
    orig_i, orig_j = np.meshgrid(np.arange(num_original_nodes), np.arange(num_original_nodes), indexing='ij')
    agg_i = gt_indices[orig_i.ravel()]
    agg_j = dt_indices[orig_j.ravel()]
    values = original_cm[:num_original_nodes, :num_original_nodes].ravel()
    
    # Filter and accumulate valid entries
    valid_mask = (agg_i >= 0) & (agg_j >= 0)
    if np.any(valid_mask):
        np.add.at(aggregated_cm, (agg_i[valid_mask], agg_j[valid_mask]), values[valid_mask])
    
    # Handle GT to background transitions (vectorized)
    valid_gt_mask = gt_indices >= 0
    if np.any(valid_gt_mask):
        np.add.at(aggregated_cm, 
                  (gt_indices[valid_gt_mask], np.full(np.sum(valid_gt_mask), num_agg_classes)),
                  original_cm[:num_original_nodes, num_original_nodes][valid_gt_mask])
    
    # Handle background to predictions transitions (vectorized)
    valid_dt_mask = dt_indices >= 0
    if np.any(valid_dt_mask):
        np.add.at(aggregated_cm,
                  (np.full(np.sum(valid_dt_mask), num_agg_classes), dt_indices[valid_dt_mask]),
                  original_cm[num_original_nodes, :num_original_nodes][valid_dt_mask])
    
    # Background to background
    aggregated_cm[num_agg_classes, num_agg_classes] = original_cm[num_original_nodes, num_original_nodes]
    
    return aggregated_cm, aggregated_labels


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
    # Hierarchical setup - taxonomy is required
    class_names = dataset.metainfo['classes']
    if 'taxonomy' not in dataset.metainfo or not dataset.metainfo['taxonomy']:
        raise ValueError("Taxonomy information is required in dataset.metainfo['taxonomy']")
    
    tree = HierarchyTree(dataset.metainfo['taxonomy'])
    path_lookup = {}
    for name in tree.class_to_node.keys():
        path_lookup[name] = set(tree.get_path(name))

    assert len(dataset) == len(results)
    
    # Use tqdm for better progress visualization
    for idx, per_img_res in tqdm(enumerate(results), total=len(results), 
                                desc="Analyzing detections", unit="img"):
        res_bboxes = per_img_res['pred_instances']
        gts = dataset.get_data_info(idx)['instances']
        analyze_per_img_dets(confusion_matrix, gts, res_bboxes, score_thr,
                             tp_iou_thr, nms_iou_thr, class_names, tree, path_lookup)
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
    """Analyze detection results on each image using optimized hierarchical matching.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        gts (list): Ground truth instances.
        result (dict): Detection results from the model.
        score_thr (float): Score threshold to filter bboxes.
        tp_iou_thr (float): IoU threshold to be considered as matched.
        nms_iou_thr (float|optional): NMS IoU threshold.
        class_names (list): List of class names.
        tree (HierarchyTree): The hierarchy tree (required).
        path_lookup (dict): Lookup table for hierarchy paths (required).
    """
    # Hierarchical matching is now mandatory
    if tree is None or path_lookup is None:
        raise ValueError("Hierarchical tree and path_lookup are required for analysis")
    
    assert class_names is not None

    # 1. Data Preparation - optimized with early returns
    if not gts and not result['scores'].numel():
        return  # Nothing to process
    
    gt_bboxes = np.array([gt['bbox'] for gt in gts]) if gts else np.empty((0, 4))
    gt_labels = np.array([gt['bbox_label'] for gt in gts]) if gts else np.empty((0,))
    
    # Vectorized score filtering
    det_scores = result['scores'].numpy()
    score_mask = det_scores >= score_thr
    if not np.any(score_mask):
        # All detections filtered out - add GT to background column
        for g_label in gt_labels:
            confusion_matrix[g_label, confusion_matrix.shape[0] - 1] += 1
        return
    
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

    # Early returns for edge cases
    if G == 0:
        # No GT, all detections are false positives
        np.add.at(confusion_matrix, (num_classes, det_labels), 1)
        return

    if D == 0:
        # No detections, all GT are false negatives
        np.add.at(confusion_matrix, (gt_labels, num_classes), 1)
        return

    # 2. Pre-calculate IoUs
    ious = bbox_overlaps(det_bboxes, gt_bboxes)

    # 3. Pre-compute label mappings for efficiency
    label_to_name = {i: name for i, name in enumerate(class_names)}
    
    # 4. Initialization for Non-Greedy Matching
    gtm_idx = -np.ones(G, dtype=int)
    gt_hf1 = np.zeros(G)
    d_matched = np.zeros(D, dtype=bool)
    dtIg = np.zeros(D, dtype=bool)

    detections_to_process = list(range(D))

    # 5. Iterative Matching with Stealing - optimized
    while detections_to_process:
        dind = detections_to_process.pop(0)
        d_label_name = label_to_name.get(det_labels[dind])
        if not d_label_name:
            continue

        # Pre-filter GT candidates by IoU threshold
        valid_gt_mask = ious[dind, :] >= tp_iou_thr
        valid_gt_indices = np.where(valid_gt_mask)[0]
        
        if len(valid_gt_indices) == 0:
            continue

        best_boost = 1e-9
        best_gt_match_idx = -1
        best_iou = -1.0
        best_unmatched_hf1 = -1.0
        best_unmatched_gt_idx = -1

        for gind in valid_gt_indices:
            gt_label_name = label_to_name.get(int(gt_labels[gind]))
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
                detections_to_process.insert(0, int(prev_d_idx))

            gtm_idx[best_gt_match_idx] = dind
            gt_hf1[best_gt_match_idx] = best_boost + gt_hf1[best_gt_match_idx]
            d_matched[dind] = True
        else:
            # 6. Ancestor-Ignoring Logic
            if best_unmatched_gt_idx != -1:
                final_match_d_idx = gtm_idx[best_unmatched_gt_idx]
                if final_match_d_idx != -1:
                    final_match_d_label = label_to_name.get(int(det_labels[final_match_d_idx]))
                    if final_match_d_label and tree.is_descendant(final_match_d_label, d_label_name):
                        dtIg[dind] = True

    # 7. Update Confusion Matrix - vectorized where possible
    matched_detections = np.where(d_matched)[0]
    for dind in matched_detections:
        matched_gind = np.where(gtm_idx == dind)[0]
        if len(matched_gind) > 0:
            gind = matched_gind[0]
            confusion_matrix[gt_labels[gind], det_labels[dind]] += 1

    # Handle unmatched detections (false positives)
    unmatched_non_ignored = np.where(~d_matched & ~dtIg)[0]
    if len(unmatched_non_ignored) > 0:
        np.add.at(confusion_matrix, (num_classes, det_labels[unmatched_non_ignored]), 1)

    # Handle unmatched GT (false negatives)
    matched_gt = gtm_idx >= 0
    for gind in range(G):
        if not matched_gt[gind]:
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
                          wrap_width=15,
                          paper_size=False):
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
        paper_size (bool): Whether to use paper-sized plots.
    """
    # Normalize the confusion matrix.
    per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    # Handle division by zero for labels with no instances
    confusion_matrix_normalized = np.zeros_like(confusion_matrix, dtype=np.float32)
    np.divide(confusion_matrix, per_label_sums, out=confusion_matrix_normalized, where=per_label_sums!=0)
    confusion_matrix_normalized *= 100

    num_classes = len(labels)

    if paper_size:
        # Paper-ready styling
        fig_width = max(7, 0.3 * num_classes)
        fig_height = max(7, 0.3 * num_classes * 0.9)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
        title_font = {'weight': 'bold', 'size': 12}
        label_font = {'size': 12}
        tick_label_size = 10
        cbar_label_size = 10
        cell_text_size = 7
    else:
        # Default, larger styling for interactive viewing
        fig_width = int(num_classes * 0.5) + 5
        fig_height = int(num_classes * 0.5) + 4
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
        title_font = {'weight': 'bold', 'size': 16}
        label_font = {'size': 14}
        tick_label_size = 12
        cbar_label_size = 12
        cell_text_size = 8

    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix_normalized, cmap=cmap, vmin=0, vmax=100)
    
    # Set colorbar with specific font sizes
    cbar = plt.colorbar(mappable=im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=cbar_label_size)

    ax.set_title(title, fontdict=title_font)
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
        
        for level, _, start, end in boundaries_info_sorted:
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
    ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True, pad=10, labelsize=tick_label_size)
    ax.tick_params(axis='y', labelsize=tick_label_size, pad=10)
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
                j, i, f'{val:.1f}', ha='center', va='center', color=text_color, fontsize=cell_text_size)

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


def prepare_hierarchical_data(dataset, taxonomy_tree):
    """
    Prepare hierarchical data for confusion matrix generation.
    
    Optimized with efficient filtering and single-pass operations.
    """
    original_labels = dataset.metainfo['classes']
    original_label_set = set(original_labels)
    
    # Get leaf hierarchy paths using optimized function
    all_leaf_paths = get_leaf_hierarchy_paths(taxonomy_tree)
    
    # Efficient filtering using set membership
    plottable_leaf_paths = [
        item for item in all_leaf_paths if item[0] in original_label_set
    ]
    
    if not plottable_leaf_paths:
        raise ValueError("No common leaf classes found between taxonomy and dataset")
    
    # Get sorted order and boundaries using optimized function
    sorted_leaf_paths, boundaries = compute_hierarchy_boundaries(plottable_leaf_paths)
    ordered_leaf_names = [item[0] for item in sorted_leaf_paths]
    
    # Efficient set difference for excluded classes
    excluded_classes = original_label_set - set(ordered_leaf_names)
    if excluded_classes:
        print(f"Warning: Excluding non-leaf classes from hierarchical matrix: {sorted(excluded_classes)}")
    
    return ordered_leaf_names, boundaries


def generate_level_confusion_matrix(confusion_matrix, taxonomy_tree, level, dataset_classes, args, cache=None):
    """Generate confusion matrix for a specific aggregation level."""
    # Create cache once if not provided
    if cache is None:
        cache = HierarchyCache(taxonomy_tree)
    
    # Aggregate matrix using cached hierarchy information
    agg_matrix, agg_labels = aggregate_confusion_matrix(
        confusion_matrix, taxonomy_tree, level, dataset_classes, cache
    )
    
    plot_matrix = agg_matrix
    plot_labels = agg_labels
    boundaries = None
    agg_labels_no_bg = agg_labels[:-1]
    
    # Determine title
    max_level = taxonomy_tree.root.get_height()
    if level == max_level:
        title_suffix = "Leaf Level (Full Hierarchy)"
    elif level == 0:
        title_suffix = "Root Level (Most Aggregated)"
    else:
        title_suffix = f"Aggregated Level {level}"
    
    # Compute boundaries for visualization if multiple classes
    if len(agg_labels_no_bg) > 1:
        items_for_boundaries = []
        for label_name in agg_labels_no_bg:
            if label_name in taxonomy_tree.class_to_node:
                ancestors = taxonomy_tree.get_ancestors(label_name)
                items_for_boundaries.append((label_name, ancestors))
            else:
                items_for_boundaries.append((label_name, []))
        
        sorted_items, boundaries = compute_hierarchy_boundaries(items_for_boundaries)
        
        # Reorder matrix if needed
        if boundaries:
            final_labels_no_bg = [item[0] for item in sorted_items]
            if final_labels_no_bg != agg_labels_no_bg:
                plot_matrix, plot_labels = reorder_confusion_matrix(
                    agg_matrix, agg_labels_no_bg, final_labels_no_bg
                )
    
    # Generate plot
    plot_confusion_matrix(
        plot_matrix,
        plot_labels,
        save_path=os.path.join(args.save_dir, f'confusion_matrix_level_{level}.png'),
        show=args.show,
        title=f'Normalized Confusion Matrix ({title_suffix})',
        color_theme=args.color_theme,
        boundaries_info=boundaries,
        paper_size=args.paper_size
    )


def reorder_confusion_matrix(matrix, original_labels_no_bg, new_labels_no_bg):
    """
    Reorder confusion matrix according to new label order.
    
    Optimized with vectorized numpy operations.
    """
    N = len(original_labels_no_bg)
    
    # Create reorder mapping efficiently
    label_to_idx = {label: i for i, label in enumerate(original_labels_no_bg)}
    reorder_indices = np.array([label_to_idx[label] for label in new_labels_no_bg])
    
    # Vectorized reordering using advanced indexing
    new_matrix = np.zeros_like(matrix)
    
    # Reorder main NxN block in one operation
    new_matrix[:N, :N] = matrix[np.ix_(reorder_indices, reorder_indices)]
    
    # Reorder background row and column
    new_matrix[N, :N] = matrix[N, reorder_indices]
    new_matrix[:N, N] = matrix[reorder_indices, N]
    
    # Keep background-to-background unchanged
    new_matrix[N, N] = matrix[N, N]
    
    new_labels = new_labels_no_bg + ['background']
    return new_matrix, new_labels


def main():
    args = parse_args()

    # Load and configure
    cfg = Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))
    
    # Load results and dataset
    results = load(args.prediction_path)
    os.makedirs(args.save_dir, exist_ok=True)
    
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    
    # Validate taxonomy requirement
    if 'taxonomy' not in dataset.metainfo or not dataset.metainfo['taxonomy']:
        raise ValueError("Taxonomy information is required in dataset.metainfo['taxonomy']")
    
    # Calculate raw confusion matrix
    confusion_matrix_raw = calculate_confusion_matrix(
        dataset, results, args.score_thr, args.nms_iou_thr, args.tp_iou_thr
    )
    
    # Prepare hierarchical data
    taxonomy_tree = HierarchyTree(dataset.metainfo['taxonomy'])
    ordered_leaf_names, _ = prepare_hierarchical_data(dataset, taxonomy_tree)
    
    # Create hierarchy cache once for reuse across all levels
    hierarchy_cache = HierarchyCache(taxonomy_tree)
    
    # Generate confusion matrices for all aggregation levels
    max_agg_level = taxonomy_tree.root.get_height()
    print(f"Generating confusion matrices for levels 0 (root) to {max_agg_level} (leaves)")
    
    for level in tqdm(range(max_agg_level + 1), desc="Generating level matrices"):
        if not ordered_leaf_names:
            print(f"No leaf labels to aggregate at level {level}. Skipping.")
            continue
            
        generate_level_confusion_matrix(
            confusion_matrix_raw, taxonomy_tree, level, 
            dataset.metainfo['classes'], args, hierarchy_cache
        )


if __name__ == '__main__':
    main()
