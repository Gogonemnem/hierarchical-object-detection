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


def flatten_taxonomy_with_paths(taxonomy, current_path=None):
    """
    Recursively flatten a nested taxonomy dictionary.
    Returns a list of tuples: (leaf_name, full_path)
    For example, given:
      {"Military Aircraft": {"Fixed-Wing": {"Fighters": ["F-16", "F-22"],
                                             "Bombers": ["B-52", "B-1"]}}}
    This function might return:
      [("F-16", ["Military Aircraft", "Fixed-Wing", "Fighters"]),
       ("F-22", ["Military Aircraft", "Fixed-Wing", "Fighters"]),
       ("B-52", ["Military Aircraft", "Fixed-Wing", "Bombers"]),
       ("B-1",  ["Military Aircraft", "Fixed-Wing", "Bombers"])]
    """
    if current_path is None:
        current_path = []
    flat_list = []
    if isinstance(taxonomy, dict):
        for key, value in taxonomy.items():
            new_path = current_path + [key]
            flat_list.extend(flatten_taxonomy_with_paths(value, new_path))
    elif isinstance(taxonomy, list):
        for item in taxonomy:
            flat_list.append((item, current_path))
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
    for _, path in flat_tax_sorted:
        max_depth = max(max_depth, len(path))

    # For each level from 0 to max_depth-1, group leaves by path[level]
    # If a leaf doesn't have that level, treat it as "Unknown" or skip.
    for level in range(max_depth):
        current_group = None
        start_idx = 0
        for idx, (leaf, path) in enumerate(flat_tax_sorted):
            if len(path) > level:
                group_name = path[level]
            else:
                group_name = None
                # continue

            if group_name != current_group:
                # if we had a previous group, close it out
                if current_group is not None:
                    boundaries_per_level.append(
                        (level, current_group, start_idx, idx - 1)
                    )
                current_group = group_name
                start_idx = idx

        # close the last group
        if current_group is not None:
            boundaries_per_level.append(
                (level, current_group, start_idx, len(flat_tax_sorted) - 1)
            )

    return flat_tax_sorted, boundaries_per_level


def aggregate_confusion_matrix(conf_matrix, flat_tax_sorted, agg_level, ordered_labels):
    """
    Aggregate the confusion matrix at a given hierarchical level.
    
    Args:
        conf_matrix (ndarray): Re-ordered confusion matrix including background.
        flat_tax_sorted (list): List of (leaf, full_path) tuples in the same order as ordered_labels.
        agg_level (int): The hierarchical level at which to aggregate (1 for top-level, etc.).
        ordered_labels (list[str]): List of leaf labels in order, excluding background.
    
    Returns:
        agg_matrix (ndarray): Aggregated confusion matrix including background.
        agg_labels (list[str]): New labels after aggregation, with background appended.
    """
    # Map each leaf to its aggregated group label.
    leaf_to_group = {}
    for (leaf, path), _ in zip(flat_tax_sorted, ordered_labels):
        if len(path) >= agg_level:
            group = path[agg_level - 1]
        else:
            group = leaf  # fallback if not deep enough
        leaf_to_group[leaf] = group

    # Build group mapping: group label -> list of indices in ordered_labels.
    group_to_indices = {}
    for idx, leaf in enumerate(ordered_labels):
        group = leaf_to_group[leaf]
        group_to_indices.setdefault(group, []).append(idx)
    # Background is assumed to be the last row/column.
    bg_idx = conf_matrix.shape[0] - 1

    # Define new aggregated groups sorted alphabetically.
    # agg_groups = sorted(group_to_indices.keys())
    agg_groups = list(group_to_indices.keys())
    num_groups = len(agg_groups)
    # Create aggregated matrix: shape (num_groups+1, num_groups+1) to include background.
    agg_matrix = np.zeros((num_groups + 1, num_groups + 1))
    # Sum over the groups.
    for i, group_i in enumerate(agg_groups):
        indices_i = group_to_indices[group_i]
        for j, group_j in enumerate(agg_groups):
            indices_j = group_to_indices[group_j]
            agg_matrix[i, j] = conf_matrix[np.ix_(indices_i, indices_j)].sum()
        # Background column for group i.
        agg_matrix[i, -1] = conf_matrix[indices_i, bg_idx].sum()
    # For background row: sum false negatives per group.
    for j, group_j in enumerate(agg_groups):
        indices_j = group_to_indices[group_j]
        agg_matrix[-1, j] = conf_matrix[bg_idx, indices_j].sum()
    agg_matrix[-1, -1] = conf_matrix[bg_idx, bg_idx]
    agg_labels = agg_groups + ['background']
    return agg_matrix, agg_labels


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
    confusion_matrix = confusion_matrix.astype(np.float32) / per_label_sums * 100

    num_classes = len(labels)
    # Compute dynamic figure size but set minimum dimensions.
    fig_width = max(8, 0.5 * num_classes)
    fig_height = max(8, 0.5 * num_classes * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)

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
    if boundaries_info is not None:
        import matplotlib.colors as mcolors
        color_list = list(mcolors.TABLEAU_COLORS.keys())
        line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
        boundaries_info_sorted = sorted(boundaries_info, key=lambda x: x[0])
        max_depth = max([depth for depth, *_ in boundaries_info_sorted])
        for (level, group_name, start_idx, end_idx) in boundaries_info_sorted:
            c = mcolors.TABLEAU_COLORS[color_list[level % len(color_list)]]
            style = line_styles[level % len(line_styles)]
            line_width = max(1, 8 - 2 * level)
            ax.axhline(y=start_idx - 0.5, color=c, linewidth=line_width, linestyle=style)
            ax.axhline(y=end_idx + 0.5, color=c, linewidth=line_width, linestyle=style)
            ax.axvline(x=start_idx - 0.5, color=c, linewidth=line_width, linestyle=style)
            ax.axvline(x=end_idx + 0.5, color=c, linewidth=line_width, linestyle=style)
            mid_point = (start_idx + end_idx) / 2.0
            shift = -np.log2(max_depth - level + 2) * 5
            ax.text(mid_point, shift, group_name, ha='center', va='center',
                    fontsize=8, fontweight='bold', color=c, rotation=45)
            ax.text(shift, mid_point, group_name, ha='center', va='center',
                    fontsize=8, fontweight='bold', color=c, rotation=0)
            
    # Wrap long labels.
    wrapped_labels = [textwrap.fill(label, width=wrap_width) for label in labels]
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(wrapped_labels)
    ax.set_yticklabels(wrapped_labels)

    # # draw label
    # ax.set_xticks(np.arange(num_classes))
    # ax.set_yticks(np.arange(num_classes))
    # ax.set_xticklabels(labels)
    # ax.set_yticklabels(labels)
    
    # Adjust tick parameters.
    ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True, pad=10, labelsize=10)
    ax.tick_params(axis='y', labelsize=10, pad=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # Draw the numeric values inside each cell.
    for i in range(num_classes):
        for j in range(num_classes):
            val = int(confusion_matrix[i, j]) if not np.isnan(confusion_matrix[i, j]) else -1
            ax.text(j, i, f'{val}%', ha='center', va='center', color='w', size=7)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1
    # Adjust margins so labels have room.
    plt.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.3)

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(
            save_path, format='png')
    if show:
        plt.show()


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
    
    # Reorder using taxonomy if available
    if 'taxonomy' in dataset.metainfo:
        flat_tax = flatten_taxonomy_with_paths(dataset.metainfo['taxonomy'])
        # Build nested boundaries (which sorts the flat taxonomy)
        flat_tax_sorted, boundaries_per_level = build_nested_boundaries(flat_tax)
        
        # Derive hierarchical order from the sorted taxonomy
        hierarchical_order = [leaf for leaf, _ in flat_tax_sorted]
        classes = list(dataset.metainfo['classes'])
        
        # Map hierarchical order to indices
        new_order = []
        for cls in hierarchical_order:
            if cls in classes:
                new_order.append(classes.index(cls))
        new_order.append(len(classes))  # background index
        
        # Reorder confusion matrix
        confusion_matrix = confusion_matrix[new_order, :][:, new_order]
        ordered_labels = [classes[i] for i in new_order[:-1]] + ['background']
        
        # Remap the boundaries using the same flat_tax_sorted order:
        leaf_to_index = {}
        for idx, leaf in enumerate(ordered_labels[:-1]):  # skip background
            leaf_to_index[leaf] = idx

        boundaries_info = []
        for (level, group_name, start_idx, end_idx) in boundaries_per_level:
            leaf_start, _ = flat_tax_sorted[start_idx]
            leaf_end, _ = flat_tax_sorted[end_idx]
            if leaf_start not in leaf_to_index or leaf_end not in leaf_to_index:
                continue
            new_start = leaf_to_index[leaf_start]
            new_end = leaf_to_index[leaf_end]
            s, e = sorted([new_start, new_end])
            boundaries_info.append((level, group_name, s, e))

    else:
        ordered_labels = list(dataset.metainfo['classes']) + ['background']
        boundaries_info = None

    # Branch based on mode.
    if args.mode == 'leaf':
        # Only leaf nodes, no hierarchical boundaries.
        plot_confusion_matrix(confusion_matrix, ordered_labels, 
                                save_path=os.path.join(args.save_dir, 'confusion_matrix_leaf.png'),
                                show=args.show, color_theme=args.color_theme, boundaries_info=None)
    elif args.mode == 'hierarchy':
        # Leaf nodes with hierarchical boundaries drawn.
        plot_confusion_matrix(confusion_matrix, ordered_labels, 
                                save_path=os.path.join(args.save_dir, 'confusion_matrix_hierarchy.png'),
                                show=args.show, color_theme=args.color_theme, boundaries_info=boundaries_info)
    elif args.mode == 'aggregate':
        # Aggregate and plot confusion matrices at every hierarchical level.
        if 'taxonomy' not in dataset.metainfo:
            print("No taxonomy available for aggregation. Exiting.")
            return
        # Remove background from ordered labels for aggregation.
        ordered_labels_leaf = ordered_labels[:-1]
        # Determine maximum depth.
        max_depth = max(len(path) for _, path in flat_tax_sorted)
        for level in range(1, max_depth + 2):
            agg_matrix, agg_labels = aggregate_confusion_matrix(confusion_matrix, flat_tax_sorted, level, ordered_labels_leaf)
            title = f'Aggregated Confusion Matrix at Level {level}'
            save_path = os.path.join(args.save_dir, f'confusion_matrix_aggregate_level_{level}.png')
            plot_confusion_matrix(agg_matrix, agg_labels, save_path=save_path,
                                    show=args.show, title=title, color_theme=args.color_theme,
                                    boundaries_info=None)
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
