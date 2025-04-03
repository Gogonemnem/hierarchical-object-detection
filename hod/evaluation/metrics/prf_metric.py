import os
import json
import numpy as np
import torch
from typing import Dict
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from mmdet.evaluation import bbox_overlaps
from mmengine.fileio import dump
from terminaltables import AsciiTable
from mmengine.logging import MMLogger

@METRICS.register_module()
class PRFMetric(BaseMetric):
    """A flat precision/recall/F1 metric for object detection.

    This metric computes precision, recall, and F1 score at fixed IoU and
    confidence thresholds across the entire dataset. Optionally, it computes
    per-class metrics and aggregates them (micro, macro, weighted).

    Args:
        iou_thr (float): IoU threshold to consider a detection as a match.
            Default: 0.5.
        score_thr (float): Score threshold to filter detections.
            Default: 0.3.
        classwise (bool): Whether to compute metrics for each class individually.
            Default: True.
        format_only (bool): If True, only format and save the results.
            Default: False.
        outfile_prefix (str, optional): Prefix for saving results. If provided,
            results are dumped to a JSON file.
        collect_device (str): Device for result collection ('cpu' or 'gpu').
            Default: 'cpu'.
        prefix (str, optional): Prefix added to metric names.
            Default: None.
        ann_file (str, optional): Path to the annotation file in COCO format.
            This is used to load class names.
    """
    def __init__(self,
                 iou_thr=0.5,
                 score_thr=0.3,
                 classwise=True,
                 format_only=False,
                 outfile_prefix=None,
                 collect_device='cpu',
                 prefix=None,
                 ann_file=None,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thr = iou_thr
        self.score_thr = score_thr
        self.classwise = classwise
        self.format_only = format_only
        self.outfile_prefix = outfile_prefix
        self.ann_file = ann_file

        # Global accumulators for micro metrics.
        self.total_TP = 0
        self.total_FP = 0
        self.total_FN = 0

        # For per-class statistics: dict mapping class id to counts.
        self.class_stats = {}

        # Try to load COCO API to obtain class names.
        self._coco_api = None
        self.class_names = None
        if self.ann_file is not None:
            from mmdet.datasets.api_wrappers import COCO
            try:
                self._coco_api = COCO(self.ann_file)
                # Build mapping: category id -> category name.
                self.class_names = {
                    cat['id']: cat['name'] for cat in self._coco_api.dataset.get('categories', [])
                }
            except Exception as e:
                MMLogger.get_current_instance().warning(
                    f'Failed to load annotation file {self.ann_file}: {e}')
                self.class_names = None

    def process(self, data_batch, data_samples):
        """Process one batch of data samples.

        Args:
            data_batch (dict): Batch of data from the dataloader (unused).
            data_samples (Sequence[dict]): List of samples containing predictions 
                ('pred_instances') and ground truths ('gt_instances').
        """
        for sample in data_samples:
            # Process predictions.
            pred = sample['pred_instances']
            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()

            # Filter predictions.
            valid_inds = pred_scores >= self.score_thr
            pred_bboxes = pred_bboxes[valid_inds]
            pred_scores = pred_scores[valid_inds]
            pred_labels = pred_labels[valid_inds]

            # Process ground truth.
            gt = sample['gt_instances']
            gt_bboxes = (gt['bboxes'].cpu().numpy() 
                         if hasattr(gt['bboxes'], 'cpu') else np.array(gt['bboxes']))
            gt_labels = (gt['labels'].cpu().numpy() 
                         if hasattr(gt['labels'], 'cpu') else np.array(gt['labels']))

            matched = np.zeros(len(gt_labels), dtype=bool)

            # If no predictions, count all GT as FN.
            if len(pred_labels) == 0:
                self.total_FN += len(gt_labels)
                if self.classwise:
                    for label in gt_labels:
                        self.class_stats.setdefault(label, {'TP': 0, 'FP': 0, 'FN': 0})
                        self.class_stats[label]['FN'] += 1
                continue

            # Sort predictions by score.
            order = np.argsort(-pred_scores)
            pred_bboxes = pred_bboxes[order]
            pred_labels = pred_labels[order]

            # Compute IoU between predictions and GT.
            if len(gt_bboxes) > 0:
                ious = bbox_overlaps(pred_bboxes, gt_bboxes)
            else:
                ious = np.empty((len(pred_bboxes), 0))

            # Greedy matching.
            for i, (p_box, p_label) in enumerate(zip(pred_bboxes, pred_labels)):
                found_match = False
                if ious.shape[1] > 0:
                    iou_vals = ious[i]
                    candidate_inds = np.where((iou_vals >= self.iou_thr) &
                                              (gt_labels == p_label) &
                                              (~matched))[0]
                    if candidate_inds.size > 0:
                        best_idx = candidate_inds[np.argmax(iou_vals[candidate_inds])]
                        matched[best_idx] = True
                        self.total_TP += 1
                        found_match = True
                        if self.classwise:
                            self.class_stats.setdefault(p_label, {'TP': 0, 'FP': 0, 'FN': 0})
                            self.class_stats[p_label]['TP'] += 1
                if not found_match:
                    self.total_FP += 1
                    if self.classwise:
                        self.class_stats.setdefault(p_label, {'TP': 0, 'FP': 0, 'FN': 0})
                        self.class_stats[p_label]['FP'] += 1

            # Unmatched GT are FN.
            fn_count = np.sum(~matched)
            self.total_FN += fn_count
            if self.classwise:
                for label in gt_labels[~matched]:
                    self.class_stats.setdefault(label, {'TP': 0, 'FP': 0, 'FN': 0})
                    self.class_stats[label]['FN'] += 1

    def compute_metrics(self, results) -> Dict[str, float]:
        """Compute and return aggregated precision, recall, and F1 metrics.

        Args:
            results (list): Processed results (unused here).

        Returns:
            Dict[str, float]: Dictionary containing micro, macro, and weighted metrics,
            and per-class metrics if classwise is enabled.
        """
        logger = MMLogger.get_current_instance()

        # Micro metrics.
        micro_precision = self.total_TP / (self.total_TP + self.total_FP + 1e-6)
        micro_recall = self.total_TP / (self.total_TP + self.total_FN + 1e-6)
        micro_f1 = (2 * micro_precision * micro_recall /
                    (micro_precision + micro_recall + 1e-6)
                    if (micro_precision + micro_recall) > 0 else 0.0)

        metrics = {
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
        }

        # Compute macro and weighted metrics if classwise is enabled.
        if self.classwise and self.class_stats:
            per_class_precisions = []
            per_class_recalls = []
            for cls, stats in self.class_stats.items():
                TP = stats['TP']
                FP = stats['FP']
                FN = stats['FN']
                precision = TP / (TP + FP + 1e-6)
                recall = TP / (TP + FN + 1e-6)
                f1 = (2 * precision * recall / (precision + recall + 1e-6)
                      if (precision + recall) > 0 else 0.0)
                # Use class name if available.
                if self.class_names is not None:
                    class_name = self.class_names.get(cls, str(cls))
                else:
                    class_name = str(cls)
                metrics[f'{class_name}_precision'] = precision
                metrics[f'{class_name}_recall'] = recall
                metrics[f'{class_name}_f1'] = f1
                per_class_precisions.append(precision)
                per_class_recalls.append(recall)
            macro_precision = np.mean(per_class_precisions)
            macro_recall = np.mean(per_class_recalls)
            macro_f1 = (2 * macro_precision * macro_recall /
                        (macro_precision + macro_recall + 1e-6)
                        if (macro_precision + macro_recall) > 0 else 0.0)
            metrics['macro_precision'] = macro_precision
            metrics['macro_recall'] = macro_recall
            metrics['macro_f1'] = macro_f1

            total_support = sum(stats['TP'] + stats['FN'] for stats in self.class_stats.values())
            weighted_precision = sum(((stats['TP'] / (stats['TP'] + stats['FP'] + 1e-6)) *
                                      (stats['TP'] + stats['FN']))
                                     for stats in self.class_stats.values()) / (total_support + 1e-6)
            weighted_recall = sum(((stats['TP'] / (stats['TP'] + stats['FN'] + 1e-6)) *
                                   (stats['TP'] + stats['FN']))
                                  for stats in self.class_stats.values()) / (total_support + 1e-6)
            weighted_f1 = (2 * weighted_precision * weighted_recall /
                           (weighted_precision + weighted_recall + 1e-6)
                           if (weighted_precision + weighted_recall) > 0 else 0.0)
            metrics['weighted_precision'] = weighted_precision
            metrics['weighted_recall'] = weighted_recall
            metrics['weighted_f1'] = weighted_f1

            # Build an aggregated table with micro, macro, and weighted metrics.
            overall_table_data = [
                ['Aggregate', 'Precision', 'Recall', 'F1'],
                ['Micro', f'{micro_precision:.4f}', f'{micro_recall:.4f}', f'{micro_f1:.4f}'],
                ['Macro', f'{macro_precision:.4f}', f'{macro_recall:.4f}', f'{macro_f1:.4f}'],
                ['Weighted', f'{weighted_precision:.4f}', f'{weighted_recall:.4f}', f'{weighted_f1:.4f}']
            ]
        else:
            # If classwise is not enabled, only micro metrics are available.
            overall_table_data = [
                ['Aggregate', 'Precision', 'Recall', 'F1'],
                ['Micro', f'{micro_precision:.4f}', f'{micro_recall:.4f}', f'{micro_f1:.4f}']
            ]

        overall_table = AsciiTable(overall_table_data)
        logger.info('\nAggregated Metrics:\n' + overall_table.table)

        # Optionally, print per-class metrics if classwise is enabled.
        if self.classwise and self.class_stats:
            per_class_table_data = [['Class', 'Precision', 'Recall', 'F1']]
            for cls, stats in sorted(self.class_stats.items(), key=lambda x: x[0]):
                TP = stats['TP']
                FP = stats['FP']
                FN = stats['FN']
                precision = TP / (TP + FP + 1e-6)
                recall = TP / (TP + FN + 1e-6)
                f1 = (2 * precision * recall / (precision + recall + 1e-6)
                      if (precision + recall) > 0 else 0.0)
                if self.class_names is not None:
                    class_name = self.class_names.get(cls, str(cls))
                else:
                    class_name = str(cls)
                per_class_table_data.append([class_name, f'{precision:.4f}', f'{recall:.4f}', f'{f1:.4f}'])
            per_class_table = AsciiTable(per_class_table_data)
            logger.info('\nPer-Class Metrics:\n' + per_class_table.table)

        # If format_only is True and outfile_prefix is provided, dump the metrics.
        if self.format_only and self.outfile_prefix is not None:
            save_path = os.path.join(self.outfile_prefix, 'prf_metrics.json')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(metrics, f, indent=4)
        return metrics
