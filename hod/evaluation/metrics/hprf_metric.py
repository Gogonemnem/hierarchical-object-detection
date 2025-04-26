import os
import json
import numpy as np
from typing import Dict

from terminaltables import AsciiTable

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.registry import METRICS

from mmdet.datasets.api_wrappers import COCO
from mmdet.evaluation import bbox_overlaps

from hod.utils.tree import HierarchyTree

@METRICS.register_module()
class HierarchicalPRFMetric(BaseMetric):
    def __init__(self,
                 taxonomy: Dict,
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
        self.hierarchy = HierarchyTree(taxonomy)
        self.iou_thr = iou_thr
        self.score_thr = score_thr
        self.classwise = classwise
        self.format_only = format_only
        self.outfile_prefix = outfile_prefix
        self.ann_file = ann_file

        self.total_TP = 0.0  # Now partial TP allowed
        self.total_FP = 0.0
        self.total_FN = 0.0
        self.class_stats = {}

        self.class_names = None
        self.label_to_name = None
        if self.ann_file is not None:
            try:
                coco_api = COCO(self.ann_file)
                cats = coco_api.loadCats(coco_api.getCatIds())
                self.label_to_name = {cat['id']: cat['name'] for cat in cats}
                self.class_names = self.label_to_name  # Optional: expose under this name too
            except Exception as e:
                MMLogger.get_current_instance().warning(
                    f"Failed to load annotation file {self.ann_file}: {e}")

    def _hscore(self, pred_label, gt_label):
        if self.label_to_name is not None:
            pred_label = self.label_to_name.get(int(pred_label), str(pred_label))
            gt_label = self.label_to_name.get(int(gt_label), str(gt_label))
        pred_path = self.hierarchy.get_path(pred_label)
        gt_path = self.hierarchy.get_path(gt_label)
        common = set(pred_path) & set(gt_path)
        hprec = len(common) / len(pred_path)
        hrec = len(common) / len(gt_path)
        hf1 = (2 * hprec * hrec / (hprec + hrec + 1e-6)) if (hprec + hrec) > 0 else 0.0
        return hprec, hrec, hf1

    def process(self, data_batch, data_samples):
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

            if len(gt_labels) == 0:
                self.total_FP += len(pred_labels)
                continue

            if len(pred_labels) == 0:
                self.total_FN += len(gt_labels)
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
                best_score = 0.0
                best_j = -1
                for j, (gt_box, gt_label) in enumerate(zip(gt_bboxes, gt_labels)):
                    if matched[j]:
                        continue
                    if ious[i, j] < self.iou_thr:
                        continue
                    metrics = hierarchical_prf_metric(self.hierarchy, p_label, gt_label, self.label_to_name)
                    f1 = metrics['hf1']
                    if f1 > best_score:
                        best_score = f1
                        best_j = j
                if best_j >= 0:
                    matched[best_j] = True
                    self.total_TP += best_score
                    self.total_FN += 1.0 - best_score
                    if self.classwise:
                        self.class_stats.setdefault(p_label, {'TP': 0.0, 'FP': 0.0, 'FN': 0.0})
                        self.class_stats[p_label]['TP'] += best_score
                        self.class_stats[p_label]['FN'] += 1.0 - best_score
                else:
                    self.total_FP += 1
                    if self.classwise:
                        self.class_stats.setdefault(p_label, {'TP': 0.0, 'FP': 0.0, 'FN': 0.0})
                        self.class_stats[p_label]['FP'] += 1

            # Any unmatched GTs = full FN
            for j, matched_flag in enumerate(matched):
                if not matched_flag:
                    self.total_FN += 1
                    if self.classwise:
                        self.class_stats.setdefault(gt_labels[j], {'TP': 0.0, 'FP': 0.0, 'FN': 0.0})
                        self.class_stats[gt_labels[j]]['FN'] += 1

    def compute_metrics(self, results) -> Dict[str, float]:
        logger = MMLogger.get_current_instance()

        # Micro metrics
        micro_precision = self.total_TP / (self.total_TP + self.total_FP + 1e-6)
        micro_recall = self.total_TP / (self.total_TP + self.total_FN + 1e-6)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-6)

        metrics = {
            'hierarchical_micro_precision': micro_precision,
            'hierarchical_micro_recall': micro_recall,
            'hierarchical_micro_f1': micro_f1,
        }

        # Macro & weighted if per-class stats are enabled
        if self.classwise and self.class_stats:
            per_class_precisions = []
            per_class_recalls = []
            per_class_supports = []

            table_data = [['Class', 'Precision', 'Recall', 'F1']]

            for cls, stats in sorted(self.class_stats.items(), key=lambda x: x[0]):
                tp = stats['TP']
                fp = stats['FP']
                fn = stats['FN']

                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = 2 * precision * recall / (precision + recall + 1e-6) if (precision + recall) > 0 else 0.0
                support = tp + fn  # support = number of actual positives

                per_class_precisions.append(precision)
                per_class_recalls.append(recall)
                per_class_supports.append(support)

                class_name  = self.label_to_name.get(int(cls), str(cls)) if self.label_to_name else cls
                table_data.append([class_name, f'{precision:.4f}', f'{recall:.4f}', f'{f1:.4f}'])

            macro_precision = np.mean(per_class_precisions)
            macro_recall = np.mean(per_class_recalls)
            macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall + 1e-6)

            total_support = sum(per_class_supports)
            weighted_precision = sum(p * s for p, s in zip(per_class_precisions, per_class_supports)) / (total_support + 1e-6)
            weighted_recall = sum(r * s for r, s in zip(per_class_recalls, per_class_supports)) / (total_support + 1e-6)
            weighted_f1 = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall + 1e-6)

            metrics.update({
                'hierarchical_macro_precision': macro_precision,
                'hierarchical_macro_recall': macro_recall,
                'hierarchical_macro_f1': macro_f1,
                'hierarchical_weighted_precision': weighted_precision,
                'hierarchical_weighted_recall': weighted_recall,
                'hierarchical_weighted_f1': weighted_f1,
            })

            # Logging table
            overall_table = AsciiTable([
                ['Aggregate', 'Precision', 'Recall', 'F1'],
                ['Micro', f'{micro_precision:.4f}', f'{micro_recall:.4f}', f'{micro_f1:.4f}'],
                ['Macro', f'{macro_precision:.4f}', f'{macro_recall:.4f}', f'{macro_f1:.4f}'],
                ['Weighted', f'{weighted_precision:.4f}', f'{weighted_recall:.4f}', f'{weighted_f1:.4f}']
            ])
            logger.info('\nHierarchical Aggregate Metrics:\n' + overall_table.table)

            per_class_table = AsciiTable(table_data)
            logger.info('\nHierarchical Per-Class Metrics:\n' + per_class_table.table)

        return metrics


def hierarchical_prf_metric(
    hierarchy_tree: HierarchyTree,
    dt_label,
    gt_label,
    label_to_name: Dict = None,
    return_paths: bool = False
):
    """Compute hierarchical precision, recall, and F1 between predicted and GT labels.

    Optionally returns the raw paths for additional use in aggregation.
    """
    if label_to_name is not None:
        dt_label = label_to_name.get(int(dt_label), str(dt_label))
        gt_label = label_to_name.get(int(gt_label), str(gt_label))

    dt_path = hierarchy_tree.get_path(dt_label)
    gt_path = hierarchy_tree.get_path(gt_label)
    overlap = set(dt_path) & set(gt_path)
    hprecision = len(overlap) / len(dt_path)
    hrecall = len(overlap) / len(gt_path)
    hf1 = (2 * hprecision * hrecall / (hprecision + hrecall + 1e-6)) if (hprecision + hrecall) > 0 else 0.0
    results = {
        'hpr': hprecision,
        'hr': hrecall,
        'hf1': hf1,
    }
    if return_paths:
        results.update({
            'len_overlap': len(overlap),
            'len_dt': len(dt_path),
            'len_gt': len(gt_path),
            'pred_path': dt_path,
            'gt_path': gt_path,
        })
    return results
