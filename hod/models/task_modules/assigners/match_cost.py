from typing import Optional, Union, List, Dict

import torch
from torch import Tensor
from mmengine.fileio import load # For loading annotation file

from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules.assigners.match_cost import FocalLossCost # Inherit from this
from hod.utils.tree import HierarchyTree

@TASK_UTILS.register_module()
class HierarchicalFocalLossCost(FocalLossCost):
    """Hierarchical FocalLossCost.

    This cost function considers the class hierarchy when calculating the
    classification cost for matching. It uses an ancestor path mask and
    class level weights, similar to HierarchicalFocalLoss.

    Args:
        ann_file (str): Path to the annotation file containing the taxonomy
            and category information.
        decay (float): Decay factor for calculating class_level_weight.
            Controls the emphasis on leaf nodes.
        num_classes (int): Number of foreground classes the model predicts.
            This is crucial for sizing tensors correctly.
        alpha (Union[float, int]): Alpha parameter for focal loss.
            Defaults to 0.25.
        gamma (Union[float, int]): Gamma parameter for focal loss.
            Defaults to 2.
        eps (float): Epsilon for numerical stability in log.
            Defaults to 1e-12.
        weight (Union[float, int]): Overall weight for this match cost
            component. Defaults to 1.
    """

    def __init__(self,
                 ann_file: str,
                 decay: float = 1.0,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        assert self.binary_input is False, "HierarchicalFocalLossCost only supports binary_input=False"

        self.decay = decay

        self.ancestor_path_target_mask: Optional[Tensor] = None
        self.class_level_weight: Optional[Tensor] = None
        self.load_taxonomy(ann_file)

    def load_taxonomy(self, ann_file):
        ann = load(ann_file)
        cats = ann['categories']
        # build name<->idx map
        self.class_to_idx = {c['name']: c['id'] for c in cats}

        # build the tree
        taxonomy = ann.get('taxonomy', {})
        self.tree = HierarchyTree(taxonomy)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        C = len(cats)
        self.ancestor_path_target_mask = torch.zeros(
            (C+1, C),
            dtype=torch.bool,
            device=device
        )
        self.class_level_weight = torch.zeros((C,), device=device)

        # differs from paper: path_weight = 1/2
        total_path_weight  = 1.0 # changing it equals changing loss_weight

        for cls, node in self.tree.class_to_node.items():
            if cls not in self.class_to_idx:
                continue
            idx = self.class_to_idx[cls]
            # get the path from the root
            path_cls = self.tree.get_path(cls)
            path_idx = [self.class_to_idx[p] for p in path_cls]

            self.ancestor_path_target_mask[idx, path_idx] = True

            # Compute the node's weight contribution using exponential decay
            height = node.get_height()
            node_weight_contribution  = (1-self.decay)/(1-self.decay**(height+1)) if self.decay != 1 else 1/(height+1)

            # differs from the original paper, root class a weight != 0
            # we also slice the node itself but the weight is zero at init
            ancestor_weight = self.class_level_weight[path_idx].sum()
            available_weight = total_path_weight - ancestor_weight

            # Assign the final weight to the current class
            class_weight = node_weight_contribution  * available_weight
            self.class_level_weight[idx] = class_weight

    def _focal_loss_cost(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_queries, num_model_classes).
            gt_labels (Tensor): Ground truth labels, shape (num_gt,).
                Expected to be 0-indexed (0 to num_model_classes-1).

        Returns:
            torch.Tensor: Hierarchical classification cost matrix of shape
                (num_queries, num_gt).
        """
        # Ensure tensors are on the same device
        current_device = cls_pred.device
        if self.ancestor_path_target_mask.device != current_device:
            self.ancestor_path_target_mask = self.ancestor_path_target_mask.to(current_device)
        if self.class_level_weight.device != current_device:
            self.class_level_weight = self.class_level_weight.to(current_device)

        p = cls_pred.sigmoid()  # (num_queries, num_model_classes)

        # Get hierarchical targets for the given gt_labels
        # gt_labels are indices for rows of ancestor_path_target_mask
        hierarchical_gt_targets = self.ancestor_path_target_mask[gt_labels]
        # hierarchical_gt_targets shape: (num_gt, num_model_classes)

        # Focal loss components (element-wise for each query and class)
        log_p = torch.log(p.clamp(min=self.eps, max=1.0 - self.eps))
        log_1_p = torch.log((1.0 - p).clamp(min=self.eps, max=1.0 - self.eps))

        # Cost for positive targets (target_ij = 1)
        pos_cost_terms = -self.alpha * ((1.0 - p).pow(self.gamma)) * log_p
        # Cost for negative targets (target_ij = 0)
        neg_cost_terms = -(1.0 - self.alpha) * (p.pow(self.gamma)) * log_1_p

        # MMDetection FocalLossCost style: diff = pos_cost_term - neg_cost_term
        # This is the "cost" for predicting a class c if c were the single true label.
        mmdet_style_class_cost = pos_cost_terms - neg_cost_terms # (num_queries, num_model_classes)

        # Apply hierarchical class_level_weight
        weighted_mmdet_style_class_cost = mmdet_style_class_cost * self.class_level_weight.unsqueeze(0)

        # Get hierarchical targets for the given gt_labels
        hierarchical_gt_targets = self.ancestor_path_target_mask[gt_labels]
        # hierarchical_gt_targets shape: (num_gt, num_model_classes)
        ht_float = hierarchical_gt_targets.float()

        # For each (query, gt_k), sum the weighted_mmdet_style_class_cost
        # ONLY for classes c that are positive in the hierarchy for gt_k.
        # cls_cost[q,k] = sum_c { weighted_mmdet_style_class_cost[q,c] * ht_float[k,c] }
        cls_cost = torch.matmul(weighted_mmdet_style_class_cost, ht_float.T) # (Q,C) @ (C,G) -> (Q,G)

        return cls_cost * self.weight

    # The __call__ method is inherited from FocalLossCost.
    # It checks self.binary_input. Since we initialize super() with binary_input=False,
    # it will correctly call self._focal_loss_cost(pred_instances.scores, gt_instances.labels).
