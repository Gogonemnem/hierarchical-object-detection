from functools import partial
import torch
from torch.nn import functional as F

from mmdet.registry import MODELS
from mmdet.models.losses.focal_loss import FocalLoss, py_sigmoid_focal_loss
from mmdet.models.losses.gfocal_loss import QualityFocalLoss
from mmdet.models.losses.utils import weight_reduce_loss

from hod.models.losses.hierarchical_loss import HierarchicalDataMixin

class HierarchicalLossBase(HierarchicalDataMixin):
    def __init__(self, ann_file, decay=1):
        HierarchicalDataMixin.__init__(self, ann_file=ann_file)
        self.decay = decay
        self.post_process_taxonomy()

    def post_process_taxonomy(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        C = len(self.class_to_idx)
        self.ancestor_path_target_mask = torch.zeros(
            (C+1, C),
            dtype=torch.bool,
            device=device
        )
        self.class_level_weight = torch.zeros((C,), device=device)
        total_path_weight  = 1.0
        for cls, node in self.tree.class_to_node.items():
            if cls not in self.class_to_idx:
                continue
            idx = self.class_to_idx[cls]
            path_cls = self.tree.get_path(cls)
            path_idx = [self.class_to_idx[p] for p in path_cls]
            self.ancestor_path_target_mask[idx, path_idx] = True
            height = node.get_height()
            node_weight_contribution  = (1-self.decay)/(1-self.decay**(height+1)) if self.decay != 1 else 1/(height+1)
            ancestor_weight = self.class_level_weight[path_idx].sum()
            available_weight = total_path_weight - ancestor_weight
            class_weight = node_weight_contribution  * available_weight
            self.class_level_weight[idx] = class_weight

    def hierarchical_mask_and_weight(self, target, weight=None):
        """
        Returns masked target and weighted loss mask for hierarchical losses.
        For FocalLoss: target is (N,) or (N,C). For QFL: target is (label, score).
        """
        if isinstance(target, (tuple, list)) and len(target) == 2:
            label, score = target
            label_masked = self.ancestor_path_target_mask[label]
        else:
            label_masked = self.ancestor_path_target_mask[target]
            score = None
        # Weight broadcasting and masking
        if weight is not None:
            if weight.shape != label_masked.shape:
                if weight.size(0) == label_masked.size(0):
                    weight = weight.view(-1, 1)
                else:
                    assert weight.numel() == label_masked.numel()
                    weight = weight.view(label_masked.size(0), -1)
            assert weight.ndim == label_masked.ndim
            weight = weight * self.class_level_weight
        return label_masked, score, weight

@MODELS.register_module()
class HierarchicalFocalLoss(FocalLoss, HierarchicalLossBase):
    def __init__(self, ann_file, decay=1, **kwargs):
        FocalLoss.__init__(self, **kwargs)
        HierarchicalLossBase.__init__(self, ann_file=ann_file, decay=decay)

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            label_masked, _, weight = self.hierarchical_mask_and_weight(target, weight)
            loss_cls = self.loss_weight * py_sigmoid_focal_loss(
                pred,
                label_masked,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls

def quality_focal_loss_tensor_target(pred, target, weight=None, beta=2.0, reduction='mean', avg_factor=None, activated=False, ):
    """`QualityFocal Loss <https://arxiv.org/abs/2008.13367>`_
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        activated (bool): Whether the input is activated.
            If True, it means the input has been activated and can be
            treated as probabilities. Else, it should be treated as logits.
            Defaults to False.
    """
    # pred and target should be of the same size
    assert pred.size() == target.size()
    if activated:
        pred_sigmoid = pred
        loss_function = F.binary_cross_entropy
    else:
        pred_sigmoid = pred.sigmoid()
        loss_function = F.binary_cross_entropy_with_logits

    target = target.type_as(pred)

    # 1. Negative loss
    neg_loss = loss_function(pred, torch.zeros_like(pred), reduction='none')
    neg_mod = pred_sigmoid.pow(beta)
    loss = neg_loss * neg_mod

    # 2. Positive loss
    pos = target != 0
    pos_loss = loss_function(pred[pos], target[pos], reduction='none')
    pos_mod = (target[pos] - pred_sigmoid[pos]).abs().pow(beta)
    loss[pos] = pos_loss * pos_mod


    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

@MODELS.register_module()
class HierarchicalQualityFocalLoss(QualityFocalLoss, HierarchicalLossBase):
    def __init__(self, ann_file, decay=1, **kwargs):
        QualityFocalLoss.__init__(self, **kwargs)
        HierarchicalLossBase.__init__(self, ann_file=ann_file, decay=decay)

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if not (isinstance(target, (tuple, list)) and len(target) == 2):
            raise ValueError("target for QualityFocalLoss must be a tuple (label, score)")
        # Use label_masked and weight to build dense soft labels
        label_masked, score, weight = self.hierarchical_mask_and_weight(target, weight)
        # Option 1: Soft labels
        # target_dense = weight * score.view(-1, 1)

        # Option 2: Hard labels
        target_dense = label_masked.float() * score.view(-1, 1)
        # label, score = target
        # bg_ratio = (label == 111).float().mean().item()
        # print(f"Fraction background: {bg_ratio:.4f}")

        loss = quality_focal_loss_tensor_target(
            pred,
            target_dense,
            weight=weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            activated=self.activated
        )
                
        return self.loss_weight * loss
