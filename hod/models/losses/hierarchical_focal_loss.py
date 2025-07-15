import torch

from mmdet.registry import MODELS
from mmdet.models.losses.focal_loss import FocalLoss, py_sigmoid_focal_loss
from mmdet.models.losses.gfocal_loss import QualityFocalLoss, quality_focal_loss

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
        label_masked, score, weight = self.hierarchical_mask_and_weight(target, weight)
        hier_target = (label_masked, score)
        loss_qfl = self.loss_weight * quality_focal_loss(
            pred,
            hier_target,
            beta=self.beta,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
            activated=self.activated
        )
        return loss_qfl
