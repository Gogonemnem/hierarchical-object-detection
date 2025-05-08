import torch

from mmengine.fileio import load
from mmdet.registry import MODELS
from mmdet.models.losses.focal_loss import FocalLoss
from mmdet.models.losses.focal_loss import py_sigmoid_focal_loss

from hod.utils.tree import HierarchyTree

@MODELS.register_module()
class HierarchicalFocalLoss(FocalLoss):
    def __init__(self,
                 ann_file='',
                 decay=1,
                 **kwargs):
        """
        kwargs are forwarded to FocalLoss,
        e.g. `num_classes=len(class_to_idx)`, `ignore_index=...`, etc.
        """
        super().__init__(**kwargs)
        # we'll keep a little cache of each parent-child index list
        self.decay = decay
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

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function. Very similar to the original Focal Loss forward
        function, but with some changes to support the hierarchical loss.
        The main difference is that we use the ancestor_path_target_mask to
        calculate the loss for each class in the hierarchy. Also, we use the
        class_level_weight to calculate the weight for each class in the
        hierarchy.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
                The target shape support (N,C) or (N,), (N,C) means
                one-hot form.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            target = self.ancestor_path_target_mask[target]
            calculate_loss_func = py_sigmoid_focal_loss

            if weight is not None:
                if weight.shape != target.shape:
                    if weight.size(0) == target.size(0):
                        # For most cases, weight is of shape (num_priors, ),
                        #  which means it does not have the second axis num_class
                        weight = weight.view(-1, 1)
                    else:
                        # Sometimes, weight per anchor per class is also needed. e.g.
                        #  in FSAF. But it may be flattened of shape
                        #  (num_priors x num_class, ), while targets is still of shape
                        #  (num_priors, num_class).
                        assert weight.numel() == target.numel()
                        weight = weight.view(target.size(0), -1)
                assert weight.ndim == target.ndim
                # weight now can be multiplied by the class level weights
                #  to get the final weight
                weight = weight * self.class_level_weight

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls
