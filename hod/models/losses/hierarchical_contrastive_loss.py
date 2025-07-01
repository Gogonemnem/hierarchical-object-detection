import torch

from mmdet.registry import MODELS
from hod.models.losses.hierarchical_focal_loss import HierarchicalFocalLoss

@MODELS.register_module()
class HierarchicalContrastiveLoss(HierarchicalFocalLoss):
    def __init__(self,
                 aggregate_per='depth',
                 **kwargs):
        self.aggregate_per = aggregate_per
        super().__init__(**kwargs)

    def load_taxonomy(self, ann_file):
        super().load_taxonomy(ann_file)
        leafs = self.tree.get_leaf_nodes()
        leaf_idx = [self.class_to_idx[leaf.name] for leaf in leafs]
        self.leaf_idx = torch.tensor(leaf_idx, device=self.class_level_weight.device)

        ancestor_mask = self.ancestor_path_target_mask[:-1].clone()

        # set diagonal to 0, distance to itself is not considered
        ancestor_mask.fill_diagonal_(0)

        if self.aggregate_per is None:
            self.hierarchical_mask = ancestor_mask.unsqueeze(0) # Shape: (1, C_i, C_j)
            self.class_level_weight = torch.ones(1, device=self.class_level_weight.device)

        elif self.aggregate_per in ['node', 'depth']:
            expanded_for_i = ancestor_mask.T.unsqueeze(2) # Shape: (C_k, C_i, 1)
            expanded_for_j = ancestor_mask.T.unsqueeze(1) # Shape: (C_k, 1, C_j)
            # self.hierarchical_mask[k, i, j] is True if (k is ancestor of i) AND (k is ancestor of j)
            self.hierarchical_mask = expanded_for_i & expanded_for_j # Shape: (C_k, C_i, C_j)

        if self.aggregate_per == 'node':
            self.class_level_weight = self.class_level_weight / self.class_level_weight.sum()

        if self.aggregate_per == 'depth':
            depth_lca = self.hierarchical_mask.sum(dim=0, keepdim=True) # Shape: (1, C_i, C_j)
            max_depth = depth_lca.max().item()
            depths = torch.arange(max_depth, device=self.class_level_weight.device)

            self.hierarchical_mask = (depth_lca > depths[:, None, None]) # Shape: (C_k, C_i, C_j)
            if self.decay == 1:
                self.class_level_weight = torch.ones_like(depths, device=self.class_level_weight.device) * 1 / max_depth # Shape: (C_k, 1)
            else:
                self.class_level_weight = (1 - self.decay) * self.decay ** depths / (1 - self.decay ** max_depth) # Shape: (C_k, 1)

    def forward(self, distance_matrix: torch.Tensor):
        epsilon = 1e-9
        # Supervised contrastive loss
        sim = torch.exp(distance_matrix)
        neg = (sim.sum(dim=1, keepdim=True) - sim)

        log_probs = -(torch.log(sim + epsilon) - torch.log(neg + epsilon))
        log_probs = log_probs.unsqueeze(0) * self.hierarchical_mask # Shape: (C_k, C_i, C_j)

        sum_loss_per_anchor_class = log_probs.sum(dim=-1)
        weighted_sum_loss_per_anchor_class = sum_loss_per_anchor_class * self.class_level_weight.unsqueeze(1)
        count_pairs_per_anchor_class = self.hierarchical_mask.sum(dim=-1)

        valid_anchor_mask = count_pairs_per_anchor_class > 0
        loss = weighted_sum_loss_per_anchor_class[valid_anchor_mask] / (count_pairs_per_anchor_class[valid_anchor_mask]).type_as(sum_loss_per_anchor_class)

        return loss.mean() * self.loss_weight
