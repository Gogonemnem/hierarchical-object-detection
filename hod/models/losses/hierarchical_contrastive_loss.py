import torch

from mmdet.registry import MODELS
from hod.models.losses.hierarchical_focal_loss import HierarchicalFocalLoss

@MODELS.register_module()
class HierarchicalContrastiveLoss(HierarchicalFocalLoss):
    def load_taxonomy(self, ann_file):
        super().load_taxonomy(ann_file)
        leafs = self.tree.get_leaf_nodes()
        leaf_idx = [self.class_to_idx[leaf.name] for leaf in leafs]
        self.leaf_idx = torch.tensor(leaf_idx, device=self.class_level_weight.device)

        hierarchical_mask = self.ancestor_path_target_mask[:-1].clone()

        # set diagonal to 0, distance to itself is not considered
        hierarchical_mask.fill_diagonal_(0)
        anchor_indices, pair_indices = hierarchical_mask.nonzero(as_tuple=True)

        self.anchor_indices = anchor_indices
        self.pair_indices = pair_indices

        self.weights_for_pairs = self.class_level_weight[self.anchor_indices]
        self.hierarchical_mask = hierarchical_mask

    def forward(self, distance_matrix: torch.Tensor):
        epsilon = 1e-9
        # Supervised contrastive loss
        sim = torch.exp(distance_matrix)
        pairs = sim[self.anchor_indices, self.pair_indices]
        sum_sim_from_anchor_to_all = sim.sum(dim=1)
        denominators_for_pairs = sum_sim_from_anchor_to_all[self.anchor_indices] # Shape: (num_positive_pairs,)

        log_probs = torch.log(pairs + epsilon) - torch.log(denominators_for_pairs + epsilon)
        loss_per_pair = -log_probs # Shape: (num_positive_pairs,)

        # loss_per_pair = loss_per_pair * self.weights_for_pairs

        device = loss_per_pair.device

        sum_losses_per_anchor_class = loss_per_pair.sum(dim=-1)
        count_pairs_per_anchor_class = self.hierarchical_mask.sum(dim=-1)
        valid_anchor_mask = count_pairs_per_anchor_class > 0

        # If no anchors contributed at all (e.g., if self.anchor_indices was empty, though caught earlier)
        if not valid_anchor_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True) * self.loss_weight

        loss = \
            sum_losses_per_anchor_class[valid_anchor_mask] / count_pairs_per_anchor_class[valid_anchor_mask].type_as(sum_losses_per_anchor_class)

        return loss.mean() * self.loss_weight
