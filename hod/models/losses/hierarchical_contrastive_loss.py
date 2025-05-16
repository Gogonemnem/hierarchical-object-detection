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

    def forward(self, distance_matrix: torch.Tensor):
        # Supervised contrastive loss
        # IN version
        hierarchical_mask = self.ancestor_path_target_mask[:-1]
        sim = torch.exp(distance_matrix * self.class_level_weight)
        pos = sim*hierarchical_mask
        pos = pos[self.leaf_idx].sum(dim=1)
        neg = sim*~hierarchical_mask
        neg = neg[self.leaf_idx].sum(dim=1)
        loss = -torch.log(pos / neg)

        # OUT version: does not make sense because the loss is on prototypes
        # thus the distance matrix to itself is always 0
        # n_pairs = distance_matrix.shape[0]
        # sim = torch.exp(distance_matrix)
        # mask = torch.eye(n_pairs, device=sim.device).bool()
        # pos = sim.masked_select(mask)
        # neg = sim.masked_select(~mask).view(n_pairs, -1).sum(dim=-1)
        # loss = -torch.log(pos / neg) * self.class_level_weight
        return loss.mean() * self.loss_weight
