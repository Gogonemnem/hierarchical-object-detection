import torch
import torch.nn as nn

from mmengine.fileio import load
from mmdet.registry import MODELS

from hod.utils import HierarchyTree

@MODELS.register_module()
class EntailmentConeLoss(nn.Module):
    """Parent-child containment loss à la Ganea & Dhall.

    Args:
        pairs (list[tuple[int,int]]): list of (parent_idx, child_idx)
        beta (float): scale for the margin at the cone border.
        euclidean (bool): if False, we treat the embeddings as Poincaré-ball.
    """
    def __init__(self,
                 ann_file='',
                 curvature=0.0,
                 beta=0.1,
                 loss_weight=1.0,
                 margin=0.1,  # For max-margin loss
                 num_negative_samples_per_positive=1 # Num negatives per positive
                 ):
        super().__init__()

        self.curvature = curvature
        self.beta = beta
        self.loss_weight = loss_weight
        self.margin = margin
        self.num_negative_samples = num_negative_samples_per_positive
        self.load_taxonomy(ann_file=ann_file)
        self._build_parent_children_index_map()

        # build the <parent,child> index list from your hierarchy
        pairs = []
        for p_idx, c_list in self._parent_idx_to_children_indices.items():
            for c_idx in c_list:
                pairs.append((p_idx, c_idx))
        self.register_buffer(
            '_pairs', torch.tensor(pairs, dtype=torch.long))

        # This can be useful if not all prototype indices are actual classes you want to sample from
        # For now, we'll primarily use arange(num_total_classes) for sampling pool
        self._all_learnable_class_indices = []
        if self.class_to_idx:
             self._all_learnable_class_indices = list(self.class_to_idx.values())

    def load_taxonomy(self, ann_file):
        ann = load(ann_file)
        taxonomy = ann.get('taxonomy', {})
        self.tree = HierarchyTree(taxonomy)
        self.class_to_idx = {c['name']: c['id'] for c in ann['categories']}
        self.idx_to_class = {c['id']: c['name'] for c in ann['categories']}

    def _build_parent_children_index_map(self):
        # Assumes self.tree and self.class_to_idx are populated
        self._parent_idx_to_children_indices = {}
        if not self.tree or not self.class_to_idx: # Guard
            return

        for p_name, p_node in self.tree.class_to_node.items():
            if p_name not in self.class_to_idx:
                continue
            p_idx = self.class_to_idx[p_name]

            child_indices = []
            for child_node in p_node.children:
                if child_node.name in self.class_to_idx:
                    child_indices.append(self.class_to_idx[child_node.name])

            if child_indices: # Only add if it has mappable children
                self._parent_idx_to_children_indices[p_idx] = child_indices

    def _get_descendant_indices(self, parent_idx: int) -> set:
        if not self.tree or parent_idx not in self.idx_to_class:
            return set()

        parent_name = self.idx_to_class[parent_idx]
        descendant_names = self.tree.get_descendants(parent_name)

        descendant_indices = set()
        for name in descendant_names:
            if name in self.class_to_idx:
                descendant_indices.add(self.class_to_idx[name])
        return descendant_indices

    def forward(self, prototypes: torch.Tensor) -> torch.Tensor:
        if self._pairs.numel() == 0:
            return torch.tensor(0.0, device=prototypes.device, dtype=prototypes.dtype)

        num_total_classes = prototypes.shape[0]
        all_class_indices_tensor = torch.arange(num_total_classes, device=prototypes.device)

        # --- 1. Positive Pair Loss (Vectorized) ---
        # Get all parent and child indices from pre-computed pairs
        pos_p_indices_all = self._pairs[:, 0]
        pos_c_indices_all = self._pairs[:, 1]

        # Filter out pairs with out-of-bounds indices for the current prototypes
        valid_pos_mask = (pos_p_indices_all < num_total_classes) & (pos_c_indices_all < num_total_classes)
        if not valid_pos_mask.any(): # No valid positive pairs for these prototypes
            return torch.tensor(0.0, device=prototypes.device, dtype=prototypes.dtype)

        # Use only valid indices
        pos_p_indices = pos_p_indices_all[valid_pos_mask]
        pos_c_indices = pos_c_indices_all[valid_pos_mask]
        
        num_valid_positive_pairs = pos_p_indices.shape[0]
        if num_valid_positive_pairs == 0: # Should be caught by valid_pos_mask.any() but good for safety
             return torch.tensor(0.0, device=prototypes.device, dtype=prototypes.dtype)

        p_embeds_pos = prototypes[pos_p_indices]  # [num_valid_positive_pairs, d]
        c_embeds_pos = prototypes[pos_c_indices]  # [num_valid_positive_pairs, d]
        
        # Calculate average energy for valid positive pairs
        avg_positive_loss = self._calculate_energy(p_embeds_pos, c_embeds_pos).mean()


        # --- 2. Negative Pair Sampling and Index Collection ---
        avg_negative_loss = torch.tensor(0.0, device=prototypes.device, dtype=prototypes.dtype)
        
        if self.num_negative_samples > 0 and num_total_classes > 1: # Need at least 2 classes to sample a different one
            collected_neg_p_indices = [] # List to store parent indices for all negative pairs
            collected_neg_c_indices = [] # List to store sampled negative child indices

            # Loop through each valid positive pair to define the context for negative sampling
            for i in range(num_valid_positive_pairs):
                # Parent and true child for the current positive pair
                current_p_idx_val = pos_p_indices[i].item() # scalar Python int for set operations
                current_c_idx_val = pos_c_indices[i].item() # scalar Python int

                descendant_indices_set = self._get_descendant_indices(current_p_idx_val)
                # Non-candidates: parent itself, its true child from this positive pair, and all its other descendants
                non_candidate_indices = descendant_indices_set.union({current_p_idx_val, current_c_idx_val})
                
                # Create a boolean mask for valid negative candidates from all_class_indices_tensor
                valid_neg_selection_mask = torch.ones(num_total_classes, dtype=torch.bool, device=prototypes.device)
                
                if non_candidate_indices: # Ensure set is not empty
                    # Filter non_candidate_list to only include valid indices within num_total_classes
                    valid_non_candidates = [idx for idx in list(non_candidate_indices) if 0 <= idx < num_total_classes]
                    if valid_non_candidates: # If there are valid indices to exclude
                        valid_neg_selection_mask[valid_non_candidates] = False
                
                potential_neg_indices_for_p = all_class_indices_tensor[valid_neg_selection_mask]

                if potential_neg_indices_for_p.numel() > 0:
                    num_to_sample = min(self.num_negative_samples, potential_neg_indices_for_p.numel())
                    
                    perm = torch.randperm(potential_neg_indices_for_p.numel(), device=prototypes.device)
                    sampled_neg_indices = potential_neg_indices_for_p[perm[:num_to_sample]] # Tensor of indices
                    
                    # Store the original parent tensor index (pos_p_indices[i]) repeated num_to_sample times
                    collected_neg_p_indices.extend([pos_p_indices[i]] * num_to_sample)
                    collected_neg_c_indices.extend(sampled_neg_indices) # sampled_neg_indices is already a list of tensors or a tensor

            # --- 3. Negative Pair Loss (Batched, using collected indices) ---
            if collected_neg_p_indices: # If any negative samples were collected
                # Stack the collected lists of tensors into single tensors
                neg_p_indices_tensor = torch.stack(collected_neg_p_indices)
                neg_c_indices_tensor = torch.stack(collected_neg_c_indices)

                p_embeds_neg = prototypes[neg_p_indices_tensor]
                c_embeds_neg = prototypes[neg_c_indices_tensor]

                negative_energies = self._calculate_energy(p_embeds_neg, c_embeds_neg)
                margin_violations_neg = torch.relu(self.margin - negative_energies)
                
                avg_negative_loss = margin_violations_neg.mean()

        # --- 4. Combine Losses ---
        final_loss = (avg_positive_loss + avg_negative_loss) / 2
        return final_loss * self.loss_weight


    def _calculate_energy(self, parent_embed: torch.Tensor, child_embed: torch.Tensor):
        # p_embed: [N, d]
        # c_embed: [N, d]
        angle = self.apex_angle(parent_embed, child_embed)
        aperture = self.aperture(parent_embed)
        energy = torch.relu(angle - aperture)
        return energy

    def apex_angle(self, parent: torch.Tensor, child: torch.Tensor, eps: float = 1e-6):
        """Cone angle Ξ(p,c).  Shapes: [...,d]."""
        p_norm = parent.norm(dim=-1)

        if self.curvature == 0.0:
            diff = child - parent
            num = (child.pow(2).sum(-1) - parent.pow(2).sum(-1) - diff.pow(2).sum(-1))
            denom = 2 * p_norm * diff.norm(dim=-1)

        elif self.curvature == -1.0:
            c_norm = child.norm(dim=-1)
            dot = (parent * child).sum(-1)
            num = dot * (1 + p_norm**2) - p_norm**2 * (1 + c_norm**2)

            omega = p_norm * torch.norm(parent - child, dim=-1)
            den_sq_arg = 1 + p_norm**2 * c_norm**2 - 2 * dot
            denom = omega * (den_sq_arg.clamp(min=0)).sqrt()

        else:
            raise NotImplementedError("Curvature must be 0 or -1")

        cosang = torch.clamp(num / (denom+eps), -1 + eps, 1 - eps)
        return torch.acos(cosang)

    def aperture(self, parent: torch.Tensor, eps: float = 1e-6):
        """Cone aperture Ψ(p).  Shapes: [...,d]."""
        p_norm = parent.norm(dim=-1)

        aperture = self.beta / (p_norm + eps)
        if self.curvature == -1.0:
            aperture *= (1 - p_norm**2)
        return torch.asin(torch.clamp(aperture, 0, 1 - eps))
