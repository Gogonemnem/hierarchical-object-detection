import torch
from torch import nn
from typing import Dict, List

from mmdet.registry import MODELS

from hod.models.losses.hierarchical_loss import HierarchicalDataMixin

@MODELS.register_module()
class EntailmentConeLoss(nn.Module, HierarchicalDataMixin):
    r"""Parent-child containment loss à la Ganea & Dhall.

    Args
    ----
    ann_file : str
        JSON annotation file that contains ``taxonomy`` and ``categories``.
    curvature : {0.0, -1.0}
        0 → Euclidean cone, -1 → Poincaré ball cone.
    beta : float
        Scale for the cone aperture.
    loss_weight : float
        Global multiplier.
    margin : float
        Margin for the max-margin term on negative pairs.
    """
    def __init__(
        self,
        ann_file: str,
        loss_weight: float = 1.0,
        curvature: float = 0.0,
        beta: float = 0.1,
        margin: float = 0.1,
        **kwargs
    ):
        nn.Module.__init__(self)
        HierarchicalDataMixin.__init__(self, ann_file=ann_file)

        # ---------------- static parameters ----------------------------------
        self.loss_weight = loss_weight
        self.curvature = curvature
        self.beta = beta
        self.margin = margin

        # ---------------- taxonomy -------------------------------------------
        self._build_pairs_and_neg_candidates()

    def _build_pairs_and_neg_candidates(self):
        """Pre-compute
        * self._pairs              - all (parent, child) LongTensor
        """
        num_cls = len(self.idx_to_class)
        pos_mask = torch.zeros((num_cls, num_cls), dtype=torch.bool)
        neg_mask = torch.ones((num_cls, num_cls), dtype=torch.bool)
        neg_mask.fill_diagonal_(False)  # an object cannot be its own negative

        root_name = self.tree.root.name
        for p_name, p_node in self.tree.class_to_node.items():
            if p_name == root_name or p_name not in self.class_to_idx:
                continue
            p_idx = self.class_to_idx[p_name]

            desc_idx = [
                self.class_to_idx[d]
                for d in
                p_node.descendants()  # all descendants including itself
            ]

            # descendants and itself are not a negatives
            neg_mask[p_idx, desc_idx] = False  
            
            # remove itself from descendants
            desc_idx.remove(p_idx)
            pos_mask[p_idx, desc_idx] = True

        # register buffers
        self.register_buffer("_pos_mask", pos_mask, persistent=False)
        self.register_buffer("_neg_mask", neg_mask, persistent=False)

    def forward(self, prototypes: torch.Tensor, eps=1e-6) -> torch.Tensor:\
        # check if there are any positive pairs to compute loss on
        if not self._pos_mask.any():
            return prototypes.new_tensor(0.0)

        # parent = prototypes[:, None]  # [N, 1, d]
        # child = prototypes[None, :]   # [1, N, d]

        # Norms
        proto_norm = prototypes.norm(dim=-1)
        proto_norm_sq = proto_norm**2

        # ---------- aperture Ψ(p) -------------------------------------------
        ap = self.beta / (proto_norm + eps)
        if self.curvature == -1.0:  # Poincaré factor
            ap = ap * (1 - proto_norm**2)
        # Aperture is a property of the parent cone (row index), so we shape
        # it to [N, 1] for correct broadcasting against the [N, N] angle matrix.
        aperture = torch.asin(torch.clamp(ap, 0.0, 1.0 - eps))[:, None]

        # ---------- apex angle ----------------------------------------------
        dot = prototypes @ prototypes.T
        diff_norm_sq = proto_norm_sq[:, None] + proto_norm_sq[None, :] - 2 * dot
        diff_norm = torch.sqrt(torch.relu(diff_norm_sq) + eps)

        if self.curvature == 0.0:  # ─ Euclidean ─
            num = (
                proto_norm_sq[None, :] - proto_norm_sq[:, None]
                - torch.square(diff_norm)
            )
            denom = 2 * proto_norm[:, None] * diff_norm

        elif self.curvature == -1.0:  # ─ Poincaré ─
            num = dot * (1 + proto_norm_sq[:, None]) - proto_norm_sq[:, None] * (1 + proto_norm_sq[None, :])

            omega = proto_norm[:, None] * diff_norm
            den_sq = 1 + proto_norm_sq[:, None] * proto_norm_sq[None, :] - 2 * dot
            denom = omega * torch.sqrt(torch.clamp(den_sq, min=0.0))
        else:
            raise NotImplementedError("curvature must be 0 or -1")

        cosang = torch.clamp(num / (denom + eps), -1 + eps, 1 - eps)
        angle = torch.acos(cosang)

        energy = torch.relu(angle - aperture)
        pos_energy = energy[self._pos_mask].mean()
        neg_energy = torch.relu(self.margin - energy[self._neg_mask]).mean()
        return self.loss_weight * 0.5*(pos_energy + neg_energy)
