from typing import Dict, List, Set

import torch
import torch.nn as nn

from mmengine.fileio import load
from mmdet.registry import MODELS

from hod.utils import HierarchyTree

@MODELS.register_module()
class EntailmentConeLoss(nn.Module):
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
    num_negative_samples_per_positive : int
        How many negatives to draw per positive pair.
    """
    def __init__(
        self,
        ann_file: str,
        curvature: float = 0.0,
        beta: float = 0.1,
        loss_weight: float = 1.0,
        margin: float = 0.1,
        num_negative_samples_per_positive: int = 5,
    ):
        super().__init__()

        # ---------------- static parameters ----------------------------------
        self.curvature = curvature
        self.beta = beta
        self.loss_weight = loss_weight
        self.margin = margin
        self.num_neg = num_negative_samples_per_positive

        # ---------------- taxonomy -------------------------------------------
        self._load_taxonomy(ann_file)
        self._build_pairs_and_neg_candidates()

    def _load_taxonomy(self, ann_file: str):
        ann = load(ann_file)
        self.tree = HierarchyTree(ann.get('taxonomy', {}))
        self.class2idx = {c["name"]: c["id"] for c in ann["categories"]}
        self.idx2class = {v: k for k, v in self.class2idx.items()}

    def _build_pairs_and_neg_candidates(self):
        """Pre-compute
        * self._pairs              - all (parent, child) LongTensor
        """
        pairs: List[List[int]] = []
        neg_candidates: Dict[int, torch.Tensor] = {}

        all_ids = torch.tensor(list(self.idx2class.keys()))  # [C]

        root_name = self.tree.root.name
        for p_name, p_node in self.tree.class_to_node.items():
            if p_name == root_name:              # ← skip the root
                continue
            if p_name not in self.class2idx:
                continue
            p_idx = self.class2idx[p_name]

            # ----- direct children -----
            children = [
                self.class2idx[c.name]
                for c in p_node.children
                if c.name in self.class2idx
            ]
            for c in children:
                pairs.append([p_idx, c])

            # ----- negatives: anything that is *not* p or its descendants ---
            desc = {
                self.class2idx[d]
                for d in self.tree.get_descendants(p_name)
                if d in self.class2idx
            }
            desc.add(p_idx)
            mask = torch.ones_like(all_ids, dtype=torch.bool)
            mask[list(desc)] = False
            neg_candidates[p_idx] = all_ids[mask]

        # register buffers
        self.register_buffer("_pairs",
                     torch.tensor(pairs, dtype=torch.long), persistent=False)
        buf = [torch.as_tensor(neg_candidates.get(i, []), dtype=torch.long)
            for i in range(len(self.idx2class))]
        self.register_buffer("_neg_cand_flat", torch.cat(buf))      # 1-D
        self.register_buffer("_neg_ptrs",
                            torch.tensor([0] + [t.numel() for t in buf]).cumsum(0))

    def forward(self, prototypes: torch.Tensor) -> torch.Tensor:
        if self._pairs.numel() == 0:  # degenerate tree
            return prototypes.new_tensor(0.0)

        num_cls = prototypes.size(0)

        # ---------------- positive energy -----------------------------------
        valid_pos = (self._pairs < num_cls).all(dim=1)
        if not valid_pos.any():
            return prototypes.new_tensor(0.0)

        pos = self._pairs[valid_pos]  # [N+, 2]
        p_pos, c_pos = prototypes[pos[:, 0]], prototypes[pos[:, 1]]
        pos_energy = self._calculate_energy(p_pos, c_pos).mean()

        # ---------------- negative energy -----------------------------------
        if self.num_neg == 0 or num_cls == 1:
            neg_energy = prototypes.new_tensor(0.0)
        else:
            # pick unique parents → one sampling per parent
            p_idx_unique = torch.unique(pos[:, 0])
            neg_pairs = []  # will be cat'd later

            # sizes for each parent
            sizes = self._neg_ptrs[p_idx_unique + 1] - self._neg_ptrs[p_idx_unique]
            max_s = sizes.max()

            # build a padded table [n_parent, max_s]
            offsets = self._neg_ptrs[p_idx_unique]
            table = torch.full((p_idx_unique.size(0), max_s),
                            -1, dtype=torch.long, device=prototypes.device)
            for row, (o, s) in enumerate(zip(offsets, sizes)):
                table[row, :s] = self._neg_cand_flat[o:o+s]

            # sample indices row-wise without replacement
            n_parent = p_idx_unique.size(0)
            rand  = torch.rand(n_parent, max_s, device=prototypes.device)      # (P, W)
            order = rand.argsort(dim=1)[:, : self.num_neg]                     # (P, k)
            # pick the columns row-wise
            sel = table.gather(1, order)                                       # (P, k)

            parents = p_idx_unique.repeat_interleave(self.num_neg)
            neg_pairs = torch.stack([parents, sel.flatten()], 1)
            neg_pairs = neg_pairs[neg_pairs[:, 1] >= 0]         # drop pads

            if neg_pairs.numel():
                p_neg, c_neg = (
                    prototypes[neg_pairs[:, 0]],
                    prototypes[neg_pairs[:, 1]],
                )
                neg_e = self._calculate_energy(p_neg, c_neg)
                neg_energy = torch.relu(self.margin - neg_e).mean()
            else:
                neg_energy = prototypes.new_tensor(0.0)

        return 0.5 * (pos_energy + neg_energy) * self.loss_weight

    def _calculate_energy(self,
                        parent: torch.Tensor,
                        child: torch.Tensor,
                        eps: float = 1e-6) -> torch.Tensor:
        """
        Vectorised cone energy in one pass.

        Shapes
        ------
        parent, child : [..., d]

        Returns
        -------
        energy : [... ]   # same leading dims, scalar per pair
        """
        p_norm = torch.norm(parent, dim=-1)

        # ---------- apex angle ----------------------------------------------
        diff      = child - parent
        diff_norm = torch.norm(diff, dim=-1)

        if self.curvature == 0.0:                        # ─ Euclidean ─
            num        = (
                torch.square(child).sum(-1)
                - torch.square(parent).sum(-1)
                - torch.square(diff).sum(-1)
            )
            denom = 2 * p_norm * diff_norm

        elif self.curvature == -1.0:                     # ─ Poincaré ─
            c_norm = torch.norm(child, dim=-1)
            dot    = (parent * child).sum(-1)
            num    = dot * (1 + p_norm**2) - p_norm**2 * (1 + c_norm**2)

            omega  = p_norm * diff_norm
            den_sq = 1 + p_norm**2 * c_norm**2 - 2 * dot
            denom  = omega * torch.sqrt(torch.clamp(den_sq, min=0.0))
        else:
            raise NotImplementedError("curvature must be 0 or -1")

        cosang = torch.clamp(num / (denom + eps), -1 + eps, 1 - eps)
        angle  = torch.acos(cosang)

        # ---------- aperture Ψ(p) -------------------------------------------
        ap = self.beta / (p_norm + eps)
        if self.curvature == -1.0:                       # Poincaré factor
            ap = ap * (1 - p_norm**2)
        aperture = torch.asin(torch.clamp(ap, 0.0, 1.0 - eps))

        return torch.relu(angle - aperture)
