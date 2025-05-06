import torch
import torch.nn.functional as F
from mmengine.fileio import load
from mmdet.registry import MODELS
from mmdet.models.losses.focal_loss import FocalCustomLoss
from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.models.losses.focal_loss import py_sigmoid_focal_loss

from hod.utils.tree import HierarchyTree

@MODELS.register_module()
class HierarchicalFocalLoss(FocalCustomLoss):
    def __init__(self,
                 ann_file='',
                 **ce_kwargs):
        """
        ce_kwargs are forwarded to CrossEntropyCustomLoss,
        e.g. `num_classes=len(class_to_idx)`, `ignore_index=...`, etc.
        """
        super().__init__(**ce_kwargs)
        # we'll keep a little cache of each parent-child index list
        self._siblings_cache = {}
        self.load_taxonomy(ann_file)
    
    def load_taxonomy(self, ann_file):
        ann = load(ann_file)
        cats = ann['categories']
        taxonomy = ann.get('taxonomy', {})

        # build name<->idx map
        self.class_to_idx = {c['name']: c['id'] for c in cats}
        
        if not self.use_sigmoid:
            bg_id = len(self.class_to_idx)
            self.class_to_idx['background'] = bg_id
            taxonomy['background'] = {}    # make background a child of root
            taxonomy = {'root': taxonomy}

        # build the tree
        self.tree = HierarchyTree(taxonomy)

        # NOW precompute the “siblings” groups and their level‐weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # preallocate the dt_paths table
        C = len(self.class_to_idx)
        self.dt_path_mask = torch.zeros((C+1, C),
                                   dtype=torch.bool,
                                   device=device)
        self.class_depth = torch.zeros(C, device=device)

        self.siblings_idxs = []    # List[LongTensor] each = the column‐indices of one siblings‐set
        self.level_weights = []    # List[float]    each = exp(-alpha * depth_of_parent)
        self.kids = []

        # walk all nodes; group by “the parent” (skip root itself)
        for pname, parent in self.tree.class_to_node.items():
            # 1) build paths for *this* node
            # get_path returns a list of ids from root→...→pname
            this_idx = self.class_to_idx[pname]
            path = self.tree.get_path(pname)   # e.g. [0, 3, 17]
            self.class_depth[this_idx] = len(path) - 1
            for p in path:
                j = self.class_to_idx[p]
                self.dt_path_mask[this_idx, j] = True

            # 2) build sibling‐set if this node has children
            kids = parent.children
            if not kids:
                continue            # skip leaves

            # make one LongTensor of all the kids’ global idxs
            idxs = torch.tensor(
                [self.class_to_idx[ch.name] for ch in kids],
                device=device,
                dtype=torch.long
            )
            self.siblings_idxs.append(idxs)
            self.kids.append(kids)

            # depth_of_parent = #ancestors of pname
            depth = self.tree.get_depth(pname) + 1 # depth starts at zero
            w = torch.exp(-self.alpha * depth * torch.ones((), device=device))
            self.level_weights.append(w)

        max_depth = self.class_depth.max()  # Find the maximum depth
        class_level_weight = torch.exp(-self.alpha * (max_depth - self.class_depth))  # shape (C,)
        # class_level_weight = torch.exp(-self.alpha * self.class_depth)  # shape (C,)
        # class_level_weight = torch.ones_like(self.class_depth) # shape (C,)

        depths_for_norm = torch.arange(max_depth + 1, device=device)
        norm = torch.exp(-self.alpha * depths_for_norm).sum()
        # norm = max_depth

        # Normalize weights, handling potential division by zero or very small norm
        if norm > 1e-6:
            self.class_level_weight = class_level_weight / norm
        else:
            self.class_level_weight = class_level_weight

    def forward(self, cls_score, labels, weight, **kwargs):
        """
        cls_score: (Batch * Queries, C+1) logits over all fine-grained classes + background
        labels:    (Batch * Queries) each in [0..C-1] or =ignore_index
        """
        if self.use_sigmoid:
            return self.forward_sigmoid(cls_score, labels, weight, **kwargs)
        else:
            return self.forward_softmax(cls_score, labels, weight, **kwargs)

    def forward_sigmoid(self, logits, labels, weight=None, **kwargs):
        # cls_labels is an (M,) LongTensor of the ground‐truth labels in [0..C−1]`
        # Multilabel
        targets = self.dt_path_mask[labels]     # (M, C), bool
        
        # Single label
        # targets = F.one_hot(labs, num_classes=C+1)  # (M, C+1), bool
        # targets = targets[:, :C]
        # targets = targets.float()            # convert to float

        losses = py_sigmoid_focal_loss(
            logits,
            targets.float(),
            weight=None,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction='none',
            avg_factor=None,
        )
        
        weighted_loss = losses * self.class_level_weight  # (M,C) broadcast
        if weight is not None:
            if weight.shape != weighted_loss.shape:
                if weight.size(0) == weighted_loss.size(0):
                    # For most cases, weight is of shape (num_priors, ),
                    #  which means it does not have the second axis num_class
                    weight = weight.view(-1, 1)
                else:
                    # Sometimes, weight per anchor per class is also needed. e.g.
                    #  in FSAF. But it may be flattened of shape
                    #  (num_priors x num_class, ), while loss is still of shape
                    #  (num_priors, num_class).
                    assert weight.numel() == weighted_loss.numel()
                    weight = weight.view(weighted_loss.size(0), -1)
            assert weight.ndim == weighted_loss.ndim
        loss = weight_reduce_loss(weighted_loss, weight=weight, reduction='mean', avg_factor=kwargs.get('avg_factor'))
        return self.loss_weight * loss
    
    def forward_softmax(self, cls_score, labels, weight, **kwargs):
        N, Cplus1 = cls_score.shape
        device = cls_score.device

        # drop ignored and background‐labels in one go
        if self.ignore_index is None or self.ignore_index < 0:
            valid = torch.ones_like(labels, dtype=torch.bool)
        else:
            valid = (labels != self.ignore_index)
        labs  = labels[valid]                   # (N_valid,)
        logits = cls_score[valid]               # (N_valid, Cplus1)
        logp = logits.log_softmax(dim=1).exp() # global probs

        total_loss   = torch.zeros((), device=device)
        total_w = torch.zeros((), device=device)

        # we also want a global2local map per siblings‐set,
        # but we can build that on the fly from idxs.
        for (idxs, w) in zip(self.siblings_idxs, self.level_weights):
            # gather only the relevant columns (the “siblings”)
            local_logits = logp[:, idxs]       # (N_valid, S)

            # make a global->local index map for this set
            g2l = -torch.ones(Cplus1, device=device, dtype=torch.long)
            g2l[idxs] = torch.arange(len(idxs), device=device)

            # map the lab indices
            target = g2l[labs]                   # (N_valid,) each in [0..S-1]

            # After you build g2l and target = g2l[labs]:
            in_group = target >= 0           # a boolean mask of shape (N_valid,)
            in_group = target >= 0
            if in_group.any():
                # real group
                lg = local_logits[in_group]
                tg = target[in_group]
            else:
                continue
                # dummy pass so DDP sees these params in the graph
                # lg = local_logits[:1]             # pick any row
                # tg = torch.zeros(1, dtype=torch.long, device=device)
                # w = 0  # zero it out

            loss_l = F.cross_entropy(lg, tg, reduction=self.reduction)
            total_loss += w * loss_l
            total_w    += w

        total_loss = total_loss / total_w

        return self.loss_weight * total_loss
