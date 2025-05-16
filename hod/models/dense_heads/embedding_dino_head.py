import copy

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from mmengine.fileio import load
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.models.dense_heads import DINOHead
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.utils import OptConfigType

from hod.models.layers import EmbeddingClassifier
from hod.utils import HierarchyTree

@MODELS.register_module()
class EmbeddingDINOHead(DINOHead):
    def __init__(self,
                 ann_file='',
                 cls_curvature=0.0,
                 share_cls_layer=True,
                 loss_embed: OptConfigType=None,
                 **kwargs):
        """
        Args:
            ann_file (str): Path to the annotation file containing the
                taxonomy. The file should be in COCO format.
            cls_curvature (float): Curvature parameter for the embedding space.
            share_cls_layer (bool): Whether to share the classification
                layer across all prediction layers.
            loss_embed (dict, optional): Configuration for the
                embedding loss.
                Defaults to None (disabled).
                Example config:
                loss_embed=dict(
                    type='EntailmentConeLoss',
                    cone_beta=0.1,
                    init_norm_upper_offset=0.5,
                    loss_weight=1.0
                    num_negative_samples_per_positive=1,
                    margin=0.1,
                or:
                loss_embed=dict(
                    type='HierarchicalContrastiveLoss',
                    loss_weight=1.0,
                    decay=1.0,
                )
        """
        self.curvature = cls_curvature
        self.share_cls_layer = share_cls_layer
        use_embed_loss = loss_embed and isinstance(loss_embed, dict)
        self.use_cone = use_embed_loss and loss_embed.get('type', None) == "EntailmentConeLoss" and loss_embed.get('loss_weight', 0.0) > 0
        self.use_contrastive = use_embed_loss and loss_embed.get('type', None) == "HierarchicalContrastiveLoss" and loss_embed.get('loss_weight', 0.0) > 0
        self.tree = None
        self.load_taxonomy(ann_file)
        self._build_parent_children_index_map()
            
        self.beta = 0.0
        self.init_norm_upper_offset = 0.0
        if self.use_cone:
            self.beta = loss_embed['beta']
            self.init_norm_upper_offset = loss_embed['init_norm_upper_offset']
        super().__init__(**kwargs)

        if self.use_cone:
            loss_embed['curvature'] = cls_curvature
        
        if use_embed_loss:
            loss_embed['ann_file'] = ann_file
            self.loss_embed = MODELS.build(loss_embed)

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

    def _init_layers(self, *args) -> None:
        """Initialize classification branch of head."""
        super()._init_layers(*args)
        full_taxonomy = len(self.tree) == self.cls_out_channels
        clip_exempt = None
        if full_taxonomy:
            root_idx  = self.class_to_idx[self.tree.root.name]
            clip_exempt = [root_idx]

        fc_cls = EmbeddingClassifier(self.embed_dims,
                                     self.cls_out_channels,
                                     curvature=self.curvature,
                                     use_cone=self.use_cone,
                                     cone_beta=self.beta,
                                     init_norm_upper_offset=self.init_norm_upper_offset,
                                     clip_exempt_indices=clip_exempt,)

        if self.share_cls_layer:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])

    def loss(self, **kwargs):
        # normal DINO losses
        loss_dict = super().loss(**kwargs)
        if self.use_cone:
            # === entailment‑cone loss (no image data needed) ===========
            if self.share_cls_layer:
                # All classification branches are identical when shared.
                # Use the first branch's prototypes for loss calculation.
                prototypes = self.cls_branches[0].prototypes
                base_entail_loss = self.loss_embed(prototypes)
                # Scale the single loss by the number of prediction layers.
                loss_dict['loss_entail'] = base_entail_loss * self.num_pred_layer
            else:
                # Each classification branch has its own distinct prototypes.
                # Calculate and store loss for each layer.
                for i, cls_branch in enumerate(self.cls_branches):
                    prototypes = cls_branch.prototypes
                    entail_loss = self.loss_embed(prototypes)
                    # The last layer's loss is typically named 'loss_entail',
                    # while intermediate layers are prefixed (e.g., 'd0.loss_entail').
                    loss_key = 'loss_entail' if i == self.num_pred_layer - 1 else f'd{i}.loss_entail'
                    loss_dict[loss_key] = entail_loss

        if self.use_contrastive:
            # === contrastive loss (no image data needed) ===========
            if self.share_cls_layer:
                # All classification branches are identical when shared.
                # Use the first branch's prototypes for loss calculation.
                prototypes = self.cls_branches[0].prototypes
                distance_matrix = self.cls_branches[0].get_distance_logits(prototypes.unsqueeze(0), prototypes)
                contrastive_loss = self.loss_embed(distance_matrix.squeeze(0))
                # Scale the single loss by the number of prediction layers.
                loss_dict['loss_contrastive'] = contrastive_loss * self.num_pred_layer
            else:
                # Each classification branch has its own distinct prototypes.
                # Calculate and store loss for each layer.
                for i, cls_branch in enumerate(self.cls_branches):
                    prototypes = cls_branch.prototypes
                    distance_matrix = cls_branch.get_distance_logits(prototypes.unsqueeze(0), prototypes)
                    contrastive_loss = self.loss_embed(distance_matrix.squeeze(0))
                    # The last layer's loss is typically named 'loss_contrastive',
                    # while intermediate layers are prefixed (e.g., 'd0.loss_contrastive').
                    loss_key = 'loss_contrastive' if i == self.num_pred_layer - 1 else f'd{i}.loss_contrastive'
                    loss_dict[loss_key] = contrastive_loss

        return loss_dict

    def _get_empty_results(self, device):
        results = InstanceData()
        results.bboxes = torch.empty((0, 4), device=device)
        results.scores = torch.empty((0,), device=device)
        results.labels = torch.empty((0,), dtype=torch.long, device=device)
        return results

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 4].
            img_meta (dict): Image meta info.
            rescale (bool): If True, return boxes in original image
                space. Default True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        img_shape = img_meta['img_shape']
        # exclude background
        if self.loss_cls.use_sigmoid:
            # Check if hierarchical refinement is possible and enabled
            use_hierarchical_refinement = (
                self.tree is not None and 
                self.class_to_idx and # Ensure mappings are present
                hasattr(self, '_parent_idx_to_children_indices') and
                self._parent_idx_to_children_indices # Ensure map is built and not empty
            )
            use_hierarchical_refinement = False # Disable for now
            if use_hierarchical_refinement:
                old_indexes = torch.empty((0,), device=cls_score.device)
                cls_prob = cls_score.sigmoid()  # [num_queries, num_classes]

                # Create a working copy of cls_prob for all queries. We'll modify only the selected ones.
                refined_cls_prob = cls_prob.clone()

                # --- Phase 1: Determine Dynamic Activation Threshold ---
                scores, indexes = refined_cls_prob.view(-1).topk(max_per_img)
                activation_threshold = scores[30] # The 30th score in the top-k is the threshold
                og_indexes = indexes.clone()
                # --- Phase 2: Hierarchical Refinement ---
                while not torch.equal(old_indexes, indexes):
                    old_indexes = indexes.clone()
                    for _, idx in zip(scores, old_indexes):
                        det_labels = idx % self.num_classes
                        bbox_index = idx // self.num_classes

                        for p_idx in range(self.cls_out_channels): # cls_out_channels is num_classes here
                            if self.has_activated_descendant(p_idx, refined_cls_prob[bbox_index], activation_threshold):
                                # Suppress the parent class if any child is activated
                                refined_cls_prob[bbox_index, p_idx] = -1

                    # After processing all top-k scores, we can now find the new top-k
                    scores, indexes = refined_cls_prob.view(-1).topk(max_per_img)
                    activation_threshold = scores[30]

                # --- Phase 3: Final Selection ---
                det_labels = indexes % self.num_classes
                bbox_index = indexes // self.num_classes
                bbox_pred = bbox_pred[bbox_index]

            else: # Fallback to standard DINOHead sigmoid logic
                # cls_score shape: [num_queries, num_classes]
                cls_prob = cls_score.sigmoid()
                # Find the max score and class index per query
                # scores_per_query shape: [num_queries]
                # labels_per_query shape: [num_queries]
                scores_per_query, labels_per_query = cls_prob.max(dim=-1)
                # Select top-k queries based on their max score
                # scores shape: [max_per_img]
                # bbox_index shape: [max_per_img]
                num_to_select = min(max_per_img, scores_per_query.size(0))
                if num_to_select == 0: return self._get_empty_results(cls_score.device)

                scores, bbox_index = scores_per_query.topk(num_to_select)
                # Get the labels and bbox predictions for the selected queries
                det_labels = labels_per_query[bbox_index]
                bbox_pred = bbox_pred[bbox_index]
        else:
            # cls_score shape: [num_queries, num_classes + 1]
            # Find the max score excluding the background class (last channel)
            # scores shape: [num_queries]
            # det_labels shape: [num_queries]
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            # Select top-k queries based on their max score
            # scores shape: [max_per_img]
            # bbox_index shape: [max_per_img]
            scores, bbox_index = scores.topk(max_per_img)
            # Get the bbox predictions and labels for the selected queries
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        return results

    def has_activated_descendant(self, p_idx, cls_probs, threshold):
        # Check if p_idx has children and if any are activated
        children_indices_of_p = self._parent_idx_to_children_indices.get(p_idx, [])
        if not children_indices_of_p: # Not a parent with known children in class_to_idx
            return False

        for c_idx in children_indices_of_p:
            if cls_probs[c_idx] >= threshold:
                return True
            if self.has_activated_descendant(c_idx, cls_probs, threshold):
                return True

        return False
