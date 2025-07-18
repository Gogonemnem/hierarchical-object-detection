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
                 cls_config: OptConfigType=None,
                 loss_embed: OptConfigType=None,
                 **kwargs):
        """
        Args:
            ann_file (str): Path to the annotation file containing the
                taxonomy. The file should be in COCO format.
            cls_curvature (float): Curvature parameter for the embedding space.
            share_cls_layer (bool): Whether to share the classification
                layer across all prediction layers.
            
            cls_config (dict, optional): Configuration for the classification
                layer. Defaults to None (uses default values).
                Example config:
                cls_config=dict(
                    use_bias=True,
                    use_temperature=True,
                    freeze_embeddings=True, Whether to freeze the class embeddings during training.
                )
            loss_embed (dict, optional): Configuration for the
                embedding loss.
                Defaults to None (disabled).
                Example config:
                loss_embed=dict(
                    type='EntailmentConeLoss',
                    beta=0.1,
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
        self.share_cls_layer = share_cls_layer
        
        # Set default cls_config and merge with provided config
        default_cls_config = {}
        self.cls_config = default_cls_config.copy()
        if cls_config and isinstance(cls_config, dict):
            self.cls_config.update(cls_config)

        self.cls_config['curvature'] = cls_curvature

        use_embed_loss = loss_embed and isinstance(loss_embed, dict)
        self.use_cone = use_embed_loss and (loss_embed or {}).get('type', None) == "EntailmentConeLoss" and (loss_embed or {}).get('loss_weight', 0.0) > 0
        self.cls_config['use_cone'] = self.use_cone

        self.use_contrastive = use_embed_loss and (loss_embed or {}).get('type', None) == "HierarchicalContrastiveLoss" and (loss_embed or {}).get('loss_weight', 0.0) > 0

        self.tree = None
        if ann_file:
            self.load_taxonomy(ann_file)

        if self.use_cone and loss_embed:
            self.cls_config['cone_beta'] = loss_embed.get('beta', 0.1)

            loss_embed['curvature'] = cls_curvature

        super().__init__(**kwargs)

        if use_embed_loss and loss_embed:
            self.loss_embed = MODELS.build(loss_embed)

    def load_taxonomy(self, ann_file):
        ann = load(ann_file)
        taxonomy = ann.get('taxonomy', {})
        self.tree = HierarchyTree(taxonomy)
        self.class_to_idx = {c['name']: c['id'] for c in ann['categories']}

    def _init_layers(self, *args) -> None:
        """Initialize classification branch of head."""
        super()._init_layers(*args)
        full_taxonomy = (self.tree is not None) and (len(self.tree) == self.cls_out_channels)

        if full_taxonomy:
            root_idx  = self.class_to_idx[self.tree.root.name]
            clip_exempt = [root_idx]

            self.cls_config['clip_exempt_indices'] = clip_exempt

        fc_cls = EmbeddingClassifier(self.embed_dims,
                                     self.cls_out_channels,
                                     **self.cls_config)

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

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Transform outputs of a single image into bbox results.
        check detr_head.py for more details.
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        img_shape = img_meta['img_shape']
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            # --- Simple multi-label: top-k per query, then global top-k ---
            k_per_query = 1  # You can adjust this value, set to -1 for all labels per query
            num_queries, num_classes = cls_score.shape
            if k_per_query == -1 or k_per_query > num_classes:
                # Use all labels per query
                flat_scores = cls_score.reshape(-1)
                flat_labels = torch.arange(num_classes, device=cls_score.device).repeat(num_queries)
                flat_bbox_idx = torch.arange(num_queries, device=cls_score.device).repeat_interleave(num_classes)
            else:
                # Get top-k per query
                topk_scores, topk_labels = cls_score.topk(k_per_query, dim=1)
                # Flatten all (query, class) pairs
                flat_scores = topk_scores.reshape(-1)
                flat_labels = topk_labels.reshape(-1)
                flat_bbox_idx = torch.arange(num_queries, device=cls_score.device).unsqueeze(1).expand(-1, k_per_query).reshape(-1)
            # Take global top-k for the image
            final_scores, final_indices = flat_scores.topk(max_per_img)
            det_labels = flat_labels[final_indices]
            bbox_index = flat_bbox_idx[final_indices]
            scores = final_scores
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
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
        results.bbox_index = bbox_index
        return results
