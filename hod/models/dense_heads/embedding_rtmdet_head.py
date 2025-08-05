import copy
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple, List
from mmengine.fileio import load
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.models.dense_heads.rtmdet_head import RTMDetSepBNHead
from mmdet.structures.bbox import cat_boxes
from mmdet.utils import OptConfigType
from mmdet.models.layers.transformer import inverse_sigmoid
from hod.models.layers import EmbeddingClassifier
from hod.utils import HierarchyTree
from mmengine.config import ConfigDict

def sigmoid_geometric_mean(x, y):
    return (x.sigmoid() * y.sigmoid()).sqrt()

def filter_scores_and_topk(scores, score_thr, topk, results=None, k_per_anchor=None):
    """
    Filter results using score threshold and topk candidates.
    Supports both global and per-anchor top-k.
    """
    if k_per_anchor is not None and k_per_anchor > 0 and scores.dim() == 2:
        # Top-k per anchor (row)
        topk_scores, topk_labels = scores.topk(k_per_anchor, dim=1)  # (num_anchors, k)
        anchor_idxs = torch.arange(scores.size(0), device=scores.device).unsqueeze(1).expand(-1, k_per_anchor)
        # Flatten all candidates
        scores = topk_scores.reshape(-1)
        labels = topk_labels.reshape(-1)
        anchor_idxs = anchor_idxs.reshape(-1)
    else:
        # Standard: flatten all (anchor, class) pairs above threshold
        valid_mask = scores > score_thr
        scores = scores[valid_mask]
        valid_idxs = torch.nonzero(valid_mask, as_tuple=False)
        if scores.dim() == 0 or scores.numel() == 0:
            # No valid scores
            return scores, torch.empty_like(scores, dtype=torch.long), torch.empty_like(scores, dtype=torch.long), None
        anchor_idxs, labels = valid_idxs.unbind(dim=1)

    # Apply score threshold (again, for per-anchor branch)
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    labels = labels[valid_mask]
    anchor_idxs = anchor_idxs[valid_mask]

    # Global topk
    num_topk = min(topk, scores.size(0))
    if num_topk > 0:
        scores, idxs = scores.sort(descending=True)
        scores = scores[:num_topk]
        labels = labels[idxs[:num_topk]]
        anchor_idxs = anchor_idxs[idxs[:num_topk]]
    else:
        idxs = torch.empty(0, dtype=torch.long, device=scores.device)

    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: (v[anchor_idxs] if isinstance(v, torch.Tensor) else v) for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[anchor_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[anchor_idxs]
        else:
            raise NotImplementedError(f'Only supports dict or list or Tensor, but get {type(results)}.')
    return scores, labels, anchor_idxs, filtered_results

@MODELS.register_module()
class EmbeddingRTMDetSepBNHead(RTMDetSepBNHead):
    def __init__(self,
                 ann_file='',
                 cls_curvature=0.0,
                 cls_config: OptConfigType=None,
                 loss_embed: OptConfigType=None,
                 **kwargs):
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

    def _init_layers(self):
        # Replace last classifier with embedding classifier
        super()._init_layers()
        if self.tree is not None and getattr(self.tree, 'root', None) is not None:
            root_idx = self.class_to_idx[self.tree.root.name]
            clip_exempt = [root_idx]
            self.cls_config['clip_exempt_indices'] = clip_exempt
        # One embedding classifier shared across all FPN levels, as a ModuleList for MMDet compatibility
        shared_classifier = EmbeddingClassifier(self.feat_channels, self.cls_out_channels, **self.cls_config)
        self.rtm_cls = nn.ModuleList([shared_classifier for _ in range(len(self.cls_convs))])

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction

            - cls_scores (tuple[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_anchors * num_classes.
            - bbox_preds (tuple[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_anchors * 4.
        """

        cls_scores = []
        bbox_preds = []
        for idx, (x, stride) in enumerate(
                zip(feats, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x

            # Classification conv stack
            cls_conv = self.cls_convs[idx]
            if isinstance(cls_conv, nn.Sequential):
                cls_feat = cls_conv(cls_feat)
            elif isinstance(cls_conv, nn.ModuleList):
                for layer in cls_conv:
                    cls_feat = layer(cls_feat)
            else:
                cls_feat = cls_conv(cls_feat)

            # Always repeat features for each anchor (priors)
            N, C, H, W = cls_feat.shape
            cls_feat_flat = cls_feat.permute(0, 2, 3, 1).reshape(N, H * W * self.num_base_priors, C)
            cls_score_flat = self.rtm_cls[idx](cls_feat_flat)  # (N, H*W*num_base_priors, num_classes)
            cls_score = cls_score_flat.permute(0, 2, 1).reshape(N, self.num_base_priors * self.cls_out_channels, H, W)

            # Regression conv stack
            reg_conv = self.reg_convs[idx]
            if isinstance(reg_conv, nn.Sequential):
                reg_feat = reg_conv(reg_feat)
            elif isinstance(reg_conv, nn.ModuleList):
                for layer in reg_conv:
                    reg_feat = layer(reg_feat)
            else:
                reg_feat = reg_conv(reg_feat)

            # Objectness branch (if used)
            if self.with_objectness:
                objectness = self.rtm_obj[idx](reg_feat)
                cls_score = inverse_sigmoid(
                    sigmoid_geometric_mean(cls_score, objectness))

            # Regression output
            if self.exp_on_reg:
                reg_dist = self.rtm_reg[idx](reg_feat).exp() * stride[0]
            else:
                reg_dist = self.rtm_reg[idx](reg_feat) * stride[0]

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
        return tuple(cls_scores), tuple(bbox_preds)

    def loss(self, *args, **kwargs):
        # Standard RTMDet losses
        loss_dict = super().loss(*args, **kwargs)
        prototypes = getattr(self.rtm_cls, 'prototypes', None)
        if self.use_cone and prototypes is not None:
            base_entail_loss = self.loss_embed(prototypes)
            loss_dict['loss_entail'] = base_entail_loss * (len(self.rtm_reg) if hasattr(self, 'rtm_reg') else 1)
        if self.use_contrastive and prototypes is not None:
            distance_matrix = self.rtm_cls[0].get_distance_logits(prototypes.unsqueeze(0), prototypes)
            contrastive_loss = self.loss_embed(distance_matrix.squeeze(0))
            loss_dict['loss_contrastive'] = contrastive_loss * (len(self.rtm_reg) if hasattr(self, 'rtm_reg') else 1)
        return loss_dict

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

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
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)

            # the `custom_cls_channels` parameter is derived from
            # CrossEntropyCustomLoss and FocalCustomLoss, and is currently used
            # in v3det.
            if getattr(self.loss_cls, 'custom_cls_channels', False):
                scores = self.loss_cls.get_activation(cls_score)
            elif self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
                # k_per_query = -1
                # if k_per_query != -1:
                #     # Get top-k per query
                #     scores, labels = cls_score.topk(k_per_query, dim=1)
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors, k_per_anchor=1))
            scores, labels, keep_idxs, filtered_results = results

            # Use filtered_results only if it is a dict
            if isinstance(filtered_results, dict):
                bbox_pred = filtered_results['bbox_pred']
                priors = filtered_results['priors']
            # else, keep bbox_pred and priors as is

            if with_score_factors and mlvl_score_factors is not None:
                mlvl_score_factors.append(score_factor)

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)