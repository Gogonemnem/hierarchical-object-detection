import torch
from torch import Tensor

from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.models.dense_heads import DINOHead
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

@MODELS.register_module()
class FilteredDINOHead(DINOHead):
    def __init__(self,
                 *args,
                 outputs_per_query: int = -1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.num_classes
        self.outputs_per_query = max(min(outputs_per_query, num_classes), -1)
    
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

        # DINO uses sigmoid for classification
        # if self.loss_cls.use_sigmoid:
        cls_score = cls_score.sigmoid()

        # --- Simple multi-label: top-k per query, then global top-k ---
        num_queries, num_classes = cls_score.shape
        if self.outputs_per_query == -1:
            # Use all labels per query
            flat_scores = cls_score.reshape(-1)
            flat_labels = torch.arange(num_classes, device=cls_score.device).repeat(num_queries)
            flat_bbox_idx = torch.arange(num_queries, device=cls_score.device).repeat_interleave(num_classes)
        else:
            # Get top-k per query
            topk_scores, topk_labels = cls_score.topk(self.outputs_per_query, dim=1)
            # Flatten all (query, class) pairs
            flat_scores = topk_scores.reshape(-1)
            flat_labels = topk_labels.reshape(-1)
            flat_bbox_idx = torch.arange(num_queries, device=cls_score.device).unsqueeze(1).expand(-1, self.outputs_per_query).reshape(-1)

        # Take global top-k for the image
        final_scores, final_indices = flat_scores.topk(max_per_img)
        det_labels = flat_labels[final_indices]
        bbox_index = flat_bbox_idx[final_indices]
        scores = final_scores
        bbox_pred = bbox_pred[bbox_index]

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
