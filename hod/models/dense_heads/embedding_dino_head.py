from torch import nn, Tensor
from torch.nn import functional as F

from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.models.dense_heads import DINOHead
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

from hod.models.layers import EmbeddingClassifier

@MODELS.register_module()
class EmbeddingDINOHead(DINOHead):
    def _init_layers(self, *args) -> None:
        """Initialize classification branch of head."""
        super()._init_layers(*args)
        fc_cls = EmbeddingClassifier(self.embed_dims, self.cls_out_channels)

        # if self.share_pred_layer:
        # Unlike standard DINO, we do NOT create separate classifiers per layer
        # to preserve semantic consistency across layers in embedding space.
        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred_layer)])

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
            # cls_score shape: [num_queries, num_classes]
            cls_prob = cls_score.sigmoid()
            # Find the max score and class index per query
            # scores_per_query shape: [num_queries]
            # labels_per_query shape: [num_queries]
            scores_per_query, labels_per_query = cls_prob.max(dim=-1)
            # Select top-k queries based on their max score
            # scores shape: [max_per_img]
            # bbox_index shape: [max_per_img]
            scores, bbox_index = scores_per_query.topk(max_per_img)
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
