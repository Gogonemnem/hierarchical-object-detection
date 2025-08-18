import torch
from torch import Tensor

from mmengine.fileio import load
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.models.dense_heads import DINOHead
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

from hod.utils import HierarchyTree


@MODELS.register_module()
class FilteredDINOHead(DINOHead):
    def __init__(self,
                 *args,
                 ann_file: str = '',
                 outputs_per_query: int = -1,
                 **kwargs):
        
        self.load_taxonomy(ann_file)
        super().__init__(*args, **kwargs)
        num_classes = self.num_classes
        # self.outputs_per_query = max(min(outputs_per_query, num_classes), -2)
        self.outputs_per_query = -2
        if self.tree is None and self.outputs_per_query == -2:
            self.outputs_per_query = -1

    def load_taxonomy(self, ann_file):
        if not ann_file:
            self.tree = None
            self.class_to_idx = {}
            return

        ann = load(ann_file)
        taxonomy = ann.get('taxonomy', {})
        self.tree = HierarchyTree(taxonomy)
        self.class_to_idx = {c['name']: c['id'] for c in ann['categories']}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

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
        elif self.outputs_per_query == -2:
            # --- Local-Maximum Suppression (LMS-0) ---
            if self.tree is None:
                raise ValueError("LMS-0 requires a class hierarchy, but `self.tree` is None. "
                                 "Please provide a `class_hierarchy_path` during model initialization.")
            # For each query, keep a detection only if its score is >= scores of all its descendants.
            # This is implemented efficiently with a single bottom-up pass.
            
            # 1. Get scores for all (query, class) pairs.
            subtree_max_scores = cls_score.clone() # num_queries, num_classes`
            
            # keep = torch.ones_like(cls_score, dtype=torch.bool)

            # 2. Traverse hierarchy bottom-up. For each node, update parent scores.
            # A parent's score is the max of its current score and its children's scores.
            # This ensures that after the loop, a node's score is the max within its subtree.
            sorted_nodes = self.tree.get_topo_sorted_nodes()
            for node in sorted_nodes:
                if node.parent:
                    parent_name = node.parent.name
                    parent_idx = self.class_to_idx[parent_name]
                    child_idx = self.class_to_idx[node.name]
                    torch.maximum(subtree_max_scores[:, parent_idx],
                        subtree_max_scores[:, child_idx],
                        out=subtree_max_scores[:, parent_idx])
            
            keep_desc = cls_score > (subtree_max_scores - 1e-6)

            # 3) prune children whose ancestors are stronger
            keep = keep_desc.clone()
            for node in sorted_nodes:              # leaves→root
                n_idx = self.class_to_idx[node.name]
                rows = keep[:, n_idx].nonzero(as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                anc = node.parent
                while anc is not None:
                    a_idx = self.class_to_idx[anc.name]
                    # drop child where ancestor is strictly higher
                    higher_anc = cls_score[rows, a_idx] > cls_score[rows, n_idx]
                    keep[rows[higher_anc], n_idx] = False
                    anc = anc.parent

            # 6. Flatten the results for global top-k selection.
            flat_scores = cls_score[keep]
            flat_labels = torch.where(keep)[1]
            flat_bbox_idx = torch.where(keep)[0]
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
