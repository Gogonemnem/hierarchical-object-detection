import copy

from torch import nn
from mmdet.registry import MODELS
from mmdet.models.dense_heads import DINOHead

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
