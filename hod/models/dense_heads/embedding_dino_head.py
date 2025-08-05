import copy

from torch import nn

from mmengine.fileio import load
from mmdet.registry import MODELS
from mmdet.utils import OptConfigType

from hod.models.dense_heads.filtered_dino_head import FilteredDINOHead
from hod.models.layers import EmbeddingClassifier
from hod.utils import HierarchyTree

@MODELS.register_module()
class EmbeddingDINOHead(FilteredDINOHead):
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

        # Merge classification configuration with curvature directly
        if cls_config is None:
            cls_config = {}
        cls_config['curvature'] = cls_curvature
        self.cls_config = cls_config

        use_embed_loss = loss_embed and isinstance(loss_embed, dict)
        self.use_cone = (
            use_embed_loss
            and (loss_embed or {}).get('type', None) == "EntailmentConeLoss"
            and (loss_embed or {}).get('loss_weight', 0.0) > 0
        )
        self.cls_config['use_cone'] = self.use_cone

        self.use_contrastive = (
            use_embed_loss
            and (loss_embed or {}).get('type', None) == "HierarchicalContrastiveLoss"
            and (loss_embed or {}).get('loss_weight', 0.0) > 0
        )

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

    def _initialize_cls_branches(self, fc_cls):
        """Helper method to initialize classification branches."""
        if self.share_cls_layer:
            return nn.ModuleList([fc_cls for _ in range(self.num_pred_layer)])
        else:
            return nn.ModuleList([copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])

    def _process_taxonomy(self) -> None:
        """
        Process taxonomy to set clip exempt indices if applicable.

        Raises:
            ValueError: If the taxonomy tree or its root is not properly initialized.
        """
        if self.tree is None or getattr(self.tree, 'root', None) is None:
            raise ValueError("Taxonomy tree or its root is not properly initialized.")

        root_idx = self.class_to_idx[self.tree.root.name]
        self.cls_config['clip_exempt_indices'] = [root_idx]

    def _init_layers(self, *args) -> None:
        """Initialize classification branch of head."""
        super()._init_layers(*args)
        self._process_taxonomy()

        fc_cls = EmbeddingClassifier(self.embed_dims, self.cls_out_channels, **self.cls_config)
        self.cls_branches = self._initialize_cls_branches(fc_cls)

    def _calculate_entail_loss(self, prototypes):
        """
        Calculate entailment-cone loss for prototypes.

        Args:
            prototypes (Tensor): The prototype embeddings for the classes.

        Returns:
            Tensor: The calculated entailment-cone loss.
        """
        return self.loss_embed(prototypes)

    def _calculate_contrastive_loss(self, prototypes, cls_branch):
        """
        Calculate contrastive loss for prototypes.

        Args:
            prototypes (Tensor): The prototype embeddings for the classes.
            cls_branch (EmbeddingClassifier): The classification branch used to compute distance logits.

        Returns:
            Tensor: The calculated contrastive loss.
        """
        distance_matrix = cls_branch.get_distance_logits(prototypes.unsqueeze(0), prototypes)
        return self.loss_embed(distance_matrix.squeeze(0))

    def _calculate_loss_for_branches(self, loss_type, prototypes_fn, loss_fn):
        """
        Helper method to calculate loss for shared and non-shared classification branches.

        Args:
            loss_type (str): The type of loss (e.g., 'entail', 'contrastive').
            prototypes_fn (Callable[[nn.Module], object]): A function to extract prototypes from a classification branch.
            loss_fn (Callable[[object, nn.Module], object]): A function to calculate the loss given prototypes and a classification branch.

        Returns:
            Dict[str, object]: A dictionary containing the calculated losses for each branch.
        """
        loss_dict = {}
        if self.share_cls_layer:
            prototypes = prototypes_fn(self.cls_branches[0])
            loss_value = loss_fn(prototypes, self.cls_branches[0])
            loss_dict[f'loss_{loss_type}'] = loss_value * self.num_pred_layer
        else:
            for i, cls_branch in enumerate(self.cls_branches):
                prototypes = prototypes_fn(cls_branch)
                loss_value = loss_fn(prototypes, cls_branch)
                loss_key = f'loss_{loss_type}' if i == self.num_pred_layer - 1 else f'd{i}.loss_{loss_type}'
                loss_dict[loss_key] = loss_value
        return loss_dict

    def loss(self, *args, **kwargs):
        """
        Compute the total loss, including DINO losses and optional entailment-cone or contrastive losses.

        Args:
            *args: Positional arguments passed to the parent class's loss method.
            **kwargs: Keyword arguments passed to the parent class's loss method.

        Returns:
            dict: A dictionary containing the total loss and its components.
        """
        # normal DINO losses
        loss_dict = super().loss(*args, **kwargs)

        if self.use_cone:
            entail_loss_dict = self._calculate_loss_for_branches(
                'entail',
                lambda branch: branch.prototypes,
                lambda prototypes, _: self._calculate_entail_loss(prototypes)
            )
            loss_dict.update(entail_loss_dict)

        if self.use_contrastive:
            contrastive_loss_dict = self._calculate_loss_for_branches(
                'contrastive',
                lambda branch: branch.prototypes,
                self._calculate_contrastive_loss
            )
            loss_dict.update(contrastive_loss_dict)

        return loss_dict
