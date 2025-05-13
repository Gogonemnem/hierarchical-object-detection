import math

import torch
from torch import nn, Tensor

class EmbeddingClassifier(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    embeddings: Tensor

    def __init__(self,
                 in_features,
                 out_features,
                 use_cone: bool=False,
                 cone_beta: float=0.1,
                 cone_eps: float=1e-6,
                 init_norm_upper_offset: float=0.5,
                 device=None,
                 dtype=None,
            ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_cone = use_cone
        self.metric = 'euclidean'  # or 'hyperbolic'

        assert cone_beta > 0, "cone_beta must be positive"
        assert cone_eps > 0, "cone_eps must be positive"
        assert init_norm_upper_offset > cone_eps, "init_norm_upper_offset must be greater than cone_eps"
        self.cone_beta = cone_beta
        self.cone_eps = cone_eps
        
        # Calculate the runtime minimum norm threshold
        self.runtime_min_norm_threshold = self.cone_beta + self.cone_eps
        self.reset_parameters(init_norm_upper_offset)

        self.projection = nn.Linear(in_features, in_features, device=device, dtype=dtype)
        self.embeddings = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
            )
        self.logit_scale = nn.Parameter(torch.tensor(1.0).log())  # log(1.0) = 0.0

    def reset_parameters(self, init_norm_upper_offset=None) -> None:
        # 1. Initial Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.embeddings, a=math.sqrt(5))

        if not self.use_cone:
            return

        # 2. Post-initialization adjustment to match Ganea norm range
        with torch.no_grad():
            if init_norm_upper_offset is not None:
                # 2a. Normalize to unit length (as a starting point for controlled rescaling)
                current_norms = self.embeddings.data.norm(dim=-1, keepdim=True)
                normalized_embeddings = self.embeddings.data / current_norms.clamp(min=1e-12)

                # 2b. Determine target norm range for initialization
                # Lower bound for init is the same as runtime min norm threshold
                init_lower_norm_bound = self.runtime_min_norm_threshold
                init_upper_norm_bound = self.cone_beta + init_norm_upper_offset

                # 2c. Generate random scales in this range for each embedding
                scales = torch.rand(self.embeddings.shape[0], 1, device=self.embeddings.device, dtype=self.embeddings.dtype) * \
                        (init_upper_norm_bound - init_lower_norm_bound) + init_lower_norm_bound

                self.embeddings.data = normalized_embeddings * scales

            # 2d. Final clip to ensure min_norm is strictly met after random scaling (safety)
            self.embeddings.data = self._apply_min_norm_clipping(self.embeddings.data)

    @property
    def prototypes(self):
        if self.use_cone:
            # Apply runtime minimum norm clipping before returning embeddings
            return self._apply_min_norm_clipping(self.embeddings)
        return self.embeddings

    def _apply_min_norm_clipping(self, tensor_to_clip: torch.Tensor) -> torch.Tensor:
        """Clips tensor rows to have a minimum L2 norm."""
        norms = tensor_to_clip.norm(dim=-1, keepdim=True)
        # Only scale up if norm is > some tiny_eps to avoid scaling up zero vectors
        tiny_eps = 1e-12
        needs_clipping_mask = (norms < self.runtime_min_norm_threshold) & (norms > tiny_eps)

        # Use .detach() for scaling_factor if you don't want this op to affect gradients
        # However, the Ganea code implies the clipping is part of the forward pass that affects gradients.
        scaling_factor = self.runtime_min_norm_threshold / norms.clamp(min=tiny_eps)

        clipped_tensor = torch.where(
            needs_clipping_mask,
            tensor_to_clip * scaling_factor,
            tensor_to_clip
        )
        return clipped_tensor

    def forward(self, features):  # (bs, num_queries, dim)]
        features = self.projection(features)

        # Use the .prototypes property, which handles runtime min-norm clipping
        current_prototypes = self.prototypes

        dists = torch.cdist(features, current_prototypes.unsqueeze(0), p=2)
        scale = self.logit_scale.exp().clamp(max=100)
        logits = -dists * scale
        return logits
