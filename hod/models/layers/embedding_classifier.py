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
                 curvature: float=0.0,
                 use_cone: bool=False,
                 cone_beta: float=0.1,
                 init_norm_upper_offset: float=0.5,
                 device=None,
                 dtype=None,
            ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_cone = use_cone
        assert curvature <= 0.0, "curvature must be == 0.0 (Euclidean space) or -1.0 (Hyperbolic Poincaré space)"
        self.curvature = curvature

        assert cone_beta > 0, "cone_beta must be positive"
        self.cone_beta = cone_beta

        # Calculate the runtime minimum norm threshold
        if curvature == 0.0:
            self.runtime_min_norm_threshold = self.cone_beta
        elif curvature == -1.0:
            self.runtime_min_norm_threshold = (2*cone_beta) /(1+math.sqrt(1.0 + 4*cone_beta**2))

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
                if self.curvature == 0.0:
                    init_upper_norm_bound = self.cone_beta + init_norm_upper_offset
                elif self.curvature == -1.0:
                    init_upper_norm_bound = 1
                else:
                    raise NotImplementedError("Invalid curvature value. Must be 0.0 or -1.0.")

                # 2c. Generate random scales in this range for each embedding
                scales = torch.rand(self.embeddings.shape[0], 1, device=self.embeddings.device, dtype=self.embeddings.dtype) * \
                        (init_upper_norm_bound - init_lower_norm_bound) + init_lower_norm_bound

                self.embeddings.data = normalized_embeddings * scales

            # 2d. Final clip to ensure min_norm is strictly met after random scaling (safety)
            self.embeddings.data = self._apply_min_norm_clipping(self.embeddings.data)

    @property
    def prototypes(self):
        embeddings = self.embeddings
        if self.curvature < 0.0:
            embeddings = self.expmap0(embeddings, self.curvature)
        if self.use_cone:
            embeddings = self._apply_min_norm_clipping(embeddings)
        return embeddings

    def expmap0(self, x: torch.Tensor, c: float = -1.0, eps: float = 1e-5):
        # x: (..., d) unconstrained in ℝᵈ
        x_norm = torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=eps)
        curve_norm = torch.sqrt(-c) * x_norm
        scale = torch.tanh(curve_norm) / curve_norm
        return scale * x

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

        if self.curvature < 0.0:
            # Hyperbolic space
            features = self.expmap0(features, self.curvature)

        # Use the .prototypes property, which handles runtime min-norm clipping
        current_prototypes = self.prototypes

        if self.curvature == 0.0:
            dists = torch.cdist(features, current_prototypes.unsqueeze(0), p=2)
        elif self.curvature == -1.0:
            dists = self.pairwise_poincare_distance(features.unsqueeze(1), current_prototypes.unsqueeze(0))
        else:
            raise NotImplementedError("Invalid curvature value. Must be 0.0 or -1.0.")
        scale = self.logit_scale.exp().clamp(max=100)
        logits = -dists * scale
        return logits

    def pairwise_poincare_distance(self, x, y, eps=1e-5):
        # x: (B, N, D)
        # y: (C, D)
        B, N, D = x.shape
        C = y.shape[0]
        x_exp = x.unsqueeze(2)         # (B, N, 1, D)
        y_exp = y.unsqueeze(0).unsqueeze(0)  # (1, 1, C, D)
        diff = x_exp - y_exp           # (B, N, C, D)

        x_norm_sq = (x_exp**2).sum(dim=-1, keepdim=True)  # (B, N, 1, 1)
        y_norm_sq = (y_exp**2).sum(dim=-1, keepdim=True)  # (1, 1, C, 1)
        diff_norm_sq = (diff**2).sum(dim=-1, keepdim=True)  # (B, N, C, 1)

        num = 2 * diff_norm_sq
        denom = (1 - x_norm_sq) * (1 - y_norm_sq)
        denom = torch.clamp(denom, min=eps)

        argument = 1 + num / denom
        argument = torch.clamp(argument, min=1.0 + eps)
        dist = torch.acosh(argument)
        return dist.squeeze(-1)  # (B, N, C)
