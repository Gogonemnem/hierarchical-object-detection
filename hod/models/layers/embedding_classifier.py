import math
from typing import List

import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter


class EmbeddingClassifier(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    embeddings: Tensor

    def __init__(self,
                 in_features,
                 out_features,
                 use_bias: bool=True,
                 use_temperature: bool=True,
                 curvature: float=0.0,
                 use_cone: bool=False,
                 cone_beta: float=0.1,
                 init_norm_upper_offset: float=0.5,
                 clip_exempt_indices: List[int] | None = None,
                 freeze_embeddings: bool=False,
                 device=None,
                 dtype=None,
            ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_cone = use_cone
        self.use_temperature = use_temperature
        assert curvature <= 0.0, "curvature must be == 0.0 (Euclidean space) or -1.0 (Hyperbolic Poincaré space)"
        self.curvature = curvature

        if use_cone:
            assert cone_beta > 0, "cone_beta must be positive"
        self.cone_beta = cone_beta

        # Calculate the runtime minimum norm threshold
        if curvature == 0.0:
            self.runtime_min_norm_threshold = self.cone_beta
        elif curvature == -1.0:
            self.runtime_min_norm_threshold = (2*cone_beta) /(1+math.sqrt(1.0 + 4*cone_beta**2))
        self.clip_exempt = set(clip_exempt_indices or [])

        self.projection = nn.Linear(in_features, in_features, device=device, dtype=dtype)
        self.embeddings = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
            )
        # Set requires_grad based on the new parameter
        if freeze_embeddings:
            self.embeddings.requires_grad_(False)
        if use_bias:
            self.geometric_bias = Parameter(torch.empty(1, **factory_kwargs))
        else:
            self.register_parameter("geometric_bias", None)
        
        if use_temperature:
            self.logit_scale = nn.Parameter(torch.tensor(1.0).log())  # log(1.0) = 0.0
        else:
            self.register_parameter("logit_scale", None)
        
        self.reset_parameters(init_norm_upper_offset)

    def reset_parameters(self, init_norm_upper_offset=None) -> None:
        # 1. Initial Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.embeddings, a=math.sqrt(5))
        if self.geometric_bias is not None:
            nn.init.constant_(self.geometric_bias, 9.0)
            # nn.init.uniform_(self.geometric_bias, a=5, b=15)

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
        raw_embeddings = self.embeddings
        if self.curvature < 0.0:
            raw_embeddings = self.expmap0(raw_embeddings, c=self.curvature)
        
        # Apply norm clipping if cone loss is used (use_cone is set at init)
        if self.use_cone:
            # _apply_min_norm_clipping handles the exempt_mask internally.
            # It returns a new tensor if clipping occurs, otherwise the original.
            # This is differentiable.
            raw_embeddings = self._apply_min_norm_clipping(raw_embeddings)
            
        return raw_embeddings

    def expmap0(self, x: torch.Tensor, c: float = -1.0, eps: float = 1e-5):
        # x: (..., d) unconstrained in ℝᵈ
        x_norm = torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=eps)
        curve_norm = torch.sqrt(torch.tensor(-c)) * x_norm
        scale = torch.tanh(curve_norm) / curve_norm
        return scale * x

    def _apply_min_norm_clipping(self, tensor_to_clip: torch.Tensor) -> torch.Tensor:
        """
        Ensure every row of `W` has L2-norm ≥ runtime_min_norm_threshold,
        except the indices listed in `self.clip_exempt`.

        Parameters
        ----------
        tensor_to_clip : (C, d) tensor   - prototype matrix (may be a view!)

        Returns
        -------
        (C, d) tensor with the same storage as `tensor_to_clip` if no clipping happened,
        otherwise a cloned-and-modified copy (safe for autograd).
        """
        if not self.use_cone:                         # cone disabled → no clip
            return tensor_to_clip

        C = tensor_to_clip.size(0)
        if not self.clip_exempt:                      # fast path, no exempt rows
            exempt_mask = None
        else:
            exempt_mask = torch.zeros(
                C, dtype=torch.bool, device=tensor_to_clip.device)
            exempt_mask[list(self.clip_exempt)] = True

        norms = tensor_to_clip.norm(dim=-1, keepdim=True)          # (C, 1)
        tiny_eps = 1e-12
        needs_clip = (norms < self.runtime_min_norm_threshold) & (norms > tiny_eps)
        if exempt_mask is not None:
            needs_clip &= ~exempt_mask.unsqueeze(1)   # skip exempt rows

        if not needs_clip.any():                      # nothing to do
            return tensor_to_clip

        scale = self.runtime_min_norm_threshold / norms.clamp(min=tiny_eps)
        scaled = tensor_to_clip * scale                            # same shape
        clipped_tensor = torch.where(needs_clip, scaled, tensor_to_clip)
        return clipped_tensor

    def forward(self, features):  # (bs, num_queries, dim)]
        features = self.get_projected_features(features)
        logits = self.get_distance_logits(features, self.prototypes)
        # Add the bias if it exists
        if self.geometric_bias is not None:
            logits = logits + self.geometric_bias # bias (shape [1]) will broadcast
        return logits

    def get_projected_features(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        if self.curvature < 0.0:
            x = self.expmap0(x, self.curvature)
        return x

    def get_distance_logits(self, features, embeddings):
        if self.curvature == 0.0:
            dists = torch.cdist(features, embeddings.unsqueeze(0), p=2)
        elif self.curvature == -1.0:
            dists = self.pairwise_poincare_distance(features, embeddings)
        else:
            raise NotImplementedError("Invalid curvature value. Must be 0.0 or -1.0.")
        
        if self.use_temperature and self.logit_scale is not None:
            scale = self.logit_scale.exp().clamp(max=100)
            logits = -dists * scale
        else:
            logits = -dists
        
        return logits

    def pairwise_poincare_distance(self, x, y, eps=1e-5,
                                   chunk_size: int | None = None):
        """
        x : (B, N, D)
        y : (C, D)      -- *no* batch dim needed, broadcast is free
        Returns : (B, N, C)
        Optional `chunk_size` lets you trade a bit of speed for a lower peak‑RAM.
        """
        B, N, D = x.shape
        C = y.shape[0]
        x_norm_sq = (x.square()).sum(-1, keepdim=True)       # (B, N, 1)
        y_norm_sq = (y.square()).sum(-1, keepdim=True)       # (C, 1)

        def _compute(x_norm_sq, y_norm_sq, y_block):
            dot   = torch.einsum('bnd,cd->bnc', x, y_block)  # (B, N, C')
            diff2 = x_norm_sq + y_norm_sq.t() - 2.0 * dot    # (B, N, C')
            num   = 2.0 * diff2
            denom = (1 - x_norm_sq) * (1 - y_norm_sq.t())
            denom = denom.clamp_min(eps)
            arg   = 1 + num / denom
            return torch.acosh(arg.clamp_min(1.0 + eps))     # (B, N, C')

        if chunk_size is None:
            return _compute(x_norm_sq, y_norm_sq, y)         # full matrix
        else:
            outs = []
            for i in range(0, C, chunk_size):
                y_blk = y[i:i+chunk_size]
                outs.append(_compute(x_norm_sq, y_norm_sq[i:i+chunk_size], y_blk))
            return torch.cat(outs, dim=2)
