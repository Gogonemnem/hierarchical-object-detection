import math

import torch
from torch import nn, Tensor

class EmbeddingClassifier(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    embeddings: Tensor

    def __init__(
            self, in_features, out_features,
            hierarchy_structure=None, device=None, dtype=None,
            ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.projection = nn.Linear(in_features, in_features, device=device, dtype=dtype)
        self.embeddings = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
            )
        self.logit_scale = nn.Parameter(torch.tensor(1.0).log())  # log(1.0) = 0.0
        self.reset_parameters()
        self.metric = 'euclidean'  # or 'hyperbolic'

    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.embeddings, a=math.sqrt(5))

    def forward(self, features):  # (bs, num_queries, dim)]
        features = self.projection(features)
        dists = torch.cdist(features, self.embeddings.unsqueeze(0), p=2)
        scale = self.logit_scale.exp().clamp(max=100)
        logits = -dists * scale
        # print("logits stats:", logits.min().item(), logits.max().item())
        return logits
