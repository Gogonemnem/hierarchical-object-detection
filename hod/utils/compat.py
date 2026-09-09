"""Checkpoint-loading compatibility helpers."""

import torch

_original_torch_load = torch.load


def patch_torch_load():
    """Force ``weights_only=False`` on ``torch.load``.

    PyTorch >= 2.6 switched the default of ``weights_only`` to ``True``,
    which breaks loading of mmengine checkpoints (``ConfigDict``,
    ``HistoryBuffer`` and other full objects). Only use with checkpoints
    from trusted sources.
    """
    real_load = _original_torch_load

    def safe_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return real_load(*args, **kwargs)

    torch.load = safe_load
