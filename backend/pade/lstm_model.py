"""Backend LSTM adapter aligned to the canonical CostGuard architecture."""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    _HAS_TORCH = False

from ._canonical import canonical_attr

SEQ_LEN = 30
META_SIZE = 7


if _HAS_TORCH:

    class CostLSTM(nn.Module):
        """Adapter that reshapes the backend flat vector into canonical LSTM inputs."""

        def __init__(self):
            super().__init__()
            model_cls = canonical_attr("BahdanauBiLSTM")
            self.model = model_cls(n_channels=1, n_ctx=META_SIZE, hidden=256, dropout=0.30, num_layers=3)

        def forward(self, x):
            seq = x[:, :SEQ_LEN].unsqueeze(-1)
            ctx = x[:, SEQ_LEN : SEQ_LEN + META_SIZE]
            logits, _ = self.model(seq, ctx)
            return logits

else:

    class CostLSTM:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to instantiate CostLSTM.")
