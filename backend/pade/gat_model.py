"""Backend GAT wrapper aligned to the canonical CostGuard architecture."""
from __future__ import annotations

from typing import Optional

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    _HAS_TORCH = False

from ._canonical import canonical_attr


if _HAS_TORCH:

    class PipelineGAT(nn.Module):
        """Thin adapter around the canonical GATv2 pipeline with backend feature dims."""

        def __init__(self, in_channels: int, hidden: int = 128, heads: int = 4, dropout: float = 0.30):
            super().__init__()
            model_cls = canonical_attr("GATv2Pipeline")
            self.model = model_cls(
                n_node_feat=in_channels,
                hidden=hidden,
                heads=heads,
                num_layers=3,
                dropout=dropout,
            )

        def forward(self, x, edge_index, batch, edge_attr: Optional["torch.Tensor"] = None):
            return self.model(x, edge_index, edge_attr, batch)

else:

    class PipelineGAT:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to instantiate PipelineGAT.")
