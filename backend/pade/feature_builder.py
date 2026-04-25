"""Feature builders for backend PADE inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

try:
    from torch_geometric.data import Data as PyGData

    _HAS_PYG = True
except ImportError:  # pragma: no cover
    PyGData = None  # type: ignore[assignment]
    _HAS_PYG = False


STAGE_ORDER = [
    "checkout",
    "build",
    "unit_test",
    "integration_test",
    "security_scan",
    "docker_build",
    "deploy_staging",
    "deploy_prod",
]
EXECUTOR_TYPES = ["github_actions", "gitlab_ci", "jenkins", "travis_ci", "bitbrains_vm"]
BRANCH_TYPES = ["main", "develop", "feature", "release", "hotfix", "production"]

if _HAS_TORCH:
    EDGE_INDEX = torch.tensor(
        [
            [0, 0, 1, 2, 3, 3, 4, 5, 6],
            [1, 2, 3, 3, 4, 5, 6, 6, 7],
        ],
        dtype=torch.long,
    )
else:
    EDGE_INDEX = None


@dataclass
class GraphDataFallback:
    x: Any
    edge_index: Any

    @property
    def num_nodes(self) -> int:
        return int(len(self.x))

    def to(self, *_args, **_kwargs):
        return self


def build_pipeline_graph(stage_data: Dict[str, dict]):
    """Build a graph object for a single pipeline run."""
    node_features = []
    for stage in STAGE_ORDER:
        d = stage_data.get(stage, {})
        node_features.append(
            [
                float(d.get("cost", 0.0)),
                float(d.get("deviation", 0.0)),
                float(d.get("duration", 0.0)),
                float(d.get("hist_avg_cost", 0.0)),
                float(d.get("hist_avg_dur", 0.0)),
                float(d.get("executor_enc", 0.0)),
                float(d.get("branch_enc", 0.0)),
                float(d.get("provider_enc", 0.0)),
            ]
        )
    if _HAS_TORCH and _HAS_PYG:
        x = torch.tensor(node_features, dtype=torch.float32)
        return PyGData(x=x, edge_index=EDGE_INDEX)
    return GraphDataFallback(x=node_features, edge_index=[[0], [0]])


def build_lstm_features(
    deviation_sequence: list,
    stage_name: str,
    executor_type: str,
    branch_type: str,
) -> np.ndarray:
    """Build the backend flat LSTM feature vector."""
    if len(deviation_sequence) < 30:
        deviation_sequence = [0.0] * (30 - len(deviation_sequence)) + deviation_sequence
    elif len(deviation_sequence) > 30:
        deviation_sequence = deviation_sequence[-30:]

    arr = np.asarray(deviation_sequence, dtype=np.float32)
    max_dev = float(np.max(np.abs(arr)))
    mean_dev = float(np.mean(arr))
    std_dev = float(np.std(arr))
    slope = float(np.polyfit(np.arange(30), arr, 1)[0])

    stage_idx = STAGE_ORDER.index(stage_name) if stage_name in STAGE_ORDER else 0
    executor_idx = EXECUTOR_TYPES.index(executor_type) if executor_type in EXECUTOR_TYPES else 0
    branch_idx = BRANCH_TYPES.index(branch_type) if branch_type in BRANCH_TYPES else 0

    return np.concatenate(
        [
            arr,
            np.asarray([max_dev, mean_dev, std_dev, slope], dtype=np.float32),
            np.asarray([stage_idx, executor_idx, branch_idx], dtype=np.float32),
        ]
    ).astype(np.float32)
