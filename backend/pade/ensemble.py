"""
backend/pade/ensemble.py
GAP-6 fix: CRSResult now includes ai_recommendation field.
"""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class CRSResult:
    crs: float
    decision: str
    gat_prob: float
    lstm_prob: float          # kept for API compat, not used in inference
    stage_data: Dict[str, Any]
    ai_recommendation: str = field(default="")  # GAP-6 fix


def compute_crs(gat_prob: float, temporal_prob: float = None) -> float:
    """
    Compute CRS. Currently uses only the GAT probability.
    (LSTM temporal model excluded — poor performance on real data.)
    """
    return gat_prob


def classify_crs(crs: float) -> str:
    """
    Chapter 4 §4.4 — Decision thresholds (defaults, overridden by live policy_config).
    These are static defaults; the PEG router always reads from DB.
    """
    if crs >= 0.90:
        return "BLOCK"
    if crs >= 0.75:
        return "AUTO_OPTIMISE"
    if crs >= 0.50:
        return "WARN"
    return "ALLOW"
