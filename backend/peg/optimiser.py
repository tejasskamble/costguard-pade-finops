import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# Chapter 4 §4.5 — AUTO_OPTIMISE catalogue
CATALOGUE = {
    "switch_to_spot_instance": {
        "avg_savings_pct": 65,
        "applies_to": ["deploy_prod", "deploy_staging"],
        "description": "Switch executor to AWS Spot / GCP Preemptible instance"
    },
    "enable_build_cache": {
        "avg_savings_pct": 30,
        "applies_to": ["build", "checkout"],
        "description": "Enable persistent dependency cache for build stage"
    },
    "reduce_test_parallelism": {
        "avg_savings_pct": 45,
        "applies_to": ["unit_test", "integration_test"],
        "description": "Halve the number of parallel test workers"
    },
    "scope_limited_test_suite": {
        "avg_savings_pct": 55,
        "applies_to": ["unit_test", "integration_test", "security_scan"],
        "description": "Run only tests covering changed files in this commit"
    },
}

def select_optimisation(stage_name: str) -> Optional[Dict]:
    """Return the first action that applies to the given stage, or None."""
    for action_name, action in CATALOGUE.items():
        if stage_name in action["applies_to"]:
            return {"name": action_name, **action}
    return None

@dataclass
class OptimisationResult:
    action_name: str
    avg_savings_pct: int
    description: str
    applied_at: str  # ISO timestamp

def apply_optimisation(stage_name: str) -> OptimisationResult:
    """
    Select and record an optimisation action.
    If no specific match, return a default (scope_limited_test_suite).
    """
    action = select_optimisation(stage_name)
    if action is None:
        # Default fallback
        action = {"name": "scope_limited_test_suite", **CATALOGUE["scope_limited_test_suite"]}
    return OptimisationResult(
        action_name=action["name"],
        avg_savings_pct=action["avg_savings_pct"],
        description=action["description"],
        applied_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
