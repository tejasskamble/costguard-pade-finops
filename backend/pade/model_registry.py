"""Resolve trained model artifact locations from completed IEEE trial outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

CANONICAL_DOMAINS = ("synthetic", "real", "bitbrains")


def _read_status(manifest_path: Path) -> str:
    if not manifest_path.exists():
        return "missing"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return "invalid_manifest"
    return str(payload.get("status", "unknown")).lower()


def _domain_artifacts(seed_root: Path, domain: str) -> Dict[str, Any]:
    checkpoint_dir = seed_root / domain / "run_1" / "checkpoints"
    artifact_names = (
        "lstm_best.pt",
        "gat_best.pt",
        "lstm_ckpt.pt",
        "gat_ckpt.pt",
    )
    artifacts: Dict[str, Dict[str, Any]] = {}
    for artifact_name in artifact_names:
        artifact_path = checkpoint_dir / artifact_name
        artifacts[artifact_name] = {
            "path": str(artifact_path),
            "exists": artifact_path.exists() and artifact_path.is_file() and artifact_path.stat().st_size > 0,
        }
    return {
        "domain": domain,
        "checkpoint_dir": str(checkpoint_dir),
        "artifacts": artifacts,
    }


def collect_model_registry(results_root: Path, seeds: Sequence[int]) -> Dict[str, Any]:
    """Collect checkpoint availability for each canonical domain across seeds."""
    trials_root = results_root / "trials"
    seed_entries = []
    for seed in seeds:
        seed_root = trials_root / f"seed_{seed}"
        manifest_path = seed_root / "trial_manifest.json"
        status = _read_status(manifest_path)
        seed_entries.append(
            {
                "seed": seed,
                "status": status,
                "seed_root": str(seed_root),
                "domains": [_domain_artifacts(seed_root, domain) for domain in CANONICAL_DOMAINS],
            }
        )
    return {
        "results_root": str(results_root),
        "seeds": seed_entries,
    }

