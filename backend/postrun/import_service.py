"""Filesystem-first IEEE post-run snapshot and database import helpers."""
from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import asyncpg
import numpy as np

from config import settings
from pade.model_registry import collect_model_registry

logger = logging.getLogger(__name__)

CANONICAL_SEEDS: tuple[int, ...] = (42, 52, 62, 72, 82, 92, 102, 112, 122, 132)
CANONICAL_DOMAINS: tuple[str, ...] = ("synthetic", "real", "bitbrains")
AGGREGATE_FILE_NAMES: tuple[str, ...] = (
    "ieee_aggregate_summary.json",
    "ieee_aggregate_summary.csv",
    "ieee_per_seed_summary.json",
    "ieee_per_seed_summary.csv",
)
WORKSPACE_DOMAIN_DIRS: Dict[str, tuple[str, ...]] = {
    "synthetic": ("synthetic_raw", "ml_ready_synthetic"),
    "real": ("real_data", "ml_ready_real"),
    "bitbrains": ("bitbrains_data", "ml_ready_bitbrains"),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_results_root(override: Optional[str] = None) -> Path:
    if override:
        candidate = Path(override).expanduser()
    else:
        configured = (settings.POSTRUN_RESULTS_ROOT or "results").strip()
        candidate = Path(configured)
    if not candidate.is_absolute():
        candidate = (_repo_root() / candidate).resolve()
    return candidate


def _load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to parse JSON: %s", path, exc_info=True)
        return None


def _iter_csv_chunks(path: Path, chunk_size: int) -> Iterable[list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        chunk: list[dict[str, str]] = []
        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    with suppress(ValueError):
        return float(text)
    return None


def _to_int(value: Any) -> Optional[int]:
    flt = _to_float(value)
    if flt is None:
        return None
    with suppress(ValueError, TypeError):
        return int(flt)
    return None


def _to_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    with suppress(ValueError):
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def _aggregate_file_status(results_root: Path) -> Dict[str, Dict[str, Any]]:
    aggregate_root = results_root / "aggregate"
    status: Dict[str, Dict[str, Any]] = {}
    for name in AGGREGATE_FILE_NAMES:
        path = aggregate_root / name
        status[name] = {
            "path": str(path),
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else 0,
        }
    return status


def _load_per_seed_metrics(results_root: Path, chunk_size: int) -> Dict[int, Dict[str, Any]]:
    per_seed_csv = results_root / "aggregate" / "ieee_per_seed_summary.csv"
    if not per_seed_csv.exists():
        return {}
    seed_map: Dict[int, Dict[str, Any]] = {}
    for chunk in _iter_csv_chunks(per_seed_csv, chunk_size):
        for row in chunk:
            seed = _to_int(row.get("seed"))
            if seed is None:
                continue
            typed_row: Dict[str, Any] = {"seed": seed}
            for key, value in row.items():
                if key == "seed":
                    continue
                as_float = _to_float(value)
                typed_row[key] = as_float if as_float is not None else value
            seed_map[seed] = typed_row
    return seed_map


def _seed_entries(results_root: Path, per_seed_metrics: Mapping[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    trials_root = results_root / "trials"
    model_registry = collect_model_registry(results_root, CANONICAL_SEEDS)
    model_by_seed = {
        int(entry["seed"]): entry for entry in model_registry.get("seeds", []) if "seed" in entry
    }
    rows: List[Dict[str, Any]] = []
    for seed in CANONICAL_SEEDS:
        seed_root = trials_root / f"seed_{seed}"
        manifest_path = seed_root / "trial_manifest.json"
        manifest = _load_json(manifest_path) or {}
        status = str(manifest.get("status", "missing")).lower()
        trial_complete = seed_root / "trial_complete.json"
        is_complete = status == "complete" and trial_complete.exists()
        rows.append(
            {
                "seed": seed,
                "status": status,
                "is_complete": is_complete,
                "started_at": manifest.get("started_at"),
                "completed_at": manifest.get("completed_at"),
                "manifest_path": str(manifest_path),
                "trial_complete_path": str(trial_complete),
                "trial_complete_exists": trial_complete.exists(),
                "metrics": dict(per_seed_metrics.get(seed, {})),
                "model_artifacts": dict(model_by_seed.get(seed, {})),
            }
        )
    return rows


def _quality_gate(snapshot: Mapping[str, Any], min_ensemble_f1: float) -> Dict[str, Any]:
    reasons: List[str] = []
    decision = "PASS"
    seed_rows = list(snapshot.get("seed_runs", []))
    total_trials = len(seed_rows)
    completed_trials = sum(1 for row in seed_rows if row.get("is_complete"))
    if completed_trials != total_trials:
        decision = "BLOCK"
        reasons.append(
            f"Incomplete trials detected ({completed_trials}/{total_trials} complete)."
        )

    aggregate_summary = snapshot.get("aggregate_summary", {})
    domains = aggregate_summary.get("domains", {}) if isinstance(aggregate_summary, Mapping) else {}
    for domain in CANONICAL_DOMAINS:
        ens_block = (
            domains.get(domain, {}).get("ens", {}).get("f1_at_opt", {})
            if isinstance(domains, Mapping)
            else {}
        )
        ens_mean = _to_float(ens_block.get("mean") if isinstance(ens_block, Mapping) else None)
        if ens_mean is None:
            if decision != "BLOCK":
                decision = "WARN"
            reasons.append(f"Missing ensemble F1 mean for domain '{domain}'.")
            continue
        if ens_mean < min_ensemble_f1 and decision == "PASS":
            decision = "WARN"
        if ens_mean < min_ensemble_f1:
            reasons.append(
                f"Domain '{domain}' ensemble F1 mean {ens_mean:.4f} is below gate {min_ensemble_f1:.4f}."
            )

    if not reasons:
        reasons.append("All post-run quality checks passed.")
    return {
        "decision": decision,
        "reasons": reasons,
        "thresholds": {"min_ensemble_f1": min_ensemble_f1},
        "source": "inline",
    }


def _flatten_aggregate_metrics(aggregate_summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    domains = aggregate_summary.get("domains", {})
    if isinstance(domains, Mapping):
        for domain_name, model_block in domains.items():
            if not isinstance(model_block, Mapping):
                continue
            for model_name, metric_block in model_block.items():
                if not isinstance(metric_block, Mapping):
                    continue
                scope = f"domain:{domain_name}|model:{model_name}"
                for metric_name, stat_values in metric_block.items():
                    if not isinstance(stat_values, Mapping):
                        continue
                    rows.append(
                        {
                            "scope": scope,
                            "metric_name": str(metric_name),
                            "mean_value": _to_float(stat_values.get("mean")),
                            "std_value": _to_float(stat_values.get("std")),
                            "sample_size": _to_int(stat_values.get("n")),
                        }
                    )

    bwt_block = aggregate_summary.get("bwt")
    if isinstance(bwt_block, Mapping):
        rows.append(
            {
                "scope": "lifelong",
                "metric_name": "bwt",
                "mean_value": _to_float(bwt_block.get("mean")),
                "std_value": _to_float(bwt_block.get("std")),
                "sample_size": _to_int(bwt_block.get("n")),
            }
        )
    return rows


def _latest_run_dir(mode_root: Path) -> Optional[Path]:
    if not mode_root.exists():
        return None
    run_dirs: List[Tuple[int, Path]] = []
    for entry in mode_root.iterdir():
        if not entry.is_dir() or not entry.name.startswith("run_"):
            continue
        with suppress(ValueError):
            run_dirs.append((int(entry.name.split("_", 1)[1]), entry))
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda item: item[0])
    return run_dirs[-1][1]


def _iter_seed_domain_runs(
    results_root: Path,
) -> Iterable[Tuple[int, str, Path, Optional[Path], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]]:
    for seed in CANONICAL_SEEDS:
        seed_root = results_root / "trials" / f"seed_{seed}"
        for domain in CANONICAL_DOMAINS:
            mode_root = seed_root / domain
            run_dir = _latest_run_dir(mode_root)
            run_config: Mapping[str, Any] = {}
            inference_manifest: Mapping[str, Any] = {}
            if run_dir is not None:
                run_config = _load_json(run_dir / "run_config.json") or {}
                inference_manifest = _load_json(run_dir / "inference_manifest.json") or {}
            if not inference_manifest:
                inference_manifest = _load_json(mode_root / "inference_manifest.json") or {}
            best_scores = _load_json(mode_root / "best_scores.json") or {}
            yield seed, domain, mode_root, run_dir, run_config, best_scores, inference_manifest


def _flatten_seed_domain_metrics(
    seed: int,
    domain: str,
    source_path: str,
    run_config: Mapping[str, Any],
    best_scores: Mapping[str, Any],
    inference_manifest: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: Dict[Tuple[int, str, str, str, str], Dict[str, Any]] = {}

    def _add_row(scope: str, model_name: str, metric_name: str, value: Any) -> None:
        metric_value = _to_float(value)
        if metric_value is None:
            return
        key = (seed, domain, scope, model_name, metric_name)
        rows[key] = {
            "seed": seed,
            "domain": domain,
            "metric_scope": scope,
            "model_name": model_name,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "source_path": source_path,
        }

    for scope_name, block in (
        ("test", run_config.get("metrics", {})),
        ("validation", run_config.get("validation_metrics", {})),
    ):
        if not isinstance(block, Mapping):
            continue
        for model_name, metrics in block.items():
            if not isinstance(metrics, Mapping):
                continue
            for metric_name, value in metrics.items():
                _add_row(scope_name, str(model_name), str(metric_name), value)

    baseline_results = run_config.get("baseline_results", {})
    if isinstance(baseline_results, Mapping):
        for baseline_model, split_map in baseline_results.items():
            if not isinstance(split_map, Mapping):
                continue
            for split_name, metrics in split_map.items():
                if not isinstance(metrics, Mapping):
                    continue
                for metric_name, value in metrics.items():
                    _add_row(f"baseline_{split_name}", str(baseline_model), str(metric_name), value)

    for manifest_key in ("lstm_f1", "gat_f1", "ens_f1"):
        model_name = manifest_key.split("_", 1)[0]
        _add_row("manifest", model_name, "f1_at_opt", inference_manifest.get(manifest_key))

    for model_name in ("lstm", "gat", "ens"):
        best_block = best_scores.get(f"best_{model_name}", {})
        if not isinstance(best_block, Mapping):
            continue
        for metric_name, value in best_block.items():
            _add_row("best_scores", model_name, str(metric_name), value)

    return list(rows.values())


def _collect_training_rows(
    seed: int,
    domain: str,
    run_dir: Optional[Path],
    run_config: Mapping[str, Any],
    best_scores: Mapping[str, Any],
    inference_manifest: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    if run_dir is None or not run_config:
        return []
    run_number = _to_int(run_config.get("run_number")) or 1
    return [
        {
            "seed": seed,
            "domain": domain,
            "run_number": run_number,
            "run_dir": str(run_dir),
            "started_at": run_config.get("started_at"),
            "elapsed_s": _to_float(run_config.get("elapsed_s")),
            "note": str(run_config.get("note") or ""),
            "runtime_controls": run_config.get("runtime_controls", {}),
            "hardware_profile": run_config.get("hardware_profile", {}),
            "training_state": run_config.get("training_state", {}),
            "config": {
                "lstm_config": run_config.get("lstm_config", {}),
                "gat_config": run_config.get("gat_config", {}),
                "metrics": run_config.get("metrics", {}),
                "validation_metrics": run_config.get("validation_metrics", {}),
                "baseline_results": run_config.get("baseline_results", {}),
                "best_scores": best_scores,
                "inference_manifest": inference_manifest,
            },
        }
    ]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _iter_array_chunks(path: Path, chunk_size: int = 100_000) -> Iterable[np.ndarray]:
    arr = np.load(path, mmap_mode="r")
    flat = arr.reshape(-1)
    total = int(flat.shape[0])
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        yield np.asarray(flat[start:end], dtype=np.float64)


def _summarize_single_logits(path: Path, threshold: float) -> Tuple[int, int, float]:
    total_samples = 0
    anomaly_count = 0
    running_sum = 0.0
    for chunk in _iter_array_chunks(path):
        probs = _sigmoid(chunk)
        total_samples += int(probs.size)
        anomaly_count += int(np.count_nonzero(probs >= threshold))
        running_sum += float(np.sum(probs))
    mean_score = (running_sum / total_samples) if total_samples > 0 else 0.0
    return total_samples, anomaly_count, mean_score


def _summarize_ensemble_logits(
    lstm_path: Path,
    gat_path: Path,
    threshold: float,
    strategy: str,
    lstm_weight: float,
    gat_weight: float,
    chunk_size: int = 100_000,
) -> Tuple[int, int, float]:
    lstm_arr = np.load(lstm_path, mmap_mode="r").reshape(-1)
    gat_arr = np.load(gat_path, mmap_mode="r").reshape(-1)
    total = min(int(lstm_arr.shape[0]), int(gat_arr.shape[0]))
    if total <= 0:
        return 0, 0, 0.0

    anomaly_count = 0
    running_sum = 0.0
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        lstm_probs = _sigmoid(np.asarray(lstm_arr[start:end], dtype=np.float64))
        gat_probs = _sigmoid(np.asarray(gat_arr[start:end], dtype=np.float64))
        if strategy == "lstm_only":
            ens_probs = lstm_probs
        elif strategy == "gat_only":
            ens_probs = gat_probs
        else:
            ens_probs = (lstm_probs * lstm_weight) + (gat_probs * gat_weight)
        anomaly_count += int(np.count_nonzero(ens_probs >= threshold))
        running_sum += float(np.sum(ens_probs))
    mean_score = running_sum / float(total)
    return total, anomaly_count, mean_score


def _prediction_threshold(run_config: Mapping[str, Any], model_name: str, fallback: float = 0.5) -> float:
    metrics = run_config.get("metrics", {})
    if isinstance(metrics, Mapping):
        model_block = metrics.get(model_name, {})
        if isinstance(model_block, Mapping):
            for key in ("threshold", "opt_threshold"):
                value = _to_float(model_block.get(key))
                if value is not None:
                    return value
    return fallback


def _collect_prediction_rows(
    seed: int,
    domain: str,
    run_dir: Optional[Path],
    run_config: Mapping[str, Any],
    inference_manifest: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    if run_dir is None:
        return []
    predictions_dir = run_dir / "predictions"
    if not predictions_dir.exists():
        return []

    strategy = str(run_config.get("ensemble_strategy") or inference_manifest.get("ensemble_strategy") or "weighted")
    metrics_block = run_config.get("metrics", {})
    if not isinstance(metrics_block, Mapping):
        metrics_block = {}
    ens_block = metrics_block.get("ens", {})
    if not isinstance(ens_block, Mapping):
        ens_block = {}

    lstm_weight = _to_float(ens_block.get("lstm_weight")) or 0.5
    gat_weight = _to_float(ens_block.get("gat_weight")) or 0.5
    if lstm_weight <= 0 and gat_weight <= 0:
        lstm_weight, gat_weight = 0.5, 0.5
    total_weight = lstm_weight + gat_weight
    lstm_weight = lstm_weight / total_weight
    gat_weight = gat_weight / total_weight

    rows: List[Dict[str, Any]] = []
    for split_name in ("test", "val"):
        lstm_path = predictions_dir / f"lstm_{split_name}_logits.npy"
        gat_path = predictions_dir / f"gat_{split_name}_logits.npy"
        lstm_summary: Optional[Tuple[int, int, float, float]] = None
        gat_summary: Optional[Tuple[int, int, float, float]] = None

        if lstm_path.exists():
            threshold = _prediction_threshold(run_config, "lstm")
            total, anomalies, mean_score = _summarize_single_logits(lstm_path, threshold)
            if total > 0:
                lstm_summary = (total, anomalies, mean_score, threshold)
                rows.append(
                    {
                        "seed": seed,
                        "domain": domain,
                        "split_name": split_name,
                        "model_name": "lstm",
                        "total_samples": total,
                        "anomaly_count": anomalies,
                        "anomaly_rate": anomalies / float(total),
                        "threshold": threshold,
                        "mean_score": mean_score,
                        "source_path": str(lstm_path),
                    }
                )

        if gat_path.exists():
            threshold = _prediction_threshold(run_config, "gat")
            total, anomalies, mean_score = _summarize_single_logits(gat_path, threshold)
            if total > 0:
                gat_summary = (total, anomalies, mean_score, threshold)
                rows.append(
                    {
                        "seed": seed,
                        "domain": domain,
                        "split_name": split_name,
                        "model_name": "gat",
                        "total_samples": total,
                        "anomaly_count": anomalies,
                        "anomaly_rate": anomalies / float(total),
                        "threshold": threshold,
                        "mean_score": mean_score,
                        "source_path": str(gat_path),
                    }
                )

        if lstm_summary is not None and gat_summary is not None:
            threshold = _prediction_threshold(run_config, "ens")
            total, anomalies, mean_score = _summarize_ensemble_logits(
                lstm_path=lstm_path,
                gat_path=gat_path,
                threshold=threshold,
                strategy=strategy,
                lstm_weight=lstm_weight,
                gat_weight=gat_weight,
            )
            if total > 0:
                rows.append(
                    {
                        "seed": seed,
                        "domain": domain,
                        "split_name": split_name,
                        "model_name": "ens",
                        "total_samples": total,
                        "anomaly_count": anomalies,
                        "anomaly_rate": anomalies / float(total),
                        "threshold": threshold,
                        "mean_score": mean_score,
                        "source_path": str(predictions_dir),
                    }
                )
        elif strategy == "lstm_only" and lstm_summary is not None:
            total, anomalies, mean_score, _ = lstm_summary
            threshold = _prediction_threshold(
                run_config,
                "ens",
                fallback=_prediction_threshold(run_config, "lstm"),
            )
            rows.append(
                {
                    "seed": seed,
                    "domain": domain,
                    "split_name": split_name,
                    "model_name": "ens",
                    "total_samples": total,
                    "anomaly_count": anomalies,
                    "anomaly_rate": anomalies / float(total),
                    "threshold": threshold,
                    "mean_score": mean_score,
                    "source_path": str(lstm_path),
                }
            )
        elif strategy == "gat_only" and gat_summary is not None:
            total, anomalies, mean_score, _ = gat_summary
            threshold = _prediction_threshold(
                run_config,
                "ens",
                fallback=_prediction_threshold(run_config, "gat"),
            )
            rows.append(
                {
                    "seed": seed,
                    "domain": domain,
                    "split_name": split_name,
                    "model_name": "ens",
                    "total_samples": total,
                    "anomaly_count": anomalies,
                    "anomaly_rate": anomalies / float(total),
                    "threshold": threshold,
                    "mean_score": mean_score,
                    "source_path": str(gat_path),
                }
            )
    return rows


def _scan_csv_shape(path: Path) -> Tuple[Optional[int], Optional[int], List[str], str]:
    row_count = 0
    headers: List[str] = []
    parse_status = "ok"
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header_row = next(reader)
            except StopIteration:
                return 0, 0, [], "empty"
            headers = [str(item) for item in header_row]
            for _ in reader:
                row_count += 1
    except Exception:
        logger.warning("CSV parse failed: %s", path, exc_info=True)
        return None, len(headers) if headers else None, headers[:20], "parse_error"
    return row_count, len(headers), headers[:20], parse_status


def _scan_file_summary(seed: int, domain: str, dataset_name: str, path: Path) -> Dict[str, Any]:
    file_ext = path.suffix.lower()
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    schema_preview: List[str] = []
    parse_status = "skipped"

    if file_ext == ".csv":
        row_count, column_count, schema_preview, parse_status = _scan_csv_shape(path)
    elif file_ext == ".npy":
        try:
            arr = np.load(path, mmap_mode="r")
            shape = list(arr.shape)
            row_count = int(shape[0]) if shape else 0
            column_count = int(shape[1]) if len(shape) > 1 else 1
            schema_preview = [f"shape={shape}"]
            parse_status = "ok"
        except Exception:
            logger.warning("NPY parse failed: %s", path, exc_info=True)
            parse_status = "parse_error"

    return {
        "seed": seed,
        "domain": domain,
        "dataset_name": dataset_name,
        "file_path": str(path),
        "file_name": path.name,
        "file_ext": file_ext,
        "size_bytes": path.stat().st_size,
        "row_count": row_count,
        "column_count": column_count,
        "parse_status": parse_status,
        "schema_preview": schema_preview,
    }


def _iter_dataset_files(root: Path, max_depth: int = 3, max_files: int = 250) -> Iterable[Path]:
    root_depth = len(root.parts)
    yielded = 0
    for current_root, _dirs, files in os.walk(root):
        current_path = Path(current_root)
        depth = len(current_path.parts) - root_depth
        if depth > max_depth:
            continue
        for file_name in files:
            yielded += 1
            if yielded > max_files:
                return
            yield current_path / file_name


def _workspace_roots_for_seed_domain(results_root: Path, seed: int, domain: str) -> Iterable[Tuple[str, Path]]:
    for seed_root in (
        results_root / f"seed_{seed}" / "_workspace",
        results_root / "trials" / f"seed_{seed}" / "_workspace",
    ):
        if not seed_root.exists():
            continue
        for dataset_name in WORKSPACE_DOMAIN_DIRS.get(domain, ()):
            candidate = seed_root / dataset_name
            if candidate.exists():
                yield dataset_name, candidate


def _collect_prepared_dataset_rows(results_root: Path) -> List[Dict[str, Any]]:
    rows: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    for seed in CANONICAL_SEEDS:
        for domain in CANONICAL_DOMAINS:
            for dataset_name, dataset_root in _workspace_roots_for_seed_domain(results_root, seed, domain):
                for file_path in _iter_dataset_files(dataset_root):
                    summary_row = _scan_file_summary(seed, domain, dataset_name, file_path)
                    key = (seed, domain, summary_row["file_path"])
                    rows[key] = summary_row
    return list(rows.values())


def build_postrun_snapshot(
    *,
    results_root_override: Optional[str] = None,
    chunk_size: Optional[int] = None,
    min_ensemble_f1: Optional[float] = None,
) -> Dict[str, Any]:
    """Collect a deterministic snapshot from existing IEEE outputs only."""
    results_root = resolve_results_root(results_root_override)
    chunk_size_value = int(chunk_size or settings.POSTRUN_IMPORT_CHUNK_SIZE or 100_000)
    min_f1_value = float(min_ensemble_f1 if min_ensemble_f1 is not None else settings.POSTRUN_MIN_ENSEMBLE_F1)

    aggregate_file_status = _aggregate_file_status(results_root)
    aggregate_json_path = results_root / "aggregate" / "ieee_aggregate_summary.json"
    aggregate_summary = _load_json(aggregate_json_path) or {}
    per_seed_metrics = _load_per_seed_metrics(results_root, chunk_size_value)
    seed_runs = _seed_entries(results_root, per_seed_metrics)
    completed_trials = sum(1 for row in seed_runs if row.get("is_complete"))

    seed_domain_metric_rows: List[Dict[str, Any]] = []
    training_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []

    for seed, domain, mode_root, run_dir, run_config, best_scores, inference_manifest in _iter_seed_domain_runs(results_root):
        source_path = str(mode_root)
        seed_domain_metric_rows.extend(
            _flatten_seed_domain_metrics(
                seed=seed,
                domain=domain,
                source_path=source_path,
                run_config=run_config,
                best_scores=best_scores,
                inference_manifest=inference_manifest,
            )
        )
        training_rows.extend(
            _collect_training_rows(
                seed=seed,
                domain=domain,
                run_dir=run_dir,
                run_config=run_config,
                best_scores=best_scores,
                inference_manifest=inference_manifest,
            )
        )
        prediction_rows.extend(
            _collect_prediction_rows(
                seed=seed,
                domain=domain,
                run_dir=run_dir,
                run_config=run_config,
                inference_manifest=inference_manifest,
            )
        )

    prepared_dataset_rows = _collect_prepared_dataset_rows(results_root)

    snapshot: Dict[str, Any] = {
        "results_root": str(results_root),
        "aggregate_files": aggregate_file_status,
        "aggregate_summary": aggregate_summary,
        "seed_runs": seed_runs,
        "seed_domain_metrics": seed_domain_metric_rows,
        "training_runs": training_rows,
        "prediction_summaries": prediction_rows,
        "prepared_dataset_summaries": prepared_dataset_rows,
        "summary": {
            "total_trials": len(CANONICAL_SEEDS),
            "completed_trials": completed_trials,
            "all_complete": completed_trials == len(CANONICAL_SEEDS),
            "domains": list(CANONICAL_DOMAINS),
            "seed_domain_metric_rows": len(seed_domain_metric_rows),
            "training_rows": len(training_rows),
            "prediction_summary_rows": len(prediction_rows),
            "prepared_dataset_rows": len(prepared_dataset_rows),
        },
    }
    snapshot["quality_gate"] = _quality_gate(snapshot, min_f1_value)
    return snapshot


async def _executemany_chunked(
    conn: asyncpg.Connection,
    sql: str,
    rows: Sequence[Tuple[Any, ...]],
    chunk_size: int = 500,
) -> None:
    if not rows:
        return
    for start in range(0, len(rows), chunk_size):
        end = min(start + chunk_size, len(rows))
        await conn.executemany(sql, rows[start:end])


async def import_snapshot_to_db(
    conn: Optional[asyncpg.Connection],
    snapshot: Mapping[str, Any],
    *,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Import post-run snapshot rows into Postgres; dry-run performs zero writes."""
    summary = dict(snapshot.get("summary", {}))
    summary["dry_run"] = bool(dry_run)
    if dry_run:
        return {
            "status": "dry_run",
            "summary": summary,
            "quality_gate": snapshot.get("quality_gate", {}),
            "seed_rows": len(snapshot.get("seed_runs", [])),
            "aggregate_metric_rows": len(
                _flatten_aggregate_metrics(
                    snapshot.get("aggregate_summary", {})
                    if isinstance(snapshot.get("aggregate_summary"), Mapping)
                    else {}
                )
            ),
            "seed_domain_metric_rows": len(snapshot.get("seed_domain_metrics", [])),
            "training_rows": len(snapshot.get("training_runs", [])),
            "prediction_summary_rows": len(snapshot.get("prediction_summaries", [])),
            "prepared_dataset_rows": len(snapshot.get("prepared_dataset_summaries", [])),
        }
    if conn is None:
        raise RuntimeError("Database connection is required when dry_run is False.")

    import_run_id: Optional[int] = None
    source_root = str(snapshot.get("results_root", ""))
    seed_rows = list(snapshot.get("seed_runs", []))
    aggregate_summary = snapshot.get("aggregate_summary", {})
    aggregate_metric_rows = _flatten_aggregate_metrics(aggregate_summary if isinstance(aggregate_summary, Mapping) else {})
    seed_domain_metric_rows = list(snapshot.get("seed_domain_metrics", []))
    training_rows = list(snapshot.get("training_runs", []))
    prediction_rows = list(snapshot.get("prediction_summaries", []))
    prepared_rows = list(snapshot.get("prepared_dataset_summaries", []))

    async with conn.transaction():
        import_run_id = await conn.fetchval(
            """
            INSERT INTO ieee_import_runs (source_root, dry_run, status, summary)
            VALUES ($1, FALSE, 'running', $2::jsonb)
            RETURNING id
            """,
            source_root,
            json.dumps(summary),
        )

        await _executemany_chunked(
            conn,
            """
            INSERT INTO ieee_seed_runs (
                seed, status, started_at, completed_at, manifest_path,
                metrics, model_artifacts, import_batch_id, imported_at
            )
            VALUES (
                $1, $2, $3::timestamptz, $4::timestamptz, $5,
                $6::jsonb, $7::jsonb, $8, NOW()
            )
            ON CONFLICT (seed) DO UPDATE SET
                status = EXCLUDED.status,
                started_at = EXCLUDED.started_at,
                completed_at = EXCLUDED.completed_at,
                manifest_path = EXCLUDED.manifest_path,
                metrics = EXCLUDED.metrics,
                model_artifacts = EXCLUDED.model_artifacts,
                import_batch_id = EXCLUDED.import_batch_id,
                imported_at = NOW()
            """,
            [
                (
                    row.get("seed"),
                    row.get("status"),
                    _to_datetime(row.get("started_at")),
                    _to_datetime(row.get("completed_at")),
                    row.get("manifest_path"),
                    json.dumps(row.get("metrics", {})),
                    json.dumps(row.get("model_artifacts", {})),
                    import_run_id,
                )
                for row in seed_rows
            ],
        )

        source_file = str(Path(source_root) / "aggregate" / "ieee_aggregate_summary.json")
        await _executemany_chunked(
            conn,
            """
            INSERT INTO ieee_aggregate_metrics (
                scope, metric_name, mean_value, std_value, sample_size,
                source_file, import_batch_id, imported_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            ON CONFLICT (scope, metric_name, source_file) DO UPDATE SET
                mean_value = EXCLUDED.mean_value,
                std_value = EXCLUDED.std_value,
                sample_size = EXCLUDED.sample_size,
                import_batch_id = EXCLUDED.import_batch_id,
                imported_at = NOW()
            """,
            [
                (
                    metric_row["scope"],
                    metric_row["metric_name"],
                    metric_row["mean_value"],
                    metric_row["std_value"],
                    metric_row["sample_size"],
                    source_file,
                    import_run_id,
                )
                for metric_row in aggregate_metric_rows
            ],
        )

        await _executemany_chunked(
            conn,
            """
            INSERT INTO ieee_seed_domain_metrics (
                seed, domain, metric_scope, model_name, metric_name,
                metric_value, source_path, import_batch_id, imported_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
            ON CONFLICT (seed, domain, metric_scope, model_name, metric_name) DO UPDATE SET
                metric_value = EXCLUDED.metric_value,
                source_path = EXCLUDED.source_path,
                import_batch_id = EXCLUDED.import_batch_id,
                imported_at = NOW()
            """,
            [
                (
                    row.get("seed"),
                    row.get("domain"),
                    row.get("metric_scope"),
                    row.get("model_name"),
                    row.get("metric_name"),
                    row.get("metric_value"),
                    row.get("source_path"),
                    import_run_id,
                )
                for row in seed_domain_metric_rows
            ],
        )

        await _executemany_chunked(
            conn,
            """
            INSERT INTO ieee_training_runs (
                seed, domain, run_number, run_dir, started_at, elapsed_s, note,
                runtime_controls, hardware_profile, training_state, config,
                import_batch_id, imported_at
            )
            VALUES (
                $1, $2, $3, $4, $5::timestamptz, $6, $7,
                $8::jsonb, $9::jsonb, $10::jsonb, $11::jsonb,
                $12, NOW()
            )
            ON CONFLICT (seed, domain, run_number) DO UPDATE SET
                run_dir = EXCLUDED.run_dir,
                started_at = EXCLUDED.started_at,
                elapsed_s = EXCLUDED.elapsed_s,
                note = EXCLUDED.note,
                runtime_controls = EXCLUDED.runtime_controls,
                hardware_profile = EXCLUDED.hardware_profile,
                training_state = EXCLUDED.training_state,
                config = EXCLUDED.config,
                import_batch_id = EXCLUDED.import_batch_id,
                imported_at = NOW()
            """,
            [
                (
                    row.get("seed"),
                    row.get("domain"),
                    row.get("run_number"),
                    row.get("run_dir"),
                    _to_datetime(row.get("started_at")),
                    row.get("elapsed_s"),
                    row.get("note"),
                    json.dumps(row.get("runtime_controls", {})),
                    json.dumps(row.get("hardware_profile", {})),
                    json.dumps(row.get("training_state", {})),
                    json.dumps(row.get("config", {})),
                    import_run_id,
                )
                for row in training_rows
            ],
        )

        await _executemany_chunked(
            conn,
            """
            INSERT INTO ieee_prediction_summaries (
                seed, domain, split_name, model_name, total_samples, anomaly_count,
                anomaly_rate, threshold, mean_score, source_path, import_batch_id, imported_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
            ON CONFLICT (seed, domain, split_name, model_name) DO UPDATE SET
                total_samples = EXCLUDED.total_samples,
                anomaly_count = EXCLUDED.anomaly_count,
                anomaly_rate = EXCLUDED.anomaly_rate,
                threshold = EXCLUDED.threshold,
                mean_score = EXCLUDED.mean_score,
                source_path = EXCLUDED.source_path,
                import_batch_id = EXCLUDED.import_batch_id,
                imported_at = NOW()
            """,
            [
                (
                    row.get("seed"),
                    row.get("domain"),
                    row.get("split_name"),
                    row.get("model_name"),
                    row.get("total_samples"),
                    row.get("anomaly_count"),
                    row.get("anomaly_rate"),
                    row.get("threshold"),
                    row.get("mean_score"),
                    row.get("source_path"),
                    import_run_id,
                )
                for row in prediction_rows
            ],
        )

        await _executemany_chunked(
            conn,
            """
            INSERT INTO ieee_prepared_dataset_summaries (
                seed, domain, dataset_name, file_path, file_name, file_ext,
                size_bytes, row_count, column_count, parse_status, schema_preview,
                import_batch_id, imported_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12, NOW())
            ON CONFLICT (seed, domain, file_path) DO UPDATE SET
                dataset_name = EXCLUDED.dataset_name,
                file_name = EXCLUDED.file_name,
                file_ext = EXCLUDED.file_ext,
                size_bytes = EXCLUDED.size_bytes,
                row_count = EXCLUDED.row_count,
                column_count = EXCLUDED.column_count,
                parse_status = EXCLUDED.parse_status,
                schema_preview = EXCLUDED.schema_preview,
                import_batch_id = EXCLUDED.import_batch_id,
                imported_at = NOW()
            """,
            [
                (
                    row.get("seed"),
                    row.get("domain"),
                    row.get("dataset_name"),
                    row.get("file_path"),
                    row.get("file_name"),
                    row.get("file_ext"),
                    row.get("size_bytes"),
                    row.get("row_count"),
                    row.get("column_count"),
                    row.get("parse_status"),
                    json.dumps(row.get("schema_preview", [])),
                    import_run_id,
                )
                for row in prepared_rows
            ],
        )

        final_summary = {
            **summary,
            "seed_rows": len(seed_rows),
            "aggregate_metric_rows": len(aggregate_metric_rows),
            "seed_domain_metric_rows": len(seed_domain_metric_rows),
            "training_rows": len(training_rows),
            "prediction_summary_rows": len(prediction_rows),
            "prepared_dataset_rows": len(prepared_rows),
        }
        await conn.execute(
            """
            UPDATE ieee_import_runs
            SET status = 'complete', summary = $2::jsonb
            WHERE id = $1
            """,
            import_run_id,
            json.dumps(final_summary),
        )

    return {
        "status": "imported",
        "import_run_id": import_run_id,
        "summary": {
            **summary,
            "seed_rows": len(seed_rows),
            "aggregate_metric_rows": len(aggregate_metric_rows),
            "seed_domain_metric_rows": len(seed_domain_metric_rows),
            "training_rows": len(training_rows),
            "prediction_summary_rows": len(prediction_rows),
            "prepared_dataset_rows": len(prepared_rows),
        },
        "quality_gate": snapshot.get("quality_gate", {}),
    }
