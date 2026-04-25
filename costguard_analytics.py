#!/usr/bin/env python3
"""Unified CostGuard IEEE analytics and figure engine."""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from costguard_runtime import (
    StageTimer,
    atomic_write_file,
    atomic_write_json,
    atomic_write_text,
    configure_console_logger,
    configure_warning_filters,
    format_duration_s,
    install_global_exception_hooks,
    log_event,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    np = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]


DOMAINS: List[str] = ["synthetic", "real", "bitbrains"]
MODELS: List[str] = ["lstm", "gat", "ens"]
DEFAULT_TRIALS_SUBDIR = "trials"
MODEL_JSON_KEYS: Dict[str, str] = {
    "lstm": "best_lstm",
    "gat": "best_gat",
    "ens": "best_ens",
}
RESULT_KEYS: List[str] = [
    "f1_at_opt",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "threshold",
]
DOMAIN_LABELS: Dict[str, str] = {
    "synthetic": "Synthetic (D0)",
    "real": "TravisTorrent (L1)",
    "bitbrains": "BitBrains (L2)",
}
MODEL_LABELS: Dict[str, str] = {"lstm": "LSTM", "gat": "GAT", "ens": "ENS"}
PALETTE: Dict[str, str] = {
    "lstm": "#0072B2",
    "gat": "#D55E00",
    "ens": "#009E73",
    "bwt": "#4C4C4C",
}
VECTOR_FORMATS: Tuple[str, ...] = ("pdf", "eps")
logger = configure_console_logger("costguard.analytics")
configure_warning_filters()
install_global_exception_hooks(logger)


def _event(*tags: object, level: int = logging.INFO, message: Optional[str] = None, **fields: object) -> None:
    log_event(logger, *tags, level=level, message=message, **fields)


@dataclass
class SeedSummary:
    seed: int
    completed: bool
    domains: Dict[str, Dict[str, float]]
    bwt: Optional[float]


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Skipping unreadable JSON file %s: %s", path, exc)
        return None


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    cleaned = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not cleaned:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    if len(cleaned) == 1:
        return {"mean": float(cleaned[0]), "std": 0.0, "n": 1}
    return {"mean": float(mean(cleaned)), "std": float(stdev(cleaned)), "n": len(cleaned)}


def compute_bwt_from_matrix(matrix: Dict) -> Optional[float]:
    if not matrix:
        return None
    try:
        r00 = float(matrix["after_D0"]["D0"])
        r11 = float(matrix["after_L1"]["L1"])
        r02 = float(matrix["after_L2"]["D0"])
        r12 = float(matrix["after_L2"]["L1"])
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("BWT matrix is incomplete or invalid: %s", exc)
        return None
    bwt = float(0.5 * ((r02 - r00) + (r12 - r11)))
    _event("AGGREGATE", "BWT", r00=r00, r11=r11, r02=r02, r12=r12, bwt=bwt)
    return bwt


def _extract_model_scores(best_scores: Dict, model: str) -> Dict[str, float]:
    payload = best_scores.get(MODEL_JSON_KEYS[model], {}) if best_scores else {}
    scores: Dict[str, float] = {}
    for key in RESULT_KEYS:
        raw = payload.get(key)
        if raw is None and key == "threshold":
            raw = payload.get("opt_threshold")
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            logger.warning("Skipping invalid %s metric for %s: %r", key, model, raw)
            continue
        if math.isfinite(value):
            scores[key] = value
    return scores


def _sorted_seed_dirs(root: Path) -> List[Path]:
    return sorted(
        [path for path in root.glob("seed_*") if path.is_dir()],
        key=lambda path: int(path.name.split("_")[-1]),
    )


def resolve_seed_root(results_root: Path, trials_subdir: str = DEFAULT_TRIALS_SUBDIR) -> Path:
    direct_seed_dirs = _sorted_seed_dirs(results_root)
    if direct_seed_dirs:
        return results_root
    nested_root = results_root / trials_subdir
    nested_seed_dirs = _sorted_seed_dirs(nested_root)
    if nested_seed_dirs:
        return nested_root
    return results_root


def load_seed_summary(seed_dir: Path) -> SeedSummary:
    seed = int(seed_dir.name.split("_")[-1])
    completed = (seed_dir / "trial_complete.json").exists()
    domains: Dict[str, Dict[str, float]] = {}
    for domain in DOMAINS:
        best_scores = _load_json(seed_dir / domain / "best_scores.json") or {}
        domains[domain] = {}
        for model in MODELS:
            model_scores = _extract_model_scores(best_scores, model)
            for key, value in model_scores.items():
                domains[domain][f"{model}_{key}"] = value
    bwt_matrix = _load_json(seed_dir / "bwt_matrix.json") or {}
    return SeedSummary(seed=seed, completed=completed, domains=domains, bwt=compute_bwt_from_matrix(bwt_matrix))


def collect_summaries(results_root: Path, trials_subdir: str = DEFAULT_TRIALS_SUBDIR) -> List[SeedSummary]:
    seed_root = resolve_seed_root(results_root, trials_subdir=trials_subdir)
    seed_dirs = _sorted_seed_dirs(seed_root)
    return [load_seed_summary(seed_dir) for seed_dir in seed_dirs]


def build_flat_rows(summaries: Sequence[SeedSummary]) -> List[Dict]:
    rows: List[Dict] = []
    for summary in summaries:
        for domain in DOMAINS:
            row = {"seed": summary.seed, "domain": domain, "completed": summary.completed}
            row.update(summary.domains.get(domain, {}))
            row["bwt"] = summary.bwt if summary.bwt is not None else ""
            rows.append(row)
    return rows


def build_seed_overview_rows(summaries: Sequence[SeedSummary]) -> List[Dict]:
    rows: List[Dict] = []
    for summary in summaries:
        row: Dict[str, object] = {
            "seed": summary.seed,
            "completed": summary.completed,
            "bwt": summary.bwt if summary.bwt is not None else "",
        }
        for domain in DOMAINS:
            row[f"{domain}_ens_f1_at_opt"] = summary.domains.get(domain, {}).get("ens_f1_at_opt", "")
            row[f"{domain}_ens_roc_auc"] = summary.domains.get(domain, {}).get("ens_roc_auc", "")
            row[f"{domain}_ens_pr_auc"] = summary.domains.get(domain, {}).get("ens_pr_auc", "")
        rows.append(row)
    return rows


def build_aggregate_summary(summaries: Sequence[SeedSummary]) -> Dict:
    aggregate: Dict[str, Dict] = {
        "domains": {},
        "bwt": _mean_std([summary.bwt for summary in summaries if summary.bwt is not None]),
        "completed_trials": sum(1 for summary in summaries if summary.completed),
        "total_trials": len(summaries),
        "is_final": bool(summaries) and all(summary.completed for summary in summaries),
        "lifelong_order": ["D0=synthetic", "L1=travistorrent", "L2=bitbrains"],
    }
    for domain in DOMAINS:
        domain_payload: Dict[str, Dict] = {}
        for model in MODELS:
            domain_payload[model] = {
                metric: _mean_std(
                    [
                        summary.domains.get(domain, {}).get(f"{model}_{metric}")
                        for summary in summaries
                        if f"{model}_{metric}" in summary.domains.get(domain, {})
                    ]
                )
                for metric in RESULT_KEYS
            }
        aggregate["domains"][domain] = domain_payload
    return aggregate


def write_csv(rows: Sequence[Dict], out_path: Path) -> None:
    def _writer(tmp_path: Path) -> None:
        pd.DataFrame(rows).to_csv(tmp_path, index=False)

    atomic_write_file(out_path, _writer)


def write_latex(summary: Dict, out_path: Path) -> None:
    table_rows: List[Dict[str, object]] = []
    for domain in DOMAINS:
        payload = summary["domains"].get(domain, {})
        for model in MODELS:
            metrics = payload.get(model, {})
            table_rows.append(
                {
                    "Domain": DOMAIN_LABELS[domain],
                    "Model": MODEL_LABELS[model],
                    "F1@opt": metrics.get("f1_at_opt", {}).get("mean", 0.0),
                    "ROC-AUC": metrics.get("roc_auc", {}).get("mean", 0.0),
                    "PR-AUC": metrics.get("pr_auc", {}).get("mean", 0.0),
                    "Precision": metrics.get("precision", {}).get("mean", 0.0),
                    "Recall": metrics.get("recall", {}).get("mean", 0.0),
                    "Threshold": metrics.get("threshold", {}).get("mean", 0.0),
                }
            )
    df = pd.DataFrame(table_rows)
    latex = df.to_latex(
        index=False,
        float_format="%.4f",
        caption="CostGuard PADE v17.0 3-domain aggregate results.",
        label="tab:costguard_v17_aggregate",
    )
    atomic_write_text(out_path, latex, encoding="utf-8")


def write_json(summary: Dict, out_path: Path) -> None:
    atomic_write_json(out_path, summary, indent=2)


def _configure_plot_style() -> None:
    if not _HAS_MPL:
        return
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError as exc:
        logger.debug("Plot style fallback engaged: %s", exc)
        plt.style.use("default")
    matplotlib.rcParams["font.family"] = ["Times New Roman", "Times", "DejaVu Serif", "serif"]
    matplotlib.rcParams["axes.titlesize"] = 12
    matplotlib.rcParams["axes.labelsize"] = 11
    matplotlib.rcParams["legend.fontsize"] = 10
    matplotlib.rcParams["figure.titlesize"] = 13


def _save_figure(fig, out_dir: Path, stem: str, dpi: int, formats: Sequence[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        def _writer(tmp_path: Path, *, _fmt: str = fmt) -> None:
            fig.savefig(tmp_path, format=_fmt, dpi=dpi, bbox_inches="tight")

        atomic_write_file(path, _writer)
        _event("FIGURE", "WRITE", path=path.resolve(), format=fmt, dpi=dpi)
    plt.close(fig)


def _domain_model_matrix(summary: Dict, metric: str) -> "np.ndarray":
    matrix = []
    for domain in DOMAINS:
        row = []
        for model in MODELS:
            row.append(summary["domains"][domain][model][metric]["mean"])
        matrix.append(row)
    return np.asarray(matrix, dtype=float)


def plot_metric_bars(summary: Dict, out_dir: Path, dpi: int, formats: Sequence[str]) -> None:
    if not _HAS_MPL:
        return
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(DOMAINS))
    width = 0.22
    for offset, model in zip((-width, 0.0, width), MODELS):
        values = [summary["domains"][domain][model]["f1_at_opt"]["mean"] for domain in DOMAINS]
        ax.bar(x + offset, values, width=width, label=MODEL_LABELS[model], color=PALETTE[model])
    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1@opt")
    ax.set_title("CostGuard v17.0 F1@opt by Domain and Model")
    ax.legend(frameon=False)
    _save_figure(fig, out_dir, "ieee_f1_comparison", dpi, formats)


def plot_metric_heatmap(summary: Dict, out_dir: Path, dpi: int, formats: Sequence[str]) -> None:
    if not _HAS_MPL:
        return
    _configure_plot_style()
    matrix = _domain_model_matrix(summary, "roc_auc")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    heatmap = ax.imshow(matrix, cmap="cividis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_yticks(np.arange(len(DOMAINS)))
    ax.set_yticklabels([DOMAIN_LABELS[d] for d in DOMAINS])
    ax.set_title("ROC-AUC Heatmap")
    for i in range(len(DOMAINS)):
        for j in range(len(MODELS)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    _save_figure(fig, out_dir, "ieee_roc_auc_heatmap", dpi, formats)


def plot_bwt_distribution(summaries: Sequence[SeedSummary], out_dir: Path, dpi: int, formats: Sequence[str]) -> None:
    if not _HAS_MPL:
        return
    _configure_plot_style()
    valid = [(summary.seed, summary.bwt) for summary in summaries if summary.bwt is not None]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if valid:
        seeds = [seed for seed, _ in valid]
        values = [float(value) for _, value in valid]
        colors = [PALETTE["bwt"] if -0.01 <= value <= 0.01 else "#CC3311" for value in values]
        ax.bar(seeds, values, color=colors)
        ax.axhline(0.0, color="#000000", linewidth=1.0)
        ax.axhline(0.01, color="#666666", linewidth=0.8, linestyle="--")
        ax.axhline(-0.01, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_title("Backward Transfer (T=3)")
    ax.set_xlabel("Seed")
    ax.set_ylabel("BWT")
    _save_figure(fig, out_dir, "ieee_bwt_summary", dpi, formats)


def generate_figures(
    results_root: Path,
    aggregate_summary: Optional[Dict] = None,
    out_dir: Optional[Path] = None,
    dpi: int = 1000,
    formats: Optional[Sequence[str]] = None,
    trials_subdir: str = DEFAULT_TRIALS_SUBDIR,
) -> Dict[str, Path]:
    timer = StageTimer()
    summaries = collect_summaries(results_root, trials_subdir=trials_subdir)
    summary = aggregate_summary or build_aggregate_summary(summaries)
    figure_dir = out_dir or (results_root / "paper_figures")
    figure_formats = list(formats or VECTOR_FORMATS)
    if "pdf" not in figure_formats:
        figure_formats.insert(0, "pdf")
    _event(
        "FIGURE",
        "START",
        results_root=results_root.resolve(),
        out_dir=figure_dir.resolve(),
        formats=",".join(figure_formats),
        trials=len(summaries),
    )
    written = {
        "figure_dir": figure_dir,
    }
    if not _HAS_MPL:
        _event("FIGURE", "END", out_dir=figure_dir.resolve(), duration=format_duration_s(timer.elapsed_s), available=False)
        return written
    plot_metric_bars(summary, figure_dir, dpi=dpi, formats=figure_formats)
    plot_metric_heatmap(summary, figure_dir, dpi=dpi, formats=figure_formats)
    plot_bwt_distribution(summaries, figure_dir, dpi=dpi, formats=figure_formats)
    _event("FIGURE", "END", out_dir=figure_dir.resolve(), duration=format_duration_s(timer.elapsed_s), available=True)
    return written


def write_aggregate_bundle(
    results_root: Path,
    write_csv_enabled: bool = False,
    write_latex_enabled: bool = False,
    aggregate_dir: Optional[Path] = None,
    trials_subdir: str = DEFAULT_TRIALS_SUBDIR,
) -> Dict:
    timer = StageTimer()
    out_dir = aggregate_dir or (results_root / "aggregate")
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = collect_summaries(results_root, trials_subdir=trials_subdir)
    _event(
        "AGGREGATE",
        "START",
        results_root=results_root.resolve(),
        seed_root=resolve_seed_root(results_root, trials_subdir=trials_subdir).resolve(),
        out_dir=out_dir.resolve(),
        trials=len(summaries),
        write_csv=write_csv_enabled,
        write_latex=write_latex_enabled,
    )
    rows = build_flat_rows(summaries)
    seed_rows = build_seed_overview_rows(summaries)
    summary = build_aggregate_summary(summaries)
    write_json(summary, out_dir / "ieee_aggregate_summary.json")
    atomic_write_json(out_dir / "ieee_per_seed_summary.json", seed_rows, indent=2)
    if write_csv_enabled:
        write_csv(rows, out_dir / "ieee_aggregate_summary.csv")
        write_csv(seed_rows, out_dir / "ieee_per_seed_summary.csv")
    if write_latex_enabled:
        write_latex(summary, out_dir / "ieee_aggregate_summary.tex")
    _event(
        "AGGREGATE",
        "END",
        out_dir=out_dir.resolve(),
        summary_json=(out_dir / "ieee_aggregate_summary.json").resolve(),
        per_seed_json=(out_dir / "ieee_per_seed_summary.json").resolve(),
        csv=(out_dir / "ieee_aggregate_summary.csv").resolve() if write_csv_enabled else None,
        latex=(out_dir / "ieee_aggregate_summary.tex").resolve() if write_latex_enabled else None,
        duration=format_duration_s(timer.elapsed_s),
    )
    return summary


def aggregate_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate CostGuard v17.0 3-domain results.")
    parser.add_argument("--results-root", default="results", help="Root directory containing seed_* runs or a trials/ subdirectory.")
    parser.add_argument("--aggregate-dir", default=None, help="Optional output directory for aggregate artifacts.")
    parser.add_argument("--trials-subdir", default=DEFAULT_TRIALS_SUBDIR, help="Nested subdirectory to scan when results-root contains no direct seed_* folders.")
    parser.add_argument("--csv", action="store_true", help="Write CSV export.")
    parser.add_argument("--latex", action="store_true", help="Write LaTeX export.")
    args = parser.parse_args(argv)

    results_root = Path(args.results_root)
    aggregate_dir = Path(args.aggregate_dir) if args.aggregate_dir else None
    summary = write_aggregate_bundle(
        results_root,
        write_csv_enabled=args.csv,
        write_latex_enabled=args.latex,
        aggregate_dir=aggregate_dir,
        trials_subdir=args.trials_subdir,
    )
    bwt = summary["bwt"]
    print(
        f"Aggregate complete: trials={summary['completed_trials']}/{summary['total_trials']} "
        f"BWT_mean={bwt['mean']:.4f} BWT_std={bwt['std']:.4f}"
    )
    return 0


def figures_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate CostGuard IEEE analytics figures.")
    parser.add_argument("--results-dir", default="results", help="Root directory containing seed_* runs or a trials/ subdirectory.")
    parser.add_argument("--aggregate-json", default=None, help="Optional precomputed aggregate summary JSON.")
    parser.add_argument("--out-dir", default=None, help="Optional figure output directory.")
    parser.add_argument("--trials-subdir", default=DEFAULT_TRIALS_SUBDIR, help="Nested subdirectory to scan when results-dir contains no direct seed_* folders.")
    parser.add_argument("--ml-ready-dir", default=None, help="Compatibility no-op flag.")
    parser.add_argument("--force", action="store_true", help="Compatibility no-op flag.")
    parser.add_argument("--dpi", type=int, default=1000, help="Figure DPI for raster compatibility.")
    parser.add_argument("--no-local", action="store_true", help="Compatibility no-op flag.")
    parser.add_argument("--formats", nargs="+", default=list(VECTOR_FORMATS), help="Output formats, e.g. pdf eps png.")
    args = parser.parse_args(argv)

    results_root = Path(args.results_dir)
    aggregate_summary = _load_json(Path(args.aggregate_json)) if args.aggregate_json else None
    out_dir = Path(args.out_dir) if args.out_dir else None
    written = generate_figures(
        results_root,
        aggregate_summary=aggregate_summary,
        out_dir=out_dir,
        dpi=args.dpi,
        formats=args.formats,
        trials_subdir=args.trials_subdir,
    )
    print(f"Figure generation complete: {written['figure_dir']}")
    return 0


__all__ = [
    "DOMAINS",
    "MODELS",
    "RESULT_KEYS",
    "SeedSummary",
    "aggregate_cli",
    "build_aggregate_summary",
    "build_flat_rows",
    "build_seed_overview_rows",
    "collect_summaries",
    "compute_bwt_from_matrix",
    "figures_cli",
    "generate_figures",
    "load_seed_summary",
    "resolve_seed_root",
    "write_aggregate_bundle",
]
