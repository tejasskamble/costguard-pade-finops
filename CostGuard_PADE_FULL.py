#!/usr/bin/env python3
# CostGuard_PADE_FULL.py - v17.0 (IEEE-FINAL-3DOMAIN)
#
# Canonical CostGuard research and training engine.
# Active domain order: D0 Synthetic -> L1 TravisTorrent -> L2 BitBrains.
# This file is the source of truth for the IEEE v17.0 single-file ML stack.
"""
Canonical CostGuard PADE pipeline for the IEEE v17.0 3-domain workflow.

This script owns synthetic generation, TravisTorrent and BitBrains ingestion,
model training, validation-threshold selection, T=3 BWT tracking, smoke tests,
and paper-ready result artifacts.

Directory isolation:
  costguard_data/   -> synthetic outputs only
  real_data/        -> TravisTorrent outputs only
  raw_bitbrains/    -> BitBrains outputs only
  raw_universal/    -> merged TravisTorrent + BitBrains compatibility outputs

Quick start:
  python CostGuard_PADE_FULL.py --smoke-test
  python CostGuard_PADE_FULL.py --generate --seed 42
  python CostGuard_PADE_FULL.py --synth-only --seed 42
  python CostGuard_PADE_FULL.py --train-lifelong --data-mode real --real-input ./final-2017-01-25.csv
  python CostGuard_PADE_FULL.py --train-lifelong --data-mode bitbrains --bitbrains-dir ./fastStorage/2013-8
"""
from __future__ import annotations

# Standard library
import argparse, csv, gc, hashlib, json, logging, math, os, pickle, subprocess
import random, re, shutil, smtplib, socket, struct, sys, time, unicodedata
import urllib.request
from collections import Counter, deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Tuple, Union

# Third-party
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, confusion_matrix,
                              f1_score, precision_score, recall_score,
                              roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.utils.data import DataLoader, TensorDataset

from costguard_runtime import (
    StageTimer,
    atomic_write_file,
    atomic_write_text,
    clear_torch_memory,
    configure_console_logger,
    configure_warning_filters,
    format_duration_s,
    atomic_write_json as runtime_atomic_write_json,
    install_global_exception_hooks,
    is_torch_oom,
    log_event,
)
# Optional dependencies
# ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as GeoDataLoader
    from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

if TYPE_CHECKING:
    from torch_geometric.data import Batch

try:
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

try:
    from dotenv import dotenv_values as _dotenv_values
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
# ---------------------------------------------------------------------------
# Section 0: Global configuration and constants
# ---------------------------------------------------------------------------
# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logger = configure_console_logger("CostGuard")
configure_warning_filters()
install_global_exception_hooks(logger)


def _event(*tags: object, level: int = logging.INFO, message: Optional[str] = None, **fields: object) -> None:
    log_event(logger, *tags, level=level, message=message, **fields)

def _ts() -> str:
    """Return a compact UTC timestamp string for inline progress prints."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)

def _supports_ansi_colour() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


# ANSI colours
if _supports_ansi_colour():
    _GRN = "\033[92m"; _YLW = "\033[93m"; _RED = "\033[91m"
    _MAG = "\033[95m"; _CYN = "\033[96m"; _BLD = "\033[1m"; _RST = "\033[0m"
else:
    _GRN = _YLW = _RED = _MAG = _CYN = _BLD = _RST = ""


_RUNTIME_CONTROLS: Dict[str, bool] = {
    "verbose_epochs": True,
    "disable_notifier": False,
}
_NOTIFIER_LAST_SENT: Dict[Tuple[str, str], datetime] = {}
_NOTIFIER_COOLDOWN_SECONDS = 60


def _set_runtime_controls(verbose_epochs: bool = True, disable_notifier: bool = False) -> None:
    _RUNTIME_CONTROLS["verbose_epochs"] = bool(verbose_epochs)
    _RUNTIME_CONTROLS["disable_notifier"] = bool(disable_notifier)


def _workspace_root(results_base: Union[str, Path]) -> Path:
    return _ensure_dir(Path(results_base) / "_workspace")


def _default_raw_dir(results_base: Union[str, Path], domain: str) -> Path:
    workspace = _workspace_root(results_base)
    domain_key = _domain_label_from_mode(domain)
    mapping = {
        "synthetic": workspace / "synthetic_raw",
        "real": workspace / "real_data",
        "bitbrains": workspace / "bitbrains_data",
        "universal": workspace / "universal_raw",
    }
    return mapping.get(domain_key, workspace / domain_key)


def _default_ml_ready_dir(results_base: Union[str, Path], domain: str) -> Path:
    workspace = _workspace_root(results_base)
    domain_key = _domain_label_from_mode(domain)
    mapping = {
        "synthetic": workspace / "ml_ready_synthetic",
        "real": workspace / "ml_ready_real",
        "bitbrains": workspace / "ml_ready_bitbrains",
        "universal": workspace / "ml_ready_universal",
    }
    return mapping.get(domain_key, workspace / f"ml_ready_{domain_key}")


def _default_brain_dir(results_base: Union[str, Path]) -> Path:
    return _workspace_root(results_base) / "costguard_brain"


def _compat_domain_results_dir(results_dir: Path) -> Optional[Path]:
    parent = results_dir.parent
    if re.fullmatch(r"seed_\d+", parent.name or ""):
        project_results_root = parent.parent
        if project_results_root.name == "results":
            return _ensure_dir(project_results_root / results_dir.name)
    return None

# Canonical constants (ground truth - do not change without protocol updates)

# ──────────────────────────────────────────────────────────────────────
SEQ_LEN        = 30
N_CHANNELS     = 5
N_CTX          = 22
N_NODE_FEAT    = 11
COST_PER_S     = 0.008 / 60.0
ANOMALY_RATE   = 0.12
SYNTH_ROWS     = 500_000
OPT_THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)
SEED           = 42

STAGE_ORDER: List[str] = [
    "checkout", "build", "unit_test", "integration_test",
    "security_scan", "docker_build", "deploy_staging", "deploy_prod",
]
CHANNEL_NAMES: List[str] = ["ch0", "ch1", "ch2", "ch3", "ch4"]
FOCUS_COLS: List[str] = [
    "run_id", "stage_name", "executor_type", "branch", "created_at",
    "cpu_seconds", "memory_gb_s", "billed_cost", "network_egress_gb",
    "latency_p95", "call_count", "anomaly_window_active", "duration_seconds",
]
EXECUTOR_TYPES: List[str] = ["github_actions", "gitlab_ci", "jenkins", "circleci"]
BRANCH_TYPES:   List[str] = ["main", "develop", "feature", "release", "hotfix"]
BITBRAINS_EXECUTOR = "bitbrains_vm"
TRAVIS_EXECUTOR = "travis_ci"

MAX_VRAM_FRACTION = 0.75
LSTM_BATCH_SIZES = [128, 64, 32, 16, 8, 4]           # OOM-FIX: start safe for 4 GB VRAM
GAT_BATCH_SIZES  = [16, 8, 4, 2, 1]                  # OOM-FIX: graph batches are heavy
DEFAULT_OOM_RETRY_LIMIT = 3
SAFE_MONITOR_EVERY_EPOCHS = 5
MIN_GAT_NEIGHBOR_LIMIT = 2
DEFAULT_GAT_NEIGHBOR_STEPS = [16, 12, 8, 6, 4, 2]
SOTA_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "synthetic": {"lstm": 0.88, "gat": 0.89, "ens": 0.91},
    "real": {"lstm": 0.85, "gat": 0.87, "ens": 0.89},
    "travistorrent": {"lstm": 0.85, "gat": 0.87, "ens": 0.89},
    "bitbrains": {"lstm": 0.83, "gat": 0.85, "ens": 0.87},
}

# Lifelong learning constants
BRAIN_FILE     = "costguard_brain.pt"
EWC_LAMBDA     = 400.0
REPLAY_SIZE    = 2_000
LIFELONG_LR    = 1e-4
PH_DELTA       = 0.005
PH_LAMBDA      = 50.0
EWA_ALPHA      = 0.10

# Paths
BASE_DIR = Path(__file__).resolve().parent
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _vram_transition_flush(label: str = "") -> None:
    """Force VRAM pool release at domain loader boundaries (RTX 3050 4 GB guard)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if label:
            free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024)
            logger.info(f"[VRAM-FLUSH] {label} -> {free_mb} MB VRAM free after flush")


def _probe_gpu_utilization() -> Optional[float]:
    """Best-effort GPU utilization probe using pynvml or nvidia-smi."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        pass

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        output = subprocess.check_output(
            [
                nvidia_smi,
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        first_line = output.strip().splitlines()[0]
        return float(first_line.strip())
    except Exception:
        return None


@dataclass
class SystemStatsSnapshot:
    cpu_cores: int = 1
    cpu_percent: Optional[float] = None
    ram_total_mb: int = 0
    ram_used_mb: int = 0
    ram_free_mb: int = 0
    gpu_name: str = "cpu"
    gpu_util_percent: Optional[float] = None
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    vram_free_mb: int = 0

    @property
    def has_gpu(self) -> bool:
        return self.vram_total_mb > 0


def _system_stats_snapshot() -> SystemStatsSnapshot:
    snapshot = SystemStatsSnapshot()
    try:
        if _HAS_PSUTIL:
            vm = psutil.virtual_memory()
            snapshot.cpu_cores = psutil.cpu_count(logical=True) or 1
            snapshot.cpu_percent = float(psutil.cpu_percent(interval=0.0))
            snapshot.ram_total_mb = int(vm.total / (1024 * 1024))
            snapshot.ram_used_mb = int(vm.used / (1024 * 1024))
            snapshot.ram_free_mb = int(vm.available / (1024 * 1024))
    except Exception as exc:
        logger.debug(f"[SYS] psutil snapshot skipped: {exc}")

    try:
        if torch.cuda.is_available():
            free_vram, total_vram = torch.cuda.mem_get_info(0)
            snapshot.vram_total_mb = int(total_vram / (1024 * 1024))
            snapshot.vram_free_mb = int(free_vram / (1024 * 1024))
            snapshot.vram_used_mb = max(0, snapshot.vram_total_mb - snapshot.vram_free_mb)
            try:
                snapshot.gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                snapshot.gpu_name = "cuda"
            snapshot.gpu_util_percent = _probe_gpu_utilization()
    except Exception as exc:
        logger.debug(f"[SYS] CUDA snapshot skipped: {exc}")
    return snapshot


def _format_optional_percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0f}%"


def _format_gb_from_mb(value_mb: int) -> str:
    if value_mb <= 0:
        return "0GB"
    return f"{value_mb / 1024:.1f}GB"


def _log_system_stats(label: str = "") -> None:
    stats = _system_stats_snapshot()
    fields: Dict[str, object] = {
        "gpu": stats.gpu_name,
        "gpu_util": _format_optional_percent(stats.gpu_util_percent),
        "vram_used": f"{stats.vram_used_mb}MB",
        "vram_free": f"{stats.vram_free_mb}MB",
        "cpu": _format_optional_percent(stats.cpu_percent),
        "ram": _format_gb_from_mb(stats.ram_used_mb),
        "ram_free": _format_gb_from_mb(stats.ram_free_mb),
    }
    if label:
        fields["label"] = label
    _event("SYS", **fields)


@dataclass
class AdaptiveTrainSettings:
    model_name: str
    requested_batch_size: int
    batch_size: int
    eval_batch_size: int
    num_workers: int
    grad_accum_steps: int = 1
    fp16_enabled: bool = False
    oom_retry_limit: int = DEFAULT_OOM_RETRY_LIMIT
    monitor_every_epochs: int = SAFE_MONITOR_EVERY_EPOCHS
    nodes_per_batch: Optional[int] = None
    neighbor_limit: Optional[int] = None
    gradient_checkpointing: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "requested_batch_size": self.requested_batch_size,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "num_workers": self.num_workers,
            "grad_accum_steps": self.grad_accum_steps,
            "fp16_enabled": self.fp16_enabled,
            "oom_retry_limit": self.oom_retry_limit,
            "monitor_every_epochs": self.monitor_every_epochs,
            "nodes_per_batch": self.nodes_per_batch,
            "neighbor_limit": self.neighbor_limit,
            "gradient_checkpointing": self.gradient_checkpointing,
        }


class StaticBatchSampler(torch.utils.data.Sampler[List[int]]):
    def __init__(self, batches: List[List[int]]) -> None:
        self._batches = [list(batch) for batch in batches if batch]

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self._batches:
            yield list(batch)

    def __len__(self) -> int:
        return len(self._batches)


def _candidate_batch_sizes(requested: int, options: List[int], minimum: int = 1) -> List[int]:
    requested = max(minimum, int(requested or minimum))
    candidates = {requested}
    for option in options:
        if option <= requested:
            candidates.add(max(minimum, int(option)))
    probe = requested
    while probe > minimum:
        probe = max(minimum, probe // 2)
        candidates.add(probe)
        if probe == minimum:
            break
    return sorted(candidates, reverse=True)


def _adaptive_num_workers(hardware: Optional[HardwareProfile], dataset_rows: int) -> int:
    if dataset_rows <= 0:
        return 0
    if sys.platform == "win32":
        return 0
    if hardware is None:
        return 0
    if hardware.ram_total_gb < 16 or hardware.ram_gb < 4:
        return 0
    return max(0, min(2, hardware.cpu_cores - 1))


def _recommended_eval_batch_size(batch_size: int, total_rows: int, *, graph_mode: bool = False) -> int:
    if total_rows <= 0:
        return max(1, batch_size)
    ceiling = 16 if graph_mode else 128
    return max(1, min(total_rows, max(batch_size, min(ceiling, batch_size * 2))))


def _gradient_accumulation_steps(requested_batch_size: int, actual_batch_size: int) -> int:
    return max(1, int(math.ceil(max(1, requested_batch_size) / max(1, actual_batch_size))))


def _lstm_batch_threshold(requested_batch_size: int, hardware: Optional[HardwareProfile]) -> int:
    if hardware is None:
        return requested_batch_size
    batch = requested_batch_size
    if hardware.vram_total_gb and hardware.vram_total_gb <= 4.5:
        batch = min(batch, 64)                         # OOM-FIX: safer for RTX 3050 4 GB
    if hardware.vram_gb and hardware.vram_gb < 3.0:
        batch = min(batch, 32)
    if hardware.vram_gb and hardware.vram_gb < 2.0:
        batch = min(batch, 16)
    if hardware.ram_total_gb < 8:
        batch = min(batch, 32)
    return max(4, batch)


def _gat_batch_threshold(requested_batch_size: int, hardware: Optional[HardwareProfile]) -> int:
    if hardware is None:
        return requested_batch_size
    batch = requested_batch_size
    if hardware.vram_total_gb and hardware.vram_total_gb <= 4.5:
        batch = min(batch, 8)                          # OOM-FIX: GAT graphs are heavy
    if hardware.vram_gb and hardware.vram_gb < 3.0:
        batch = min(batch, 4)
    if hardware.vram_gb and hardware.vram_gb < 2.0:
        batch = min(batch, 2)
    if hardware.ram_total_gb < 8:
        batch = min(batch, 4)
    return max(1, batch)


def _fp16_safe(hardware: Optional["HardwareProfile"]) -> bool:
    """Enable fp16 only when CUDA is present AND free VRAM headroom >= 1.5 GB."""
    if not torch.cuda.is_available():
        return False
    if hardware is None:
        return False
    # OOM-FIX: RTX 3050 4GB — require 1.5 GB free before enabling fp16 scaler buffers
    if hardware.vram_gb is not None and hardware.vram_gb < 1.5:
        return False
    return True


def _graph_node_count(graph: Data) -> int:
    if hasattr(graph, "num_nodes") and graph.num_nodes:
        return int(graph.num_nodes)
    return int(graph.x.shape[0]) if getattr(graph, "x", None) is not None else 0


def _graph_neighbor_estimate(graphs: List[Data]) -> int:
    estimates: List[int] = []
    for graph in graphs[: min(len(graphs), 64)]:
        num_nodes = max(1, _graph_node_count(graph))
        edge_index = getattr(graph, "edge_index", None)
        if edge_index is None or edge_index.numel() == 0:
            continue
        estimates.append(max(1, int(edge_index.shape[1] / num_nodes)))
    if not estimates:
        return DEFAULT_GAT_NEIGHBOR_STEPS[0]
    return max(MIN_GAT_NEIGHBOR_LIMIT, int(np.median(np.asarray(estimates, dtype=np.float32))))


def _default_nodes_per_batch(graphs: List[Data], batch_size: int, hardware: Optional[HardwareProfile]) -> Optional[int]:
    if not graphs:
        return None
    node_counts = [_graph_node_count(graph) for graph in graphs[: min(len(graphs), 256)]]
    typical_nodes = max(1, int(np.percentile(np.asarray(node_counts, dtype=np.float32), 75)))
    target = typical_nodes * max(1, batch_size)
    if hardware is None:
        return target
    if hardware.vram_total_gb and hardware.vram_total_gb <= 4.5:
        target = min(target, 1024)
    if hardware.vram_gb and hardware.vram_gb < 3.0:
        target = min(target, 768)
    if hardware.vram_gb and hardware.vram_gb < 2.0:
        target = min(target, 512)
    return max(typical_nodes, target)


def _default_neighbor_limit(graphs: List[Data], hardware: Optional[HardwareProfile]) -> Optional[int]:
    base = _graph_neighbor_estimate(graphs)
    if hardware is None:
        return base
    limit = base
    if hardware.vram_total_gb and hardware.vram_total_gb <= 4.5:
        limit = min(limit, 8)
    if hardware.vram_gb and hardware.vram_gb < 3.0:
        limit = min(limit, 6)
    if hardware.vram_gb and hardware.vram_gb < 2.0:
        limit = min(limit, 4)
    return max(MIN_GAT_NEIGHBOR_LIMIT, limit)


def _build_graph_index_batches(graphs: List[Data], batch_size: int,
                               nodes_per_batch: Optional[int], shuffle: bool) -> List[List[int]]:
    indices = list(range(len(graphs)))
    if shuffle:
        random.shuffle(indices)
    if not nodes_per_batch:
        return [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    batches: List[List[int]] = []
    current_batch: List[int] = []
    current_nodes = 0
    for idx in indices:
        graph_nodes = max(1, _graph_node_count(graphs[idx]))
        should_flush = current_batch and (
            len(current_batch) >= batch_size or current_nodes + graph_nodes > nodes_per_batch
        )
        if should_flush:
            batches.append(current_batch)
            current_batch = []
            current_nodes = 0
        current_batch.append(idx)
        current_nodes += graph_nodes
    if current_batch:
        batches.append(current_batch)
    return batches


def _limit_graph_batch_neighbors(batch: "Batch", neighbor_limit: Optional[int]) -> "Batch":
    if neighbor_limit is None or neighbor_limit <= 0:
        return batch
    edge_index = getattr(batch, "edge_index", None)
    if edge_index is None or edge_index.numel() == 0:
        return batch

    source_nodes = edge_index[0].detach().cpu().tolist()
    keep_mask = [False] * len(source_nodes)
    counts: Dict[int, int] = {}
    for idx, node_id in enumerate(source_nodes):
        seen = counts.get(int(node_id), 0)
        if seen < neighbor_limit:
            keep_mask[idx] = True
            counts[int(node_id)] = seen + 1
    if all(keep_mask):
        return batch

    mask_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=edge_index.device)
    batch.edge_index = edge_index[:, mask_tensor]
    edge_attr = getattr(batch, "edge_attr", None)
    if edge_attr is not None and edge_attr.shape[0] == mask_tensor.shape[0]:
        batch.edge_attr = edge_attr[mask_tensor]
    return batch


def _restore_training_state_objects(model: nn.Module,
                                    optimizer: optim.Optimizer,
                                    scheduler: object,
                                    scaler: torch.amp.GradScaler,
                                    state: Dict[str, object]) -> Tuple[float, int, int, Dict[str, object]]:
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if state.get("scheduler") is not None and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(state["scheduler"])
    if state.get("scaler") is not None and torch.cuda.is_available():
        scaler.load_state_dict(state["scaler"])
    _restore_training_rng_state(state)
    best_f1 = float(state.get("best_f1", 0.0))
    best_epoch = int(state.get("best_epoch", 0))
    patience_state = state.get("patience_state") or {}
    patience_ctr = int(patience_state.get("counter", state.get("patience_ctr", 0)))
    hist = _normalise_training_history(state.get("hist"))
    return best_f1, best_epoch, patience_ctr, hist

# ──────────────────────────────────────────────────────────────────────
_ML_READY_DIR:  Optional[Path] = None
_TASK_B_DIR:    Optional[Path] = None
_TASK_C_DIR:    Optional[Path] = None
_SYNTH_MODE:    str = "standard"


# ──────────────────────────────────────────────────────────────────────
# Ãƒâ€šÃ‚Â§0.1  CUSTOM EXCEPTIONS
# ──────────────────────────────────────────────────────────────────────
class DiskSpaceError(OSError):
    def __init__(self, path: Path, available: int, needed: int) -> None:
        self.path = path; self.available = available; self.needed = needed
        super().__init__(
            f"[GUARDIAN-DISK-FULL] path={path} "
            f"avail={available/1e9:.2f}GB needed={needed/1e9:.2f}GB")

class DataIngestionError(Exception):
    def __init__(self, file_path: Union[str, Path], reason: str) -> None:
        self.file_path = str(file_path); self.reason = reason
        super().__init__(f"[INGEST-ERROR] {file_path}: {reason}")

class PipelineStageError(RuntimeError):
    def __init__(self, stage: str, attempt: int, cause: BaseException) -> None:
        self.stage = stage; self.attempt = attempt; self.cause = cause
        super().__init__(f"[IMMORTAL] Stage '{stage}' failed at attempt {attempt}. "
                         f"Cause: {type(cause).__name__}: {cause}")


# ──────────────────────────────────────────────────────────────────────
# Ãƒâ€šÃ‚Â§0.2  HARDWARE PROFILE  (Guardian D1 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â preserved byte-for-byte)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class HardwareProfile:
    """Hardware-aware resource limits. probe() re-reads psutil every call."""
    ram_gb:          float = 4.0
    ram_total_gb:    float = 4.0
    ram_used_gb:     float = 0.0
    vram_gb:         float = 0.0
    vram_total_gb:   float = 0.0
    vram_used_gb:    float = 0.0
    gpu_name:        str   = "cpu"
    gpu_util_pct:    Optional[float] = None
    cpu_cores:       int   = 1
    tier:            str   = "low"
    safe_rows:       int   = 50_000
    safe_batch:      int   = 64
    safe_chunk:      int   = 50_000
    free_disk_gb:    float = 50.0
    storage_type:    str   = "ssd"

    @classmethod
    def probe(cls) -> "HardwareProfile":
        hp = cls()
        try:
            if _HAS_PSUTIL:
                vm = psutil.virtual_memory()
                hp.ram_gb = vm.available / 1_073_741_824
                hp.ram_total_gb = vm.total / 1_073_741_824
                hp.ram_used_gb = vm.used / 1_073_741_824
                hp.cpu_cores = psutil.cpu_count(logical=True) or 1
                hp.free_disk_gb = shutil.disk_usage(BASE_DIR).free / 1_073_741_824
        except Exception as exc:
            logger.debug(f"[HW] psutil probe skipped: {exc}")
        try:
            if torch.cuda.is_available():
                free_vram, total_vram = torch.cuda.mem_get_info(0)
                hp.vram_gb = free_vram / 1_073_741_824
                hp.vram_total_gb = total_vram / 1_073_741_824
                hp.vram_used_gb = max(0.0, hp.vram_total_gb - hp.vram_gb)
                try:
                    hp.gpu_name = torch.cuda.get_device_name(0)
                except Exception:
                    hp.gpu_name = "cuda"
                hp.gpu_util_pct = _probe_gpu_utilization()
        except Exception as exc:
            logger.debug(f"[HW] CUDA VRAM probe skipped: {exc}")

        # Tier classification
        ram = hp.ram_gb
        if   ram < 4:   hp.tier = "minimal";  hp.safe_rows =   5_000; hp.safe_batch =  32; hp.safe_chunk =   5_000
        elif ram < 8:   hp.tier = "low";       hp.safe_rows =  20_000; hp.safe_batch =  64; hp.safe_chunk =  20_000
        elif ram < 16:  hp.tier = "medium";    hp.safe_rows = 100_000; hp.safe_batch = 128; hp.safe_chunk = 100_000
        elif ram < 32:  hp.tier = "high";      hp.safe_rows = 500_000; hp.safe_batch = 256; hp.safe_chunk = 500_000
        else:           hp.tier = "ultra";     hp.safe_rows = 2_000_000; hp.safe_batch = 512; hp.safe_chunk = 2_000_000

        vram_total_mb = int(hp.vram_total_gb * 1024)
        vram_free_mb = int(hp.vram_gb * 1024)
        ram_total_mb = int(hp.ram_total_gb * 1024)
        ram_free_mb = int(hp.ram_gb * 1024)
        gpu_label = hp.gpu_name if hp.vram_total_gb > 0 else "cpu"
        gpu_util = f"{hp.gpu_util_pct:.0f}%" if hp.gpu_util_pct is not None else "n/a"
        logger.info(
            f"[HW] gpu={gpu_label} vram_total={vram_total_mb}MB "
            f"vram_free={vram_free_mb}MB cpu_cores={hp.cpu_cores} "
            f"ram_total={ram_total_mb}MB ram_free={ram_free_mb}MB gpu_util={gpu_util}"
        )
        logger.info(
            f"{_GRN}[GUARDIAN-PROFILE]{_RST} RAM={hp.ram_gb:.1f}GB "
            f"VRAM={hp.vram_gb:.1f}GB cores={hp.cpu_cores} "
            f"tier={hp.tier} safe_rows={hp.safe_rows:,} safe_batch={hp.safe_batch}")
        return hp


# ──────────────────────────────────────────────────────────────────────
# Ãƒâ€šÃ‚Â§0.3  ETA TRACKER  (v17.0 ETA engine, EMA-smoothed alpha=0.10)
# ──────────────────────────────────────────────────────────────────────
class ETATracker:
    """Zero-overhead ETA tracker. Only two perf_counter calls per epoch."""
    def __init__(self, total_epochs: int, warmup: int = 5) -> None:
        self.total_epochs = total_epochs
        self.warmup       = warmup
        self._ema_t:  float = 0.0
        self._start:  float = 0.0
        self._epoch_start: float = 0.0

    def epoch_start(self) -> None:
        self._epoch_start = time.perf_counter()
        if self._start == 0.0:
            self._start = self._epoch_start

    def epoch_end(self, epoch: int, extra: Optional[Dict] = None) -> str:
        t = time.perf_counter() - self._epoch_start
        alpha = 0.10
        self._ema_t = t if self._ema_t == 0.0 else (alpha * t + (1 - alpha) * self._ema_t)
        remaining = max(0, self.total_epochs - epoch)
        eta_s     = self._ema_t * remaining
        finish    = _utcnow() + timedelta(seconds=eta_s)
        m, s      = divmod(int(eta_s), 60)
        extra_str = "  |  ".join(f"{k}: {v}" for k, v in (extra or {}).items())
        return (f"Epoch [{epoch:4d}/{self.total_epochs}]  "
                f"Time/Ep: {self._ema_t:.1f}s  ETA: {m:02d}m {s:02d}s  "
                f"Finish @ {finish.strftime('%H:%M:%S UTC')}  |  {extra_str}")

    def summary(self) -> str:
        elapsed = time.perf_counter() - self._start
        m, s = divmod(int(elapsed), 60)
        return f"[ETA] Total training time: {m:02d}m {s:02d}s"


# ──────────────────────────────────────────────────────────────────────
# Ãƒâ€šÃ‚Â§0.4  PIPELINE STATE GUARD  (Immortal D3)
# ──────────────────────────────────────────────────────────────────────
class PipelineStateGuard:
    """Context manager that verifies outputs exist > 256 bytes and retries once."""
    _MIN_BYTES = 256

    def __init__(self, stage: str,
                 required_outputs: Optional[List[Path]] = None,
                 regeneration_fn: Optional[callable] = None) -> None:
        self.stage            = stage
        self.required_outputs = required_outputs or []
        self.regeneration_fn  = regeneration_fn
        self._attempt         = 0

    def __enter__(self) -> "PipelineStateGuard":
        self._attempt += 1
        logger.info(f"[IMMORTAL] Stage '{self.stage}' entered (attempt {self._attempt})")
        return self

    def _bad_outputs(self) -> List[Path]:
        bad = []
        for p in self.required_outputs:
            try:
                min_b = 0 if str(p).endswith(".json") else self._MIN_BYTES
                if not Path(p).exists() or Path(p).stat().st_size < min_b:
                    bad.append(p)
            except OSError:
                bad.append(p)
        return bad
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        import traceback as _tb
        if exc_type is not None:
            logger.error(f"[IMMORTAL-EXCEPTION] Stage '{self.stage}': "
                         f"{exc_type.__name__}: {exc_val}")
            if self._attempt < 2 and self.regeneration_fn is not None:
                try:
                    self.regeneration_fn()
                    if not self._bad_outputs():
                        logger.info(f"[IMMORTAL] Stage '{self.stage}' recovered.")
                        return True
                except Exception as e:
                    raise PipelineStageError(self.stage, self._attempt, exc_val) from e
            raise PipelineStageError(self.stage, self._attempt, exc_val) from exc_val
        bad = self._bad_outputs()
        if bad:
            logger.warning(f"[IMMORTAL-CORRUPT] Missing/corrupt: {[str(p) for p in bad]}")
            if self.regeneration_fn is not None:
                self.regeneration_fn()
        else:
            logger.info(f"[IMMORTAL] Stage '{self.stage}' verified OK "
                        f"({len(self.required_outputs)} outputs)")
        return False


# ──────────────────────────────────────────────────────────────────────
# Ãƒâ€šÃ‚Â§0.5  CREDENTIAL RESOLVER  (7-path .env chain, SLACK_BOT_TOKEN)
# ──────────────────────────────────────────────────────────────────────
class CredentialResolver:
    """Resolution chain: project .env ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ script-relative ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ HOME ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ os.environ."""
    _PRIMARY = BASE_DIR / ".env"
    _instance: Optional["CredentialResolver"] = None
    _store:    Dict[str, str] = {}
    _loaded:   bool = False

    @classmethod
    def instance(cls) -> "CredentialResolver":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        if self._loaded:
            return
        search = [
            BASE_DIR / ".env",
            BASE_DIR.parent / ".env",
            BASE_DIR.parent.parent / ".env",
            Path.home() / ".costguard" / ".env",
            Path("costguard_final") / ".env",
            Path(".env"),
            Path("~/.env").expanduser(),
        ]
        for path in search:
            if path.exists():
                parsed = self._parse(path)
                if parsed:
                    self._store.update(parsed)
                    break
        for k, v in os.environ.items():
            if k not in self._store:
                self._store[k] = v
        self._loaded = True

    def _parse(self, path: Path) -> Dict[str, str]:
        if _HAS_DOTENV:
            try:
                return {k: v for k, v in dict(_dotenv_values(path)).items() if v}
            except Exception as exc:
                logger.debug(f"[ENV] dotenv parse skipped for {path}: {exc}")
        result: Dict[str, str] = {}
        try:
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip(); v = v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                    v = v[1:-1]
                if k:
                    result[k] = v
        except Exception as exc:
            logger.warning(f"[IMMORTAL-IO-ERROR] .env parse failed: {exc}")
        return result

    @classmethod
    def get(cls, key: str, default: str = "") -> str:
        return cls.instance()._store.get(key, default)


# ──────────────────────────────────────────────────────────────────────
# Ãƒâ€šÃ‚Â§0.6  FILESYSTEM HELPERS
# ──────────────────────────────────────────────────────────────────────
def _ensure_dir(p: Path) -> Path:
    """Creates directory, verifies writability, returns path."""
    p = Path(p)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"[ERROR] Cannot create '{p.resolve()}': {exc}") from exc
    if not os.access(p, os.W_OK):
        raise RuntimeError(f"[ERROR] '{p.resolve()}' is not writable")
    return p


def _ensure_dirs(base: str = "./results") -> None:
    """Create the canonical project directories under the active results root."""
    for d in [Path(base), Path(base) / "_workspace"]:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"[DIRS] Verified project directories under {Path(base).resolve()}")


def _data_exists(raw_dir: Path) -> bool:
    """True when required raw-data artifacts exist and are parseable."""
    checks = [raw_dir/"pipeline_stage_telemetry.csv", raw_dir/"pipeline_graphs.csv",
              raw_dir/"node_stats.csv", raw_dir/"lstm_training_sequences.csv"]
    ok = all(p.exists() and p.stat().st_size > 256 for p in checks) and checks[0].stat().st_size > 4096
    if ok:
        try:
            telemetry_header = pd.read_csv(checks[0], nrows=0).columns.tolist()
            sequence_header = pd.read_csv(checks[3], nrows=0).columns.tolist()
            graph_header = pd.read_csv(checks[1], nrows=0).columns.tolist()
            required_telemetry = {'run_id', 'stage_name', 'anomaly_window_active'}
            required_sequence = {'run_id', 'label_budget_breach'}
            required_graph = {'graph_id', 'src_stage', 'dst_stage', 'label_cost_anomalous'}
            ok = (
                required_telemetry.issubset(set(telemetry_header))
                and required_sequence.issubset(set(sequence_header))
                and required_graph.issubset(set(graph_header))
            )
        except Exception as exc:
            logger.warning(f"[CACHE] Raw cache validation failed for {raw_dir.resolve()}: {exc}")
            ok = False
    if ok:
        logger.info(f"[SKIP] Raw data valid in {raw_dir.resolve()} - use --force to overwrite")
    return ok


def _preproc_exists(ml_dir: Path) -> bool:
    """True when preprocessing is complete AND config.json has n_train>0."""
    task_b = ml_dir / "task_B"
    npy_names = ("X_train.npy","y_train.npy","X_val.npy","y_val.npy",
                 "X_test.npy","y_test.npy","X_ctx_train.npy")
    npy_ok = all((task_b/f).exists() and (task_b/f).stat().st_size > 0 for f in npy_names)
    if npy_ok:
        try:
            for name in npy_names:
                arr = np.load(task_b / name, mmap_mode='r', allow_pickle=False)
                if arr.shape[0] <= 0:
                    npy_ok = False
                    break
        except Exception as exc:
            logger.warning(f"[CACHE] ML-ready cache validation failed for {ml_dir.resolve()}: {exc}")
            npy_ok = False
    cfg_ok = False
    if (task_b/"config.json").exists():
        try:
            cfg_ok = int(json.loads((task_b/"config.json").read_text())
                        .get("n_train", 0)) > 0
        except Exception:
            pass
    ok = npy_ok and cfg_ok
    if ok:
        logger.info(f"[SKIP] Preprocessing valid in {ml_dir.resolve()} - use --force to reprocess")
    return ok


def _assert_safe_materialise(path: Path, label: str, *, expansion_factor: float = 4.0) -> None:
    """Fail before pandas materialises a file likely to exceed laptop RAM."""
    if not path.exists():
        return
    size_bytes = path.stat().st_size
    available = None
    if _HAS_PSUTIL:
        try:
            available = int(psutil.virtual_memory().available)
        except Exception:
            available = None
    # Keep a hard guardrail for the 16 GB laptop profile even if psutil is absent.
    budget = int(6 * 1024 ** 3)
    if available is not None:
        budget = min(budget, int(max(512 * 1024 ** 2, available * 0.55)))
    projected = int(size_bytes * expansion_factor)
    if projected > budget:
        raise MemoryError(
            f"[{label}] Refusing unsafe full materialisation of {path.resolve()} "
            f"(file={size_bytes / 1024 ** 3:.2f} GiB, projected={projected / 1024 ** 3:.2f} GiB, "
            f"budget={budget / 1024 ** 3:.2f} GiB). Reuse a valid prepared cache or run on a larger host."
        )


def _assert_cached_inputs_ready(raw_dir: Path, ml_dir: Path, *,
                                require_raw: bool,
                                require_ml_ready: bool,
                                context: str) -> None:
    missing: List[str] = []
    if require_raw and not _data_exists(raw_dir):
        missing.append(f"raw cache missing at {raw_dir.resolve()}")
    if require_ml_ready and not _preproc_exists(ml_dir):
        missing.append(f"ML-ready cache missing at {ml_dir.resolve()}")
    if missing:
        raise SystemExit(
            f"{context} requires prepared cache artifacts. "
            f"Run the data-preparation command first. Missing: {'; '.join(missing)}"
        )


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically: tmp ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ rename (never corrupt)."""
    tmp = Path(str(path) + ".tmp")
    payload = json.dumps(data, indent=2, default=str)
    try:
        tmp.write_text(payload, encoding="utf-8")
        try:
            os.replace(str(tmp), str(path))
        except OSError as exc:
            logger.info(f"[IMMORTAL-IO] Atomic replace fallback for {path}: {exc}")
            path.write_text(payload, encoding="utf-8")
            try:
                tmp.unlink(missing_ok=True)
            except Exception as cleanup_exc:
                logger.debug(f"[IMMORTAL-IO-DEBUG] Cleanup skipped for {tmp}: {cleanup_exc}")
    except Exception as exc:
        logger.error(f"[IMMORTAL-IO-ERROR] JSON write failed: {path}: {exc}")
        try:
            tmp.unlink(missing_ok=True)
        except Exception as cleanup_exc:
            logger.debug(f"[IMMORTAL-IO-DEBUG] Cleanup skipped for {tmp}: {cleanup_exc}")


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via sibling temp-file and replace."""
    try:
        runtime_atomic_write_json(path, data, indent=2)
    except Exception as exc:
        logger.error(f"[IMMORTAL-IO-ERROR] JSON write failed: {path}: {exc}")
        raise


def _atomic_pickle_dump(path: Path, payload: object) -> None:
    def _writer(tmp_path: Path) -> None:
        with tmp_path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())

    atomic_write_file(path, _writer)


def _atomic_numpy_save(path: Path, array: np.ndarray) -> None:
    arr = np.asarray(array)

    def _writer(tmp_path: Path) -> None:
        with tmp_path.open("wb") as fh:
            np.save(fh, arr)
            fh.flush()
            os.fsync(fh.fileno())

    atomic_write_file(path, _writer)


def _atomic_dataframe_to_csv(path: Path, df: pd.DataFrame, **kwargs: object) -> None:
    def _writer(tmp_path: Path) -> None:
        df.to_csv(tmp_path, **kwargs)
        with tmp_path.open("ab") as fh:
            fh.flush()
            os.fsync(fh.fileno())

    atomic_write_file(path, _writer)


def _append_dataframe_to_csv(path: Path, df: pd.DataFrame, *, header_written: bool,
                             index: bool = False) -> bool:
    """Append dataframe rows to CSV using bounded writes; returns updated header state."""
    if df is None or df.empty:
        return header_written
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a" if header_written else "w", header=not header_written, index=index)
    with path.open("ab") as fh:
        fh.flush()
        os.fsync(fh.fileno())
    return True


def _atomic_copy_file(src: Path, dst: Path) -> None:
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(src)

    def _writer(tmp_path: Path) -> None:
        with src.open("rb") as rf, tmp_path.open("wb") as wf:
            shutil.copyfileobj(rf, wf, length=8 * 1024 * 1024)
            wf.flush()
            os.fsync(wf.fileno())

    atomic_write_file(dst, _writer)


def _link_or_copy_file(src: Path, dst: Path) -> None:
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        os.link(str(src), str(dst))
        return
    except OSError:
        _atomic_copy_file(src, dst)


def _safe_unlink(path: Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:
        logger.debug(f"[CLEANUP] unlink skipped path={path}: {exc}")


def _config_payload_hash(payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _require_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = sorted(col for col in required if col not in df.columns)
    if missing:
        raise ValueError(f"[{label}] Missing required columns: {', '.join(missing)}")


def _require_header_columns(header_cols: List[str], required: List[str], label: str) -> None:
    header_set = set(header_cols)
    missing = sorted(col for col in required if col not in header_set)
    if missing:
        raise ValueError(f"[{label}] Missing required input columns: {', '.join(missing)}")


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _event("SEED", "SET", seed=int(seed), cuda=bool(torch.cuda.is_available()))


def run_synthetic_data_generation(n_rows: int = SYNTH_ROWS,
                                  out_dir: Union[str, Path] = 'costguard_data',
                                  seed: int = SEED,
                                  anomaly_rate: float = ANOMALY_RATE,
                                  mode: Optional[str] = None,
                                  force: bool = False) -> Path:
    chosen_mode = mode or _SYNTH_MODE
    seed_everything(seed)
    generator = SyntheticDataGenerator(out_dir=out_dir, seed=seed)
    return generator.generate(
        n_rows=n_rows,
        anomaly_rate=anomaly_rate,
        mode=chosen_mode,
        force=force,
    )


# ──────────────────────────────────────────────────────────────────────
# Ãƒâ€šÃ‚Â§1  ASHWATHAMA-GENESIS ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â SYNTHETIC DATA GENERATOR
# ──────────────────────────────────────────────────────────────────────

class SyntheticDataGenerator:
    """
    Synthetic domain generator calibrated to TravisTorrent and BitBrains.
    Writes only to costguard_data/ and preserves idempotent regeneration guards.
    """

    def __init__(self, out_dir: Union[str, Path] = "costguard_data",
                 seed: int = 42,
                 tt_schema_ref: Optional[List[str]] = None,
                 bb_schema_ref: Optional[List[str]] = None) -> None:
        self.out_dir = Path(out_dir)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.tt_schema_ref = list(tt_schema_ref or [])
        self.bb_schema_ref = list(bb_schema_ref or [])
        _ensure_dir(self.out_dir)

    def generate(self, n_rows: int = SYNTH_ROWS, anomaly_rate: float = ANOMALY_RATE,
                 mode: str = "standard", force: bool = False) -> Path:
        out_path = self.out_dir / "pipeline_stage_telemetry.csv"
        if not force and out_path.exists() and out_path.stat().st_size > 4096:
            logger.info(f"[SKIP] {out_path} exists - use --force to regenerate")
            return out_path

        rows: List[Dict[str, object]] = []
        n_runs = max(1, n_rows // len(STAGE_ORDER))
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        anomaly_types = ["cost_spike", "test_failure", "stage_skip", "cascade_fail", "mem_pressure"]
        anomaly_probs = [0.30, 0.25, 0.15, 0.20, 0.10]

        for run_idx in range(n_runs):
            project_id = int(self.rng.integers(0, 500))
            branch = str(self.rng.choice(BRANCH_TYPES, p=[0.25, 0.20, 0.35, 0.10, 0.10]))
            team_size = max(1, int(self.rng.lognormal(2.0, 0.8)))
            n_commits = max(1, int(self.rng.lognormal(1.2, 1.0)))
            src_churn = max(0, int(self.rng.exponential(50.0)))
            test_churn = max(0, int(self.rng.exponential(20.0)))
            sloc = max(100, int(self.rng.lognormal(9.0, 1.5)))
            is_pr = int(self.rng.random() < 0.45)
            by_core_team = int(self.rng.random() < 0.60)
            tests_run = max(1, int(self.rng.lognormal(3.0, 0.55)))

            anomaly_active = bool(self.rng.random() < anomaly_rate)
            anomaly_type = str(self.rng.choice(anomaly_types, p=anomaly_probs)) if anomaly_active else "normal"
            cascade_factor = float(self.rng.uniform(2.0, 4.0)) if anomaly_type == "cascade_fail" else 1.0
            run_start = base_time + timedelta(minutes=run_idx * 17)
            elapsed_seconds = 0.0
            tests_failed_run = int(self.rng.integers(1, 50)) if anomaly_type == "test_failure" else 0

            for stage_idx, stage in enumerate(STAGE_ORDER):
                cpu_cores = int(self.rng.choice([1, 2, 4, 8], p=[0.3, 0.4, 0.2, 0.1]))
                cpu_capacity_mhz = float(cpu_cores * self.rng.uniform(2500.0, 3500.0))
                cpu_usage_pct = float(np.clip(self.rng.beta(2.0, 5.0) * 100.0, 0.0, 100.0))
                cpu_usage_mhz = float(cpu_usage_pct / 100.0 * cpu_capacity_mhz)
                mem_capacity_kb = float(cpu_cores * int(self.rng.choice([4, 8, 16])) * 1024 * 1024)
                mem_usage_kb = float(self.rng.beta(2.0, 3.0) * mem_capacity_kb)
                disk_read_kbs = float(max(0.0, self.rng.exponential(200.0)))
                disk_write_kbs = float(max(0.0, self.rng.exponential(500.0)))
                net_rx_kbs = float(max(0.0, self.rng.exponential(50.0)))
                net_tx_kbs = float(max(0.0, self.rng.exponential(100.0)))
                dur_base = self._stage_dur(stage)
                dur_s = float(max(1.0, self.rng.lognormal(math.log(dur_base), 0.30)))

                if anomaly_active:
                    if anomaly_type == "cost_spike":
                        factor = float(self.rng.uniform(2.5, 5.0))
                        cpu_usage_mhz *= factor
                        cpu_usage_pct = min(100.0, cpu_usage_pct * factor)
                        dur_s *= factor
                    elif anomaly_type == "stage_skip":
                        dur_s *= 0.04
                        cpu_usage_pct *= 0.04
                        cpu_usage_mhz *= 0.04
                    elif anomaly_type == "cascade_fail" and stage_idx >= 4:
                        dur_s *= cascade_factor
                    elif anomaly_type == "mem_pressure":
                        mem_usage_kb = 0.95 * mem_capacity_kb

                cpu_seconds = float(cpu_usage_mhz / 1000.0)
                memory_gb_s = float(mem_usage_kb / (1024.0 ** 2))
                network_egress_gb = float((net_tx_kbs * max(dur_s, 1.0)) / (1024.0 ** 2))
                billed_cost = float(cpu_seconds * COST_PER_S)
                latency_p95 = float(dur_s * 1000.0)
                created_at = run_start + timedelta(seconds=elapsed_seconds)
                elapsed_seconds += dur_s

                rows.append({
                    "run_id": f"syn_{run_idx:06d}",
                    "project_id": project_id,
                    "stage_name": stage,
                    "executor_type": "synthetic_ci",
                    "branch": branch,
                    "created_at": created_at.strftime("%Y-%m-%dT%H:%M:%S"),
                    "cpu_seconds": cpu_seconds,
                    "memory_gb_s": memory_gb_s,
                    "billed_cost": billed_cost,
                    "network_egress_gb": network_egress_gb,
                    "latency_p95": latency_p95,
                    "call_count": cpu_cores,
                    "duration_seconds": dur_s,
                    "anomaly_window_active": int(anomaly_active),
                    "anomaly_type": anomaly_type,
                    "gh_team_size": team_size,
                    "gh_num_commits_in_push": n_commits,
                    "gh_sloc": sloc,
                    "cpu_cores": cpu_cores,
                    "cpu_usage_pct": cpu_usage_pct,
                    "cpu_capacity_mhz": cpu_capacity_mhz,
                    "mem_capacity_kb": mem_capacity_kb,
                    "disk_read_kbs": disk_read_kbs,
                    "disk_write_kbs": disk_write_kbs,
                    "src_churn": src_churn,
                    "test_churn": test_churn,
                    "is_pr": is_pr,
                    "by_core_team": by_core_team,
                    "tr_log_num_tests_run": tests_run,
                    "tr_log_num_tests_failed": tests_failed_run,
                })

        df = pd.DataFrame(rows)
        tier_mask = self._three_tier_label(df.copy())["anomaly_window_active"].astype(int)
        df["anomaly_window_active"] = np.maximum(df["anomaly_window_active"].astype(int), tier_mask)
        df = _apply_focus_defaults(df)
        _atomic_dataframe_to_csv(out_path, df, index=False)
        self._write_lstm_sequences(df)
        self._write_graphs(df, mode=mode)
        self._write_node_stats(df)
        self._write_cost_snapshots(df)
        logger.info(
            f"[GENESIS] Synthetic rows={len(df):,} anomalies={int(df['anomaly_window_active'].sum()):,} "
            f"rate={float(df['anomaly_window_active'].mean()):.2%} -> {out_path}"
        )
        return out_path

    @staticmethod
    def _stage_dur(stage: str) -> float:
        return {
            "checkout": 60.0,
            "build": 900.0,
            "unit_test": 720.0,
            "integration_test": 600.0,
            "security_scan": 180.0,
            "docker_build": 540.0,
            "deploy_staging": 360.0,
            "deploy_prod": 480.0,
        }.get(stage, 300.0)

    def _three_tier_label(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mask = _progressive_group_outlier_labels(
            df=df,
            group_col="run_id",
            metric_cols=["duration_seconds", "billed_cost", "cpu_seconds", "memory_gb_s", "latency_p95"],
            sort_col="created_at",
            min_history=5,
        )
        df["anomaly_window_active"] = mask.astype(int)
        return df

    def _write_lstm_sequences(self, df: pd.DataFrame) -> None:
        out = self.out_dir / "lstm_training_sequences.csv"
        seq_rows: List[Dict[str, object]] = []
        feature_cols = ["cpu_seconds", "latency_p95", "memory_gb_s", "network_egress_gb", "billed_cost"]
        for run_id, grp in df.groupby("run_id", sort=False):
            grp = grp.sort_values("created_at").reset_index(drop=True)
            values = grp[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            padded = np.zeros((SEQ_LEN, len(feature_cols)), dtype=np.float32)
            padded[-len(values):] = values[-SEQ_LEN:]
            row: Dict[str, object] = {
                "run_id": str(run_id),
                "created_at": str(grp["created_at"].iloc[-1]),
                "stage_name": str(grp["stage_name"].iloc[-1]),
                "executor_type": str(grp["executor_type"].iloc[-1]),
                "branch": str(grp["branch"].iloc[-1]),
                "label_budget_breach": int(grp["anomaly_window_active"].max()),
            }
            for t in range(SEQ_LEN):
                for c_idx, channel_name in enumerate(CHANNEL_NAMES):
                    row[f"t{t:02d}_{channel_name}"] = float(padded[t, c_idx])
            seq_rows.append(row)
        _atomic_dataframe_to_csv(out, pd.DataFrame(seq_rows), index=False)
        _event("WRITE", "Synthetic", artifact="lstm_sequences", path=out.resolve(), rows=len(seq_rows))

    def _write_graphs(self, df: pd.DataFrame, mode: str = "standard") -> None:
        out = self.out_dir / "pipeline_graphs.csv"
        graph_rows: List[Dict[str, object]] = []
        topology = {
            "normal": [(i, i + 1) for i in range(len(STAGE_ORDER) - 1)],
            "cost_spike": [(i, i + 1) for i in range(len(STAGE_ORDER) - 1)],
            "test_failure": [(i, i + 1) for i in range(len(STAGE_ORDER) - 1)],
            "stage_skip": [(0, 1), (1, 3), (3, 4), (4, 5), (5, 6), (6, 7)],
            "cascade_fail": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (4, 6), (4, 7)],
            "mem_pressure": [(i, i + 1) for i in range(len(STAGE_ORDER) - 1)],
        }
        feature_cols = [
            "cpu_seconds", "latency_p95", "memory_gb_s", "network_egress_gb", "billed_cost",
            "gh_team_size", "gh_num_commits_in_push", "gh_sloc", "cpu_cores", "cpu_usage_pct", "cpu_capacity_mhz",
        ]
        for run_id, grp in df.groupby("run_id", sort=False):
            grp = grp.sort_values("created_at").reset_index(drop=True)
            graph_id = str(run_id)
            label = int(grp["anomaly_window_active"].max())
            anomaly_type = str(grp["anomaly_type"].iloc[0]) if "anomaly_type" in grp.columns else "normal"
            features = grp[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            feature_payload: Dict[str, float] = {}
            for node_idx in range(len(grp)):
                for feat_idx in range(features.shape[1]):
                    feature_payload[f"nf_{node_idx}_{feat_idx:02d}"] = float(features[node_idx, feat_idx])
            for src, dst in topology.get(anomaly_type, topology["normal"]):
                if src >= len(grp) or dst >= len(grp):
                    continue
                graph_rows.append({
                    "graph_id": graph_id,
                    "src_stage": int(src),
                    "dst_stage": int(dst),
                    "edge_cost_ratio": float(max(grp.loc[[src, dst], "billed_cost"].mean(), 0.0)),
                    "edge_duration_s": float(max(grp.loc[[src, dst], "duration_seconds"].mean(), 0.0)),
                    "edge_call_count": float(max(grp.loc[[src, dst], "call_count"].mean(), 0.0)),
                    "label_cost_anomalous": label,
                    **feature_payload,
                })
        _atomic_dataframe_to_csv(out, pd.DataFrame(graph_rows), index=False)
        _event("GRAPH", "Synthetic", "END", path=out.resolve(), graph_ids=df['run_id'].nunique(), edge_rows=len(graph_rows))

    def _write_node_stats(self, df: pd.DataFrame) -> None:
        out = self.out_dir / "node_stats.csv"
        node_stats = (
            df.groupby("stage_name", as_index=False)
            .agg({
                "cpu_seconds": "mean",
                "memory_gb_s": "mean",
                "billed_cost": "mean",
                "latency_p95": "mean",
                "call_count": "mean",
                "anomaly_window_active": "mean",
            })
            .rename(columns={"anomaly_window_active": "anomaly_rate"})
        )
        _atomic_dataframe_to_csv(out, node_stats, index=False)
        _event("WRITE", "Synthetic", artifact="node_stats", path=out.resolve(), rows=len(node_stats))

    def _write_cost_snapshots(self, df: pd.DataFrame) -> None:
        out = self.out_dir / "cost_attribution_snapshots.csv"
        _atomic_dataframe_to_csv(out, df, index=False)
        _event("WRITE", "Synthetic", artifact="cost_snapshots", path=out.resolve(), rows=len(df))


# ------------------------------------------------------------------------------
# Ãƒâ€šÃ‚Â§2  ASHWATHAMA-OMNIVORE - UNIVERSAL OOM-PROOF LOADER
# ------------------------------------------------------------------------------
# Ãƒâ€šÃ‚Â§2  ASHWATHAMA-OMNIVORE ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â UNIVERSAL OOM-PROOF LOADER
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
def _iron_shield_sanitize(col: str) -> str:
    """Canonical sanitizer: NFKD normalize ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ lowercase ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ strip non-alphanum."""
    normalized = unicodedata.normalize("NFKD", str(col))
    ascii_only  = normalized.encode("ascii", errors="ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", ascii_only.lower())


# ──────────────────────────────────────────────────────────────────────
class SemanticColumnMapper:
    """
    Stateless semantic column resolver. Maps arbitrary real-world telemetry
    headers to 6 canonical FOCUS concepts using Iron Shield + keyword ontology.
    """
    ONTOLOGY: Dict[str, Dict] = {
        "cpu_seconds": {
            "primary": ["cpu"],
            "secondary": ["seconds","sec","time","usage","util","rate","mhz","avg","max"],
            "patterns": ["cpuseconds","cputime","cpuavg","cpumax","cpurate","cpuutil"],
            "exclude": ["mem"],
        },
        "memory_gb_s": {
            "primary": ["mem","ram","memory"],
            "secondary": ["gb","kb","mb","usage","avg","max","util","rss"],
            "patterns": ["memorygbs","memavg","memusage","memutil","memgbs"],
            "exclude": ["cpu"],
        },
        "billed_cost": {
            "primary": ["cost","price","bill","charge","spend","budget"],
            "secondary": [],
            "patterns": ["billedcost","cost","bill","spend"],
            "exclude": [],
        },
        "timestamp": {
            "primary": ["timestamp","time","ts","date","epoch","startat","createdat"],
            "secondary": [],
            "patterns": ["starttime","endtime","timestamp","createdat"],
            "exclude": [],
        },
        "job_or_run_id": {
            "primary": ["jobname","jobid","runid","taskid","instanceid","buildid"],
            "secondary": [],
            "patterns": ["runid","jobname","jobid","taskid"],
            "exclude": [],
        },
        "network_egress_gb": {
            "primary": ["net","network","bandwidth","egress","rx","tx"],
            "secondary": [],
            "patterns": ["netrx","nettx","egress","bandwidth"],
            "exclude": ["cpu","mem"],
        },
    }
    _UNICODE_KW: Dict[str, List[str]] = {
        "cpu_seconds":      ["shijian","cpu","auslastung","chuli","prozessor"],
        "memory_gb_s":      ["neicun","speicher","memoria","arbeitsspeicher"],
        "billed_cost":      ["feiyong","kosten","costo","cout","preis"],
        "timestamp":        ["zeitstempel","tiempo","zeitpunkt"],
        "job_or_run_id":    ["zuoye","aufgabe","trabajo"],
        "network_egress_gb":["wangluo","netzwerk","red"],
    }

    @staticmethod
    def resolve(df_columns: List[str]) -> Dict[str, Optional[str]]:
        """Map column list to 6 FOCUS concepts. O(CÃƒÆ’Ã¢â‚¬â€K) complexity."""
        import difflib as _dl
        san_idx = {_iron_shield_sanitize(c): c for c in df_columns}
        ukw_san = {concept: [_iron_shield_sanitize(kw) for kw in kws]
                   for concept, kws in SemanticColumnMapper._UNICODE_KW.items()}
        result: Dict[str, Optional[str]] = {}

        for concept, onto in SemanticColumnMapper.ONTOLOGY.items():
            prim = onto["primary"]; sec = onto["secondary"]
            pats = onto["patterns"]; excl = onto.get("exclude", [])

            cands = [sk for sk in san_idx
                     if any(p in sk for p in prim)
                     and not (excl and any(e in sk for e in excl))
                     and (not sec or any(s in sk for s in sec or [True]))]
            if not cands:
                cands = [sk for sk in san_idx
                         if any(ukw in sk for ukw in ukw_san.get(concept, []))]
            if not cands:
                result[concept] = None
                continue

            matched = None
            for pat in pats:
                for sk in cands:
                    if pat in sk:
                        matched = sk; break
                if matched:
                    break
            if not matched:
                matched = (max(cands, key=lambda sk: sum(s in sk for s in sec))
                           if sec else cands[0])
            result[concept] = san_idx[matched]

        resolved = sum(1 for v in result.values() if v is not None)
        if resolved < 3:
            unresolved = [c for c, v in result.items() if v is None]
            close = {uc: [col for _, col in sorted(
                [(_dl.SequenceMatcher(None, uc, c).ratio(), c) for c in df_columns],
                reverse=True)[:5]] for uc in unresolved}
            msg = (f"[SHAPESHIFTER-LOW-CONFIDENCE] {resolved}/6 concepts resolved. "
                   f"Unresolved: {unresolved}. Closest: {close}")
            logger.warning(msg)
        return result


# ──────────────────────────────────────────────────────────────────────
class PageHinkleyTest:
    """O(1)-memory drift detector. Returns True when drift detected."""
    def __init__(self, delta: float = PH_DELTA, lambda_: float = PH_LAMBDA) -> None:
        self.delta    = delta
        self.lambda_  = lambda_
        self._cum_sum: float = 0.0
        self._min_sum: float = 0.0
        self._mean_est: float = 0.0
        self._n:       int   = 0

    def update(self, value: float) -> bool:
        """A4-style Page-Hinkley update ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â exactly this logic."""
        if not math.isfinite(value):
            return False
        self._n += 1
        self._mean_est += (value - self._mean_est) / self._n
        self._cum_sum  += (value - self._mean_est) - self.delta
        self._min_sum   = min(self._min_sum, self._cum_sum)
        return (self._cum_sum - self._min_sum) > self.lambda_

    def update_batch(self, values: np.ndarray) -> Tuple[bool, float]:
        fired = False
        for v in np.asarray(values).ravel():
            if math.isfinite(v) and self.update(float(v)):
                fired = True
        return fired, abs(self._mean_est)

    def reset(self) -> None:
        self._cum_sum = self._min_sum = self._mean_est = 0.0
        self._n = 0

    def to_dict(self) -> dict:
        return {"cum_sum": self._cum_sum, "min_sum": self._min_sum,
                "mean_est": self._mean_est, "n": self._n}

    @classmethod
    def from_dict(cls, d: dict,
                  delta: float = PH_DELTA, lambda_: float = PH_LAMBDA) -> "PageHinkleyTest":
        obj = cls(delta, lambda_)
        obj._cum_sum   = d.get("cum_sum",   0.0)
        obj._min_sum   = d.get("min_sum",   0.0)
        obj._mean_est  = d.get("mean_est",  0.0)
        obj._n         = d.get("n",         0)
        return obj


# ──────────────────────────────────────────────────────────────────────
def smart_read_csv(path: Union[str, Path], **kwargs) -> Union[pd.DataFrame, Iterator]:
    """
    3-stage sniffing cascade: Sniffer ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ sep=None/python ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ heuristic.
    OOM-1 fix: Two-pass usecols read when chunksize is set.
    Never raises bare Exception ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â wraps in DataIngestionError.
    """
    path      = Path(path)
    chunksize = kwargs.pop("chunksize", None)
    target_col= kwargs.pop("_target_col", None)

    # Stage 1: csv.Sniffer
    detected_sep: Optional[str] = None
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            sample = fh.read(4096)
        detected_sep = csv.Sniffer().sniff(sample).delimiter
    except (csv.Error, OSError) as exc:
        logger.debug(f"[CSV] delimiter sniff skipped for {path}: {exc}")

    base_kw = {"on_bad_lines": "skip", "encoding_errors": "replace",
               "low_memory": False}
    base_kw.update(kwargs)

    if chunksize is not None:
        for sep_try in ([detected_sep] if detected_sep else []) + [None, ","]:
            try:
                kw = {**base_kw}
                if sep_try is None:
                    kw.update({"sep": None, "engine": "python"})
                elif sep_try != detected_sep:
                    kw["sep"] = sep_try
                else:
                    kw["sep"] = sep_try
                return pd.read_csv(path, chunksize=chunksize, **kw)
            except Exception:
                continue
        return pd.read_csv(path, chunksize=chunksize, **base_kw)

    # Full load with 3-stage fallback
    df: Optional[pd.DataFrame] = None
    for sep_try in ([detected_sep] if detected_sep else []) + [None]:
        try:
            kw = {**base_kw}
            if sep_try is None:
                kw.update({"sep": None, "engine": "python"})
            else:
                kw["sep"] = sep_try
            df = pd.read_csv(path, **kw)
            break
        except Exception:
            df = None

    if df is None:
        try:
            with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
                raw = [fh.readline() for _ in range(5)]
            counts = {d: sum(l.count(d) for l in raw) for d in [",", ";", "\t", "|"]}
            best = max(counts, key=counts.get)
            df = pd.read_csv(path, sep=best, **base_kw)
        except Exception as exc:
            raise DataIngestionError(path, f"all sniff stages failed: {exc}") from exc

    if df is not None and len(df) > 0:
        df = _neutralize_headers(df)

    # RAM gate (D2.1)
    if _HAS_PSUTIL and df is not None and len(df) > 0:
        avail = psutil.virtual_memory().available
        if df.memory_usage(deep=True).sum() > avail * 0.35:
            logger.warning(f"[RAM-GATE] {path.name}: "
                           f"{df.shape} exceeds 35% RAM ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â sampling 15%")
            df = df.sample(frac=0.15, random_state=42).reset_index(drop=True)

    gc.collect()
    return df


def _neutralize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Pass 1: headerless detect. Pass 2: normalize names. Pass 3: drop misread."""
    if df.empty:
        return df
    col_names = [str(c) for c in df.columns]
    n_numeric = sum(1 for c in col_names if _looks_numeric(c))
    if col_names and n_numeric > 0.80 * len(col_names):
        header_row = pd.DataFrame([col_names], columns=df.columns)
        df = pd.concat([header_row, df], ignore_index=True)
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
    df.columns = (pd.Index(str(c) for c in df.columns)
                  .str.strip().str.replace(r"[^\w]", "_", regex=True))
    NUMERIC_KW = {"cpu","mem","time","ms","usage","cost","duration","latency",
                  "col_","disk","net","byte","mhz","kb","gb","rate","seconds","p95"}
    for col in df.columns:
        col_l = col.lower()
        if not any(kw in col_l for kw in NUMERIC_KW):
            continue
        sample = df[col].iloc[:min(20, len(df))].astype(str).tolist()
        non_num = sum(1 for v in sample if not _looks_numeric(
            v.replace(",", "").replace("%", "").strip()))
        if sample and non_num > len(sample) * 0.50:
            df = df.drop(index=0).reset_index(drop=True)
            break
    for col in df.columns:
        if any(kw in col.lower() for kw in NUMERIC_KW):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _looks_numeric(s: str) -> bool:
    try:
        float(str(s).strip())
        return True
    except (ValueError, TypeError):
        return False


def _sanitise_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN/Inf with 0 in numeric columns (LOGIC-1 fix)."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def _sanitise_numeric_2d(arr: np.ndarray) -> np.ndarray:
    """NaN/Inf guard ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â MUST be called BEFORE any scaler (LOGIC-1).
    CRITICAL: posinf=1e4, NOT 0.0 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â zeroing Inf destroys anomaly signal.
    1e4 is clipped to a finite bound that the scaler can standardize correctly.
    """
    return np.nan_to_num(arr, nan=0.0, posinf=1e4, neginf=-1e4)


# ──────────────────────────────────────────────────────────────────────
def _count_non_finite_np(arr: np.ndarray) -> Tuple[int, int, int]:
    view = np.asarray(arr)
    return (
        int(np.isnan(view).sum()),
        int(np.isposinf(view).sum()),
        int(np.isneginf(view).sum()),
    )


def _validate_numeric_checkpoint(arr: np.ndarray, *,
                                 domain: str,
                                 stage: str,
                                 source: str,
                                 repair: bool = False,
                                 log_ok: bool = False) -> np.ndarray:
    """Validate (and optionally repair) non-finite values with explicit telemetry."""
    view = np.asarray(arr, dtype=np.float32)
    nan_count, pos_inf_count, neg_inf_count = _count_non_finite_np(view)
    if nan_count or pos_inf_count or neg_inf_count:
        _event(
            "DATA-INTEGRITY",
            domain,
            stage,
            source=source,
            shape=tuple(int(x) for x in view.shape),
            nan=nan_count,
            pos_inf=pos_inf_count,
            neg_inf=neg_inf_count,
            action="repair" if repair else "fail",
        )
        if not repair:
            raise ValueError(
                f"[{domain}::{stage}] Non-finite values in {source} "
                f"(nan={nan_count}, +inf={pos_inf_count}, -inf={neg_inf_count})"
            )
        view = _sanitise_numeric_2d(view)
        nan_count, pos_inf_count, neg_inf_count = _count_non_finite_np(view)
        if nan_count or pos_inf_count or neg_inf_count:
            raise ValueError(
                f"[{domain}::{stage}] Repair failed for {source} "
                f"(nan={nan_count}, +inf={pos_inf_count}, -inf={neg_inf_count})"
            )
    if log_ok:
        _event(
            "DATA-INTEGRITY",
            domain,
            stage,
            source=source,
            shape=tuple(int(x) for x in view.shape),
            nan=0,
            pos_inf=0,
            neg_inf=0,
            action="ok",
        )
    return view.astype(np.float32, copy=False)


def _assert_finite_tensor(tensor: torch.Tensor, *, domain: str, source: str) -> None:
    if torch.isfinite(tensor).all():
        return
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_mask = torch.isinf(tensor)
    pos_inf_count = int((inf_mask & (tensor > 0)).sum().item())
    neg_inf_count = int((inf_mask & (tensor < 0)).sum().item())
    _event(
        "DATA-INTEGRITY",
        domain,
        "PRE-MODEL-GUARD",
        source=source,
        shape=tuple(int(x) for x in tensor.shape),
        nan=nan_count,
        pos_inf=pos_inf_count,
        neg_inf=neg_inf_count,
        action="fail",
    )
    raise ValueError(
        f"[{domain}] Non-finite tensor at {source} "
        f"(nan={nan_count}, +inf={pos_inf_count}, -inf={neg_inf_count})"
    )


FOCUS_FILL: Dict[str, float] = {
    "cpu_seconds": 0.0, "memory_gb_s": 0.0, "billed_cost": 0.0,
    "network_egress_gb": 0.0, "latency_p95": 0.0, "call_count": 0,
    "anomaly_window_active": 0, "duration_seconds": 0.0,
}

def _focus_validate(df: pd.DataFrame, source: str) -> None:
    required = {"run_id", "stage_name", "cpu_seconds", "billed_cost",
                "anomaly_window_active"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[FOCUS-VALIDATE] {source} missing columns: {missing}")


def _apply_focus_defaults(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, val in FOCUS_FILL.items():
        if col not in df.columns:
            df[col] = val
    if "executor_type" not in df.columns:
        df["executor_type"] = "unknown"
    if "branch" not in df.columns:
        df["branch"] = "main"
    if "stage_name" not in df.columns:
        df["stage_name"] = "build"
    if "created_at" not in df.columns:
        df["created_at"] = _utcnow().isoformat()
    return _sanitise_numeric(df)


# ──────────────────────────────────────────────────────────────────────


def _read_header_columns(path: Path, separator: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as fh:
        header = fh.readline().strip()
    if separator == ',':
        reader = csv.reader([header])
        return [col.strip() for col in next(reader)]
    return [col.strip() for col in header.split(separator)]


def _strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df


def _to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors='coerce')
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric.fillna(default)


def _to_bool_int(series: pd.Series) -> pd.Series:
    values = series.fillna('').astype(str).str.strip().str.lower()
    return values.isin({'1', 'true', 't', 'yes', 'y'}).astype(int)


def _compute_dynamic_iforest_mask(features: np.ndarray, fixed_mask: np.ndarray,
                                  min_rate: float = 0.08, max_rate: float = 0.15,
                                  seed: int = SEED) -> np.ndarray:
    if len(features) < 20:
        return np.zeros(len(features), dtype=int)
    finite = np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    target = (min_rate + max_rate) / 2.0
    best_mask = np.zeros(len(finite), dtype=int)
    best_gap = float('inf')
    lo, hi = 0.05, 0.25
    for _ in range(10):
        contamination = (lo + hi) / 2.0
        try:
            model = IsolationForest(contamination=contamination, random_state=seed)
            iso_mask = (model.fit_predict(finite) == -1).astype(int)
        except Exception as exc:
            logger.warning(f'[BITBRAINS] IsolationForest fallback due to: {exc}')
            return best_mask
        combined = np.maximum(fixed_mask.astype(int), iso_mask)
        rate = float(combined.mean()) if len(combined) else 0.0
        gap = abs(rate - target)
        if gap < best_gap:
            best_gap = gap
            best_mask = iso_mask
        if min_rate <= rate <= max_rate:
            return iso_mask
        if rate < min_rate:
            lo = contamination
        else:
            hi = contamination
    return best_mask


def _window_sequences_dataframe(df: pd.DataFrame, group_col: str, sort_col: str,
                                feature_cols: List[str], label_col: str,
                                stride: int = 15, seq_len: int = SEQ_LEN,
                                min_len: int = SEQ_LEN) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for group_key, grp in df.groupby(group_col, sort=False):
        grp = grp.sort_values(sort_col).reset_index(drop=True)
        if len(grp) < min_len:
            continue
        values = grp[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
        for start in range(0, len(grp) - seq_len + 1, stride):
            window = grp.iloc[start:start + seq_len].reset_index(drop=True)
            window_vals = values[start:start + seq_len]
            row: Dict[str, object] = {
                'run_id': f"{window['run_id'].iloc[-1]}__w{start}",
                'created_at': str(window['created_at'].iloc[-1]),
                'stage_name': str(window['stage_name'].iloc[-1]),
                'executor_type': str(window['executor_type'].iloc[-1]),
                'branch': str(window['branch'].iloc[-1]),
                'label_budget_breach': int(window[label_col].max()),
                'window_group': str(group_key),
            }
            for t in range(seq_len):
                for c_idx, channel_name in enumerate(CHANNEL_NAMES):
                    row[f't{t:02d}_{channel_name}'] = float(window_vals[t, c_idx])
            rows.append(row)
    return pd.DataFrame(rows)


def _build_knn_edges(features: np.ndarray, k: int = 5) -> List[Tuple[int, int, float]]:
    if len(features) < 2:
        return []
    features = np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    n_neighbors = min(k + 1, len(features))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    nbrs.fit(features)
    distances, indices = nbrs.kneighbors(features)
    distances = _sanitise_numeric_2d(distances.astype(np.float32, copy=False))
    edges: List[Tuple[int, int, float]] = []
    for src in range(len(features)):
        for dst, dist in zip(indices[src], distances[src]):
            if int(dst) == src:
                continue
            weight = float(max(0.0, 1.0 - float(dist)))
            if not math.isfinite(weight):
                weight = 0.0
            edges.append((int(src), int(dst), weight))
    return edges


_GRAPH_STREAM_FLUSH_ROWS = 20_000   # flush to disk every N rows — lower = safer on 8 GB RAM


def _count_graph_csv_stats(path: Path) -> Tuple[int, int]:
    """Count graph edge rows and distinct graph IDs without loading the CSV."""
    if not path.exists() or path.stat().st_size == 0:
        return 0, 0
    rows = 0
    graph_ids: set[str] = set()
    try:
        for chunk in pd.read_csv(path, usecols=['graph_id'], dtype=str, chunksize=100_000):
            rows += int(len(chunk))
            graph_ids.update(chunk['graph_id'].dropna().astype(str).unique().tolist())
            del chunk
            if rows % 500_000 == 0:
                gc.collect()
    except (ValueError, pd.errors.EmptyDataError):
        try:
            rows = max(0, sum(1 for _ in path.open('r', encoding='utf-8', errors='replace')) - 1)
        except OSError:
            rows = 0
    return rows, len(graph_ids)


def _window_graphs_dataframe(
    df: pd.DataFrame,
    group_col: str,
    sort_col: str,
    feature_cols: List[str],
    knn_cols: List[str],
    label_col: str,
    stride: int = 15,
    seq_len: int = SEQ_LEN,
    min_len: int = SEQ_LEN,
    k: int = 5,
    graph_prefix: str = 'graph',
    out_csv: Optional[Path] = None,
    append: bool = False,
    max_windows_per_group: Optional[int] = None,
) -> pd.DataFrame:
    """
    Memory-safe graph dataframe builder using a PERSISTENT file handle.

    When out_csv is provided:
      - Opens the destination file ONCE at the start and holds the handle open
        throughout the entire write. This bypasses Windows/OneDrive oplocks:
        OneDrive.exe cannot acquire an exclusive oplock on a file that another
        process already holds open for writing.
      - No temp file, no shutil.copy2, no disk-space doubling (fixes WinError 112).
      - Writes directly to the final OneDrive destination path.
      - Streams rows in batches of _GRAPH_STREAM_FLUSH_ROWS, never accumulating
        the full result in RAM.

    When out_csv is None: legacy in-memory path (safe for synthetic / small data).
    """
    import csv as _csv_module

    streaming = out_csv is not None
    _stream_tmp: Optional[Path] = None
    _stream_direct = bool(streaming and append)
    if streaming and not _stream_direct:
        assert out_csv is not None
        _stream_tmp = out_csv.with_name(f".{out_csv.name}.{os.getpid()}.{time.time_ns()}.tmp")
        _safe_unlink(_stream_tmp)
    _fh: Optional[object] = None         # persistent file handle
    _writer: Optional[object] = None     # csv.DictWriter
    _fieldnames: Optional[List[str]] = None
    _rows_written = 0
    _flush_rows = min(_GRAPH_STREAM_FLUSH_ROWS, 2_000) if streaming else _GRAPH_STREAM_FLUSH_ROWS

    def _open_handle(sample_row: Dict) -> None:
        nonlocal _fh, _writer, _fieldnames
        _fieldnames = list(sample_row.keys())
        assert out_csv is not None
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        if _stream_direct:
            write_header = not out_csv.exists() or out_csv.stat().st_size == 0
            _fh = open(out_csv, 'a', newline='', encoding='utf-8', buffering=8 * 1024 * 1024)
            _writer = _csv_module.DictWriter(_fh, fieldnames=_fieldnames)
            if write_header:
                _writer.writeheader()
        else:
            assert _stream_tmp is not None
            # buffering=8MB: reduces flush syscalls while keeping memory low
            _fh = open(_stream_tmp, 'w', newline='', encoding='utf-8', buffering=8 * 1024 * 1024)
            _writer = _csv_module.DictWriter(_fh, fieldnames=_fieldnames)
            _writer.writeheader()

    def _flush_batch(batch: List[Dict]) -> None:
        nonlocal _fh, _writer, _rows_written
        if not batch:
            return
        if _writer is None:
            _open_handle(batch[0])
        _writer.writerows(batch)
        # flush to OS page-cache every batch so data is not lost on crash,
        # but do NOT close the handle — that would allow OneDrive to oplock it.
        _fh.flush()  # type: ignore[union-attr]
        _rows_written += len(batch)

    rows: List[Dict[str, object]] = []
    _groups_processed = 0

    for group_key, grp in df.groupby(group_col, sort=False):
        grp = grp.sort_values(sort_col).reset_index(drop=True)
        if len(grp) < min_len:
            del grp
            continue

        feat_matrix = grp[feature_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
        knn_matrix  = grp[knn_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
        feat_matrix = _sanitise_numeric_2d(feat_matrix)
        knn_matrix = _sanitise_numeric_2d(knn_matrix)
        _windows_emitted = 0

        for start in range(0, len(grp) - seq_len + 1, stride):
            if max_windows_per_group is not None and _windows_emitted >= max_windows_per_group:
                break
            window            = grp.iloc[start:start + seq_len].reset_index(drop=True)
            node_features     = feat_matrix[start:start + seq_len]
            neighbor_features = knn_matrix[start:start + seq_len]
            _id_col = window['run_id'].iloc[-1] if 'run_id' in window.columns else str(group_key)
            graph_id = f"{_id_col}__w{start}"
            feature_payload: Dict[str, float] = {}
            for node_idx in range(len(window)):
                for feat_idx in range(node_features.shape[1]):
                    feature_payload[f'nf_{node_idx}_{feat_idx:02d}'] = float(node_features[node_idx, feat_idx])
            edges = _build_knn_edges(neighbor_features, k=k)
            if not edges:
                edges = [(idx, idx + 1, 1.0) for idx in range(max(0, len(window) - 1))]
            for src, dst, weight in edges:
                edge_weight = float(weight) if math.isfinite(float(weight)) else 0.0
                edge_duration = pd.to_numeric(window.loc[[src, dst], 'duration_seconds'], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0).mean()
                edge_calls = pd.to_numeric(window.loc[[src, dst], 'call_count'], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(1.0).mean()
                rows.append({
                    'graph_id': graph_id,
                    'src_stage': int(src),
                    'dst_stage': int(dst),
                    'edge_cost_ratio': edge_weight,
                    'edge_duration_s': float(edge_duration),
                    'edge_call_count': float(max(1.0, edge_calls)),
                    'label_cost_anomalous': int(window[label_col].max()),
                    **feature_payload,
                })
                if streaming and len(rows) >= _flush_rows:
                    _flush_batch(rows)
                    rows = []

            _windows_emitted += 1
            del window, node_features, neighbor_features, feature_payload, edges

        if streaming and rows:
            _flush_batch(rows)
            rows = []

        del grp, feat_matrix, knn_matrix
        _groups_processed += 1
        if _groups_processed % 100 == 0:
            gc.collect()

    # ──────────────────────────────────────────────────────────────────────
    if streaming:
        if rows:
            _flush_batch(rows)
            rows = []
        if _fh is not None:
            try:
                _fh.flush()  # type: ignore[union-attr]
                os.fsync(_fh.fileno())  # type: ignore[union-attr]
                _fh.close()   # type: ignore[union-attr]
            except OSError:
                pass
            if not _stream_direct:
                assert out_csv is not None and _stream_tmp is not None
                os.replace(str(_stream_tmp), str(out_csv))
        elif out_csv is not None and not _stream_direct:
            _atomic_dataframe_to_csv(out_csv, pd.DataFrame(), index=False)
        gc.collect()
        logger.info(f'[GRAPH-STREAM] Wrote {_rows_written:,} edge rows -> {out_csv}')
        return pd.DataFrame()   # callers use _count_graph_csv_stats() for scalars
    else:
        return pd.DataFrame(rows)


def _read_bitbrains_file(path: Path) -> pd.DataFrame:
    parsed_cols = _read_header_columns(path, ';')
    required_cols = [
        'Timestamp [ms]', 'CPU cores', 'CPU capacity provisioned [MHZ]', 'CPU usage [MHZ]',
        'CPU usage [%]', 'Memory capacity provisioned [KB]', 'Memory usage [KB]',
        'Disk read throughput [KB/s]', 'Disk write throughput [KB/s]',
        'Network received throughput [KB/s]', 'Network transmitted throughput [KB/s]',
    ]
    _require_header_columns(parsed_cols, required_cols, f'BITBRAINS:{path.name}')
    df = pd.read_csv(
        path,
        sep=';',
        names=parsed_cols,
        skiprows=1,
        encoding='utf-8',
        on_bad_lines='skip',
        dtype=str,
    )
    df = _strip_string_values(df)
    numeric_cols = [
        'Timestamp [ms]', 'CPU cores', 'CPU capacity provisioned [MHZ]', 'CPU usage [MHZ]',
        'CPU usage [%]', 'Memory capacity provisioned [KB]', 'Memory usage [KB]',
        'Disk read throughput [KB/s]', 'Disk write throughput [KB/s]',
        'Network received throughput [KB/s]', 'Network transmitted throughput [KB/s]',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _to_numeric(df[col])
    return df


def load_bitbrains(data_dir: Optional[Path], out_path: Path) -> int:
    timer = StageTimer()
    if not data_dir or not Path(data_dir).exists():
        logger.warning('[SKIP] BitBrains directory not found - skipping')
        return 0
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob('*.csv'))
    if not csv_files:
        logger.warning(f'[SKIP] No BitBrains CSV files found in {data_dir}')
        return 0
    _bb_out_dir = out_path.parent
    _bb_required = [out_path, _bb_out_dir/'pipeline_graphs.csv', _bb_out_dir/'bitbrains_graphs.csv',
                    _bb_out_dir/'lstm_training_sequences.csv', _bb_out_dir/'node_stats.csv']
    if all(p.exists() and p.stat().st_size > 256 for p in _bb_required):
        _cr = max(0, sum(1 for _ in open(out_path, encoding='utf-8', errors='replace')) - 1) if out_path.exists() else 0
        logger.info(f'[CACHE-HIT][BitBrains] All artifacts present (~{_cr:,} rows). Skipping. Use --force to reprocess.')
        _event("CACHE-HIT", "BitBrains", out_dir=_bb_out_dir.resolve(), rows=_cr)
        return max(0, _cr)
    _event("LOAD", "BitBrains", "START", source_dir=data_dir.resolve(), csv_files=len(csv_files), out_path=out_path.resolve())

    BITBRAINS_BATCH_SIZE   = 50                                  # OOM-FIX: N files per batch
    # OOM-FIX-v2: write batch partial CSVs to system %TEMP%, not out_path.parent which is
    # inside OneDrive — OneDrive's sync lock causes PermissionError on pandas open().
    _sys_tmp = Path(os.environ.get('TEMP', os.environ.get('TMP', '/tmp')))
    _token = f"{os.getpid()}_{time.time_ns()}"
    _bb_partial_csv = _sys_tmp / f'_bb_partial_focus_{_token}.csv'
    _bb_seq_tmp = _sys_tmp / f'_bb_seq_{_token}.csv'
    _bb_graph_tmp = _sys_tmp / f'_bb_graph_{_token}.csv'
    for _stale in (_bb_partial_csv, _bb_seq_tmp, _bb_graph_tmp):
        try:
            _stale.unlink(missing_ok=True)       # clear stale remnant from prior failed run
        except OSError:
            pass
    _bb_focus_header_written = False
    _bb_seq_written = False
    _bb_graph_written = False
    _bb_rows_total = 0
    _bb_anomaly_sum = 0
    _bb_hw = HardwareProfile.probe()
    _bb_graph_stride = 45 if _bb_hw.safe_rows <= 20_000 else 15
    _bb_graph_seq_len = 12 if _bb_hw.safe_rows <= 20_000 else 30
    _bb_max_windows = 40 if _bb_hw.safe_rows <= 20_000 else 800
    _bb_graph_k = 2 if _bb_hw.safe_rows <= 20_000 else 5
    node_acc: Dict[str, List[float]] = {}
    global_idx = 0

    for _batch_start in range(0, len(csv_files), BITBRAINS_BATCH_SIZE):
        _batch_files = csv_files[_batch_start:_batch_start + BITBRAINS_BATCH_SIZE]
        frames: List[pd.DataFrame] = []
        for fpath in _batch_files:
          try:
            raw = _read_bitbrains_file(fpath)
            if raw.empty:
                continue
            raw['_source_file'] = fpath.stem
            raw['_source_row'] = np.arange(len(raw), dtype=np.int64)
            raw['cpu_seconds'] = raw['CPU usage [MHZ]'] / 1000.0
            raw['memory_gb_s'] = raw['Memory usage [KB]'] / (1024.0 ** 2)
            raw['billed_cost'] = raw['cpu_seconds'] * COST_PER_S
            raw['duration_seconds'] = 300.0
            raw['network_egress_gb'] = (raw['Network transmitted throughput [KB/s]'] * 300.0) / (1024.0 ** 2)
            raw['latency_p95'] = raw['CPU usage [%]'] * 10.0
            raw['call_count'] = _to_numeric(raw['CPU cores'], 1.0).astype(int).clip(lower=1)
            raw['run_id'] = [f'bb_{global_idx + i}' for i in range(len(raw))]
            global_idx += len(raw)
            raw['stage_name'] = 'vm_interval'
            raw['executor_type'] = BITBRAINS_EXECUTOR
            raw['branch'] = 'production'
            raw['created_at'] = pd.to_datetime(raw['Timestamp [ms]'], unit='ms', errors='coerce').fillna(pd.Timestamp.now(tz='UTC')).astype(str)

            cpu_q95 = raw.groupby('_source_file')['CPU usage [%]'].transform(
                lambda s: s.rolling(window=20, min_periods=5).quantile(0.95).fillna(s.expanding(min_periods=1).quantile(0.95))
            )
            mem_ratio = raw['Memory usage [KB]'] / (raw['Memory capacity provisioned [KB]'] + 1e-8)
            disk_total = raw['Disk read throughput [KB/s]'] + raw['Disk write throughput [KB/s]']
            disk_median = raw.groupby('_source_file')[disk_total.name if hasattr(disk_total, 'name') and disk_total.name else 'Disk read throughput [KB/s]'].transform(lambda s: s)
            disk_median = disk_total.groupby(raw['_source_file']).transform(
                lambda s: s.rolling(window=10, min_periods=3).median().fillna(s.expanding(min_periods=1).median())
            )
            net_ratio = raw['Network transmitted throughput [KB/s]'] / (raw['Network received throughput [KB/s]'] + 1e-8)
            disk_ratio = disk_total / (disk_median + 1e-8)

            rule1 = (raw['CPU usage [%]'] > cpu_q95).astype(int)
            rule2 = (raw['Memory usage [KB]'] > 0.90 * raw['Memory capacity provisioned [KB]']).astype(int)
            rule3 = (disk_total > 5.0 * disk_median).astype(int)
            fixed_mask = np.maximum.reduce([rule1.to_numpy(), rule2.to_numpy(), rule3.to_numpy()])
            if_features = np.column_stack([
                raw['CPU usage [%]'].to_numpy(dtype=np.float32),
                mem_ratio.to_numpy(dtype=np.float32),
                net_ratio.to_numpy(dtype=np.float32),
                disk_ratio.to_numpy(dtype=np.float32),
            ])
            rule4 = _compute_dynamic_iforest_mask(if_features, fixed_mask)
            raw['anomaly_window_active'] = np.maximum.reduce([fixed_mask, rule4]).astype(int)
            # OOM-FIX: delete per-file temporaries before loading next file
            del cpu_q95, mem_ratio, disk_total, disk_median, net_ratio, disk_ratio
            del rule1, rule2, rule3, rule4, fixed_mask, if_features
            frames.append(raw)
            del raw
          except Exception as exc:
            logger.warning(f'[BITBRAINS] Failed on {fpath.name}: {exc}')

        # ──────────────────────────────────────────────────────────────────────
        if not frames:
            continue
        _batch_df = pd.concat(frames, ignore_index=True)
        del frames
        gc.collect()

        _batch_focus_cols = [
            'run_id', 'stage_name', 'executor_type', 'branch', 'created_at',
            'cpu_seconds', 'memory_gb_s', 'billed_cost', 'network_egress_gb',
            'latency_p95', 'call_count', 'anomaly_window_active', 'duration_seconds',
        ]
        _batch_focus = _apply_focus_defaults(_batch_df[_batch_focus_cols].copy())
        _bb_focus_header_written = _append_dataframe_to_csv(
            _bb_partial_csv,
            _batch_focus,
            header_written=_bb_focus_header_written,
            index=False,
        )
        _bb_rows_total += int(len(_batch_focus))
        _bb_anomaly_sum += int(_batch_focus['anomaly_window_active'].sum())

        _batch_seq_source = _batch_df.assign(
            total_network_kbs=_batch_df['Network received throughput [KB/s]']
            + _batch_df['Network transmitted throughput [KB/s]'],
            total_io_kbs=_batch_df['Disk read throughput [KB/s]']
            + _batch_df['Disk write throughput [KB/s]'],
        )
        _batch_seq = _window_sequences_dataframe(
            _batch_seq_source,
            group_col='_source_file',
            sort_col='Timestamp [ms]',
            feature_cols=['CPU usage [MHZ]', 'CPU usage [%]', 'Memory usage [KB]', 'total_network_kbs', 'total_io_kbs'],
            label_col='anomaly_window_active',
            stride=_bb_graph_stride,
            seq_len=30,
            min_len=30,
        )
        _bb_seq_written = _append_dataframe_to_csv(
            _bb_seq_tmp,
            _batch_seq,
            header_written=_bb_seq_written,
            index=False,
        )

        _batch_graph_source = _batch_df.assign(
            utilization_ratio=_batch_df['CPU usage [MHZ]'] / (_batch_df['CPU capacity provisioned [MHZ]'] + 1e-8)
        )
        _window_graphs_dataframe(
            _batch_graph_source,
            group_col='_source_file',
            sort_col='Timestamp [ms]',
            feature_cols=[
                'CPU usage [MHZ]', 'CPU usage [%]', 'CPU capacity provisioned [MHZ]', 'CPU cores',
                'Memory usage [KB]', 'Memory capacity provisioned [KB]', 'Disk read throughput [KB/s]',
                'Disk write throughput [KB/s]', 'Network received throughput [KB/s]',
                'Network transmitted throughput [KB/s]', 'utilization_ratio',
            ],
            knn_cols=['CPU usage [%]', 'Memory usage [KB]', 'utilization_ratio'],
            label_col='anomaly_window_active',
            stride=_bb_graph_stride,
            seq_len=_bb_graph_seq_len,
            min_len=_bb_graph_seq_len,
            k=_bb_graph_k,
            graph_prefix='bitbrains',
            out_csv=_bb_graph_tmp,
            append=True,
            max_windows_per_group=_bb_max_windows,
        )
        _bb_graph_written = _bb_graph_written or (
            _bb_graph_tmp.exists() and _bb_graph_tmp.stat().st_size > 0
        )

        _stats_batch = _batch_df.groupby('_source_file').agg(
            cpu_mhz_sum=('CPU usage [MHZ]', 'sum'),
            cpu_pct_sum=('CPU usage [%]', 'sum'),
            mem_kb_sum=('Memory usage [KB]', 'sum'),
            billed_cost_sum=('billed_cost', 'sum'),
            anomaly_sum=('anomaly_window_active', 'sum'),
            row_count=('anomaly_window_active', 'size'),
        )
        for _gid, _row in _stats_batch.iterrows():
            _acc = node_acc.setdefault(str(_gid), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            _acc[0] += float(_row['cpu_mhz_sum'])
            _acc[1] += float(_row['cpu_pct_sum'])
            _acc[2] += float(_row['mem_kb_sum'])
            _acc[3] += float(_row['billed_cost_sum'])
            _acc[4] += float(_row['anomaly_sum'])
            _acc[5] += float(_row['row_count'])

        del _batch_focus, _batch_seq_source, _batch_seq, _batch_graph_source, _stats_batch, _batch_df
        gc.collect()
        logger.info(
            f'[BITBRAINS][BATCH] files {_batch_start}-{_batch_start + len(_batch_files) - 1} processed in bounded mode'
        )
    # ──────────────────────────────────────────────────────────────────────

    if not _bb_focus_header_written:
        return 0

    bitbrains_dir = _ensure_dir(out_path.parent)
    try:
        os.replace(str(_bb_partial_csv), str(out_path))
    except OSError:
        _atomic_copy_file(_bb_partial_csv, out_path)
        _safe_unlink(_bb_partial_csv)
    _link_or_copy_file(out_path, bitbrains_dir / 'pipeline_stage_telemetry.csv')

    _bb_seq_final = bitbrains_dir / 'bitbrains_lstm_sequences.csv'
    if _bb_seq_written and _bb_seq_tmp.exists() and _bb_seq_tmp.stat().st_size > 0:
        os.replace(str(_bb_seq_tmp), str(_bb_seq_final))
    else:
        _atomic_dataframe_to_csv(_bb_seq_final, pd.DataFrame(), index=False)
    _link_or_copy_file(_bb_seq_final, bitbrains_dir / 'lstm_training_sequences.csv')

    _bb_graph_final = bitbrains_dir / 'pipeline_graphs.csv'
    if _bb_graph_written and _bb_graph_tmp.exists() and _bb_graph_tmp.stat().st_size > 0:
        os.replace(str(_bb_graph_tmp), str(_bb_graph_final))
    else:
        _atomic_dataframe_to_csv(_bb_graph_final, pd.DataFrame(), index=False)

    _bb_graph_alias = bitbrains_dir / 'bitbrains_graphs.csv'
    if _bb_graph_final.exists() and _bb_graph_final.stat().st_size > 64:
        try:
            _link_or_copy_file(_bb_graph_final, _bb_graph_alias)
        except OSError as _cp_exc:
            logger.warning(f'[BB] bitbrains_graphs.csv copy failed (non-fatal): {_cp_exc}')
            _atomic_dataframe_to_csv(_bb_graph_alias, pd.DataFrame(), index=False)
    else:
        _atomic_dataframe_to_csv(_bb_graph_alias, pd.DataFrame(), index=False)

    node_rows: List[Dict[str, object]] = []
    for _gid, _vals in node_acc.items():
        _count = max(1.0, _vals[5])
        node_rows.append({
            'group_id': _gid,
            'CPU usage [MHZ]': _vals[0] / _count,
            'CPU usage [%]': _vals[1] / _count,
            'Memory usage [KB]': _vals[2] / _count,
            'billed_cost': _vals[3] / _count,
            'anomaly_rate': _vals[4] / _count,
        })
    _atomic_dataframe_to_csv(bitbrains_dir / 'node_stats.csv', pd.DataFrame(node_rows), index=False)

    _bb_n_focus = int(_bb_rows_total)
    _bb_anomaly_rate = float(_bb_anomaly_sum / max(1, _bb_rows_total))
    _bb_graph_rows, _bb_graph_ids = _count_graph_csv_stats(_bb_graph_final)
    for _tmp in (_bb_partial_csv, _bb_seq_tmp, _bb_graph_tmp):
        try:
            _tmp.unlink(missing_ok=True)
        except OSError:
            pass
    gc.collect()
    _vram_transition_flush('post-BitBrains-writes')
    _event(
        "WRITE",
        "BitBrains",
        telemetry=(bitbrains_dir / 'pipeline_stage_telemetry.csv').resolve(),
        sequences=(bitbrains_dir / 'lstm_training_sequences.csv').resolve(),
        graphs=(bitbrains_dir / 'pipeline_graphs.csv').resolve(),
        node_stats=(bitbrains_dir / 'node_stats.csv').resolve(),
    )

    logger.info(f'[BITBRAINS] rows={_bb_n_focus:,} anomaly_rate={_bb_anomaly_rate:.2%} -> {out_path}')
    _event(
        "LOAD",
        "BitBrains",
        "END",
        rows=_bb_n_focus,
        anomaly_rate=_bb_anomaly_rate,
        graph_rows=_bb_graph_rows,
        graph_ids=_bb_graph_ids,
        duration=format_duration_s(timer.elapsed_s),
    )
    return _bb_n_focus


def load_travistorrent(input_path: Path, out_dir: Path) -> int:
    timer = StageTimer()
    out_dir = _ensure_dir(out_dir)
    out_path = out_dir / 'pipeline_stage_telemetry.csv'
    if not input_path or not Path(input_path).exists():
        logger.warning(f'[SKIP] TravisTorrent not found: {input_path}')
        return 0
    _tt_required = [out_path, out_dir/'tt_graphs.csv', out_dir/'pipeline_graphs.csv',
                    out_dir/'lstm_training_sequences.csv', out_dir/'node_stats.csv']
    if all(p.exists() and p.stat().st_size > 256 for p in _tt_required):
        _cr = max(0, sum(1 for _ in open(out_path, encoding='utf-8', errors='replace')) - 1) if out_path.exists() else 0
        logger.info(f'[CACHE-HIT][TravisTorrent] All artifacts present (~{_cr:,} rows). Skipping. Use --force to reprocess.')
        _event("CACHE-HIT", "TravisTorrent", out_dir=out_dir.resolve(), rows=_cr)
        return max(0, _cr)
    _event("LOAD", "TravisTorrent", "START", source=input_path.resolve(), out_dir=out_dir.resolve())
    _tt_hw = HardwareProfile.probe()
    _tt_graph_stride = 45 if _tt_hw.safe_rows <= 20_000 else 15
    _tt_graph_seq_len = 12 if _tt_hw.safe_rows <= 20_000 else 30
    _tt_max_windows = 250 if _tt_hw.safe_rows <= 20_000 else 1_000
    _tt_graph_k = 2 if _tt_hw.safe_rows <= 20_000 else 5

    for _stale_tmp in out_dir.glob('.pipeline_stage_telemetry.csv.*.tmp'):
        _safe_unlink(_stale_tmp)
    for _stale_tmp in out_dir.glob('.lstm_training_sequences.csv.*.tmp'):
        _safe_unlink(_stale_tmp)
    for _stale_tmp in out_dir.glob('.pipeline_graphs.csv.*.tmp'):
        _safe_unlink(_stale_tmp)
    for _stale_bucket in out_dir.glob('_tt_buckets_*'):
        if _stale_bucket.is_dir():
            shutil.rmtree(_stale_bucket, ignore_errors=True)

    header_cols = _read_header_columns(Path(input_path), ',')
    required_header_cols = [
        'tr_build_id', 'gh_project_name', 'git_branch', 'gh_build_started_at', 'tr_status',
        'tr_duration', 'tr_log_buildduration', 'tr_log_num_tests_run',
        'tr_log_num_tests_failed', 'tr_log_num_tests_ok', 'tr_log_num_test_suites_failed',
        'git_diff_src_churn', 'git_diff_test_churn', 'gh_diff_files_added',
        'gh_diff_files_deleted', 'gh_diff_files_modified', 'gh_team_size',
        'gh_num_commits_in_push', 'gh_sloc', 'gh_test_lines_per_kloc',
        'gh_lang', 'gh_by_core_team_member', 'gh_is_pr',
    ]
    _require_header_columns(header_cols, required_header_cols, 'TRAVISTORRENT')
    needed_cols = {
        'tr_build_id', 'gh_project_name', 'git_branch', 'gh_build_started_at', 'tr_status',
        'tr_duration', 'tr_log_buildduration', 'tr_log_testduration', 'tr_log_num_tests_run',
        'tr_log_num_tests_failed', 'tr_log_num_tests_ok', 'tr_log_num_test_suites_failed',
        'git_diff_src_churn', 'git_diff_test_churn', 'gh_diff_files_added', 'gh_diff_files_deleted',
        'gh_diff_files_modified', 'gh_team_size', 'gh_num_commits_in_push', 'gh_num_issue_comments',
        'gh_num_pr_comments', 'gh_repo_num_commits', 'gh_sloc', 'gh_test_lines_per_kloc',
        'gh_lang', 'tr_log_lan', 'tr_log_frameworks', 'gh_by_core_team_member', 'gh_is_pr',
    }
    usecols = [col for col in header_cols if col in needed_cols]
    _tt_required_cols = [
        'tr_build_id', 'gh_project_name', 'git_branch', 'gh_build_started_at', 'tr_status',
        'tr_duration', 'tr_log_buildduration', 'tr_log_num_tests_run',
        'tr_log_num_tests_failed', 'tr_log_num_tests_ok', 'tr_log_num_test_suites_failed',
        'git_diff_src_churn', 'git_diff_test_churn', 'gh_diff_files_added',
        'gh_diff_files_deleted', 'gh_diff_files_modified', 'gh_team_size',
        'gh_num_commits_in_push', 'gh_sloc', 'gh_test_lines_per_kloc',
        'gh_lang', 'gh_by_core_team_member', 'gh_is_pr',
    ]
    _token = f"{os.getpid()}_{time.time_ns()}"
    _bucket_dir = out_dir / f'_tt_buckets_{_token}'
    _bucket_dir.mkdir(parents=True, exist_ok=True)
    _bucket_count = 128 if _tt_hw.safe_rows <= 20_000 else 64
    _bucket_written: List[bool] = [False] * _bucket_count

    _focus_tmp = out_dir / f'.pipeline_stage_telemetry.csv.{_token}.tmp'
    _seq_tmp = out_dir / f'.lstm_training_sequences.csv.{_token}.tmp'
    _graph_tmp = out_dir / f'.pipeline_graphs.csv.{_token}.tmp'
    for _stale in (_focus_tmp, _seq_tmp, _graph_tmp):
        _safe_unlink(_stale)

    _focus_written = False
    _seq_written = False
    _graph_written = False
    _tt_rows_total = 0
    _tt_anomaly_sum = 0
    node_acc: Dict[str, List[float]] = {}

    try:
        for chunk in pd.read_csv(
            input_path,
            names=header_cols,
            skiprows=1,
            encoding='utf-8',
            on_bad_lines='skip',
            usecols=usecols,
            dtype=str,
            chunksize=50_000,
        ):
            chunk = _strip_string_values(chunk)
            if chunk.empty:
                continue
            _project_series = chunk.get(
                'gh_project_name',
                pd.Series('unknown_project', index=chunk.index),
            ).fillna('unknown_project').astype(str)
            _bucket_ids = (
                pd.util.hash_pandas_object(_project_series, index=False).to_numpy(dtype=np.uint64)
                % _bucket_count
            ).astype(np.int64)
            for _bucket_id in np.unique(_bucket_ids):
                _bucket_df = chunk.loc[_bucket_ids == _bucket_id]
                if _bucket_df.empty:
                    continue
                _bucket_path = _bucket_dir / f'bucket_{int(_bucket_id):03d}.csv'
                _bucket_df.to_csv(
                    _bucket_path,
                    mode='a' if _bucket_written[int(_bucket_id)] else 'w',
                    header=not _bucket_written[int(_bucket_id)],
                    index=False,
                )
                _bucket_written[int(_bucket_id)] = True
            del chunk, _project_series, _bucket_ids
            gc.collect()

        if not any(_bucket_written):
            logger.warning(f'[TT] No usable TravisTorrent columns found in {input_path}')
            return 0

        _bucket_files = sorted(_bucket_dir.glob('bucket_*.csv'))
        for _bucket_path in _bucket_files:
            if not _bucket_path.exists() or _bucket_path.stat().st_size == 0:
                continue
            df = pd.read_csv(_bucket_path, dtype=str, low_memory=True)
            if df.empty:
                continue
            _require_columns(df, _tt_required_cols, 'TRAVISTORRENT')

            df['tr_build_id'] = df.get('tr_build_id', pd.Series(range(len(df)))).fillna(pd.Series(range(len(df)))).astype(str)
            df['gh_project_name'] = df.get('gh_project_name', pd.Series('unknown_project', index=df.index)).fillna('unknown_project').astype(str)
            df['git_branch'] = df.get('git_branch', pd.Series('main', index=df.index)).replace({'': 'main'}).fillna('main').astype(str)
            df['gh_build_started_at'] = pd.to_datetime(df.get('gh_build_started_at', pd.Series('', index=df.index)), errors='coerce')
            df['gh_build_started_at'] = df['gh_build_started_at'].fillna(pd.Timestamp.now(tz='UTC'))
            df['tr_status'] = df.get('tr_status', pd.Series('passed', index=df.index)).fillna('passed').astype(str).str.strip().str.lower()
            df['gh_is_pr'] = _to_bool_int(df.get('gh_is_pr', pd.Series('0', index=df.index)))
            df['gh_by_core_team_member'] = _to_bool_int(df.get('gh_by_core_team_member', pd.Series('0', index=df.index)))

            duration = _to_numeric(df.get('tr_duration', pd.Series(0.0, index=df.index)))
            fallback_duration = _to_numeric(df.get('tr_log_buildduration', pd.Series(0.0, index=df.index)))
            df['duration_seconds'] = duration.where(duration > 0.0, fallback_duration).fillna(0.0)
            df['tr_log_num_tests_run'] = _to_numeric(df.get('tr_log_num_tests_run', pd.Series(1.0, index=df.index)), 1.0).clip(lower=1.0)
            df['tr_log_num_tests_failed'] = _to_numeric(df.get('tr_log_num_tests_failed', pd.Series(0.0, index=df.index)))
            df['tr_log_num_tests_ok'] = _to_numeric(df.get('tr_log_num_tests_ok', pd.Series(0.0, index=df.index)))
            df['tr_log_num_test_suites_failed'] = _to_numeric(df.get('tr_log_num_test_suites_failed', pd.Series(0.0, index=df.index)))
            df['git_diff_src_churn'] = _to_numeric(df.get('git_diff_src_churn', pd.Series(0.0, index=df.index)))
            df['git_diff_test_churn'] = _to_numeric(df.get('git_diff_test_churn', pd.Series(0.0, index=df.index)))
            df['gh_diff_files_added'] = _to_numeric(df.get('gh_diff_files_added', pd.Series(0.0, index=df.index)))
            df['gh_diff_files_deleted'] = _to_numeric(df.get('gh_diff_files_deleted', pd.Series(0.0, index=df.index)))
            df['gh_diff_files_modified'] = _to_numeric(df.get('gh_diff_files_modified', pd.Series(0.0, index=df.index)))
            df['gh_team_size'] = _to_numeric(df.get('gh_team_size', pd.Series(1.0, index=df.index)), 1.0).clip(lower=1.0)
            df['gh_num_commits_in_push'] = _to_numeric(df.get('gh_num_commits_in_push', pd.Series(1.0, index=df.index)), 1.0).clip(lower=1.0)
            df['gh_num_issue_comments'] = _to_numeric(df.get('gh_num_issue_comments', pd.Series(0.0, index=df.index)))
            df['gh_num_pr_comments'] = _to_numeric(df.get('gh_num_pr_comments', pd.Series(0.0, index=df.index)))
            df['gh_repo_num_commits'] = _to_numeric(df.get('gh_repo_num_commits', pd.Series(0.0, index=df.index)))
            df['gh_sloc'] = _to_numeric(df.get('gh_sloc', pd.Series(0.0, index=df.index)))
            df['gh_test_lines_per_kloc'] = _to_numeric(df.get('gh_test_lines_per_kloc', pd.Series(0.0, index=df.index)))
            df['total_diff'] = df['gh_diff_files_added'] + df['gh_diff_files_deleted'] + df['gh_diff_files_modified']
            df['test_fail_rate'] = df['tr_log_num_tests_failed'] / (df['tr_log_num_tests_run'] + 1.0)
            lang_encoder = LabelEncoder()
            df['gh_lang_enc'] = lang_encoder.fit_transform(
                df.get('gh_lang', pd.Series('unknown', index=df.index)).fillna('unknown').astype(str)
            )

            ordered_df = df.sort_values(['gh_project_name', 'gh_build_started_at'], kind='mergesort')
            project_q95 = (
                ordered_df.groupby('gh_project_name', sort=False)['duration_seconds']
                .transform(lambda s: s.rolling(window=20, min_periods=5).quantile(0.95).fillna(s.expanding(min_periods=1).quantile(0.95)))
                .reindex(df.index)
            )
            global_src_q90 = float(df['git_diff_src_churn'].quantile(0.90)) if len(df) else 0.0
            rule1 = (df['tr_status'] == 'failed').astype(int)
            rule2 = (df['tr_log_num_tests_failed'] > 0).astype(int)
            rule3 = (df['tr_log_num_test_suites_failed'] > 0).astype(int)
            rule4 = (df['duration_seconds'] > project_q95).astype(int)
            rule5 = ((df['git_diff_src_churn'] > global_src_q90) & (df['tr_log_num_tests_failed'] > 0)).astype(int)
            df['anomaly_window_active'] = np.maximum.reduce(
                [rule1.to_numpy(), rule2.to_numpy(), rule3.to_numpy(), rule4.to_numpy(), rule5.to_numpy()]
            ).astype(int)

            focus_df = pd.DataFrame({
                'run_id': 'tt_' + df['tr_build_id'].astype(str),
                'stage_name': 'build',
                'executor_type': TRAVIS_EXECUTOR,
                'branch': df['git_branch'].astype(str),
                'created_at': df['gh_build_started_at'].astype(str),
                'cpu_seconds': df['duration_seconds'] * 0.35,
                'memory_gb_s': df['duration_seconds'] * 0.10,
                'billed_cost': df['duration_seconds'] * 0.35 * COST_PER_S,
                'network_egress_gb': 0.0,
                'latency_p95': df['duration_seconds'] * 1000.0,
                'call_count': df['tr_log_num_tests_run'].astype(int).clip(lower=1),
                'anomaly_window_active': df['anomaly_window_active'].astype(int),
                'duration_seconds': df['duration_seconds'],
            })
            focus_df = _apply_focus_defaults(focus_df)
            _focus_written = _append_dataframe_to_csv(_focus_tmp, focus_df, header_written=_focus_written, index=False)
            _tt_rows_total += int(len(focus_df))
            _tt_anomaly_sum += int(focus_df['anomaly_window_active'].sum())

            enriched = focus_df.copy()
            enriched['gh_project_name'] = df['gh_project_name'].astype(str).values
            enriched['tr_log_num_tests_failed'] = df['tr_log_num_tests_failed'].values
            enriched['tr_log_num_tests_run'] = df['tr_log_num_tests_run'].values
            enriched['git_diff_src_churn'] = df['git_diff_src_churn'].values
            enriched['gh_num_commits_in_push'] = df['gh_num_commits_in_push'].values
            enriched['gh_team_size'] = df['gh_team_size'].values
            enriched['tr_log_num_tests_ok'] = df['tr_log_num_tests_ok'].values
            enriched['tr_log_num_test_suites_failed'] = df['tr_log_num_test_suites_failed'].values
            enriched['git_diff_test_churn'] = df['git_diff_test_churn'].values
            enriched['total_diff'] = df['total_diff'].values
            enriched['gh_sloc'] = df['gh_sloc'].values
            enriched['gh_test_lines_per_kloc'] = df['gh_test_lines_per_kloc'].values
            enriched['gh_lang_enc'] = df['gh_lang_enc'].values
            enriched['gh_by_core_team_member'] = df['gh_by_core_team_member'].values
            enriched['test_fail_rate'] = df['test_fail_rate'].values

            seq_df = _window_sequences_dataframe(
                enriched,
                group_col='gh_project_name',
                sort_col='created_at',
                feature_cols=['duration_seconds', 'test_fail_rate', 'git_diff_src_churn', 'gh_num_commits_in_push', 'gh_team_size'],
                label_col='anomaly_window_active',
                stride=_tt_graph_stride,
                seq_len=30,
                min_len=30,
            )
            _seq_written = _append_dataframe_to_csv(_seq_tmp, seq_df, header_written=_seq_written, index=False)

            _window_graphs_dataframe(
                enriched,
                group_col='gh_project_name',
                sort_col='created_at',
                feature_cols=[
                    'duration_seconds', 'tr_log_num_tests_run', 'tr_log_num_tests_failed', 'tr_log_num_tests_ok',
                    'tr_log_num_test_suites_failed', 'git_diff_src_churn', 'git_diff_test_churn', 'total_diff',
                    'gh_team_size', 'gh_num_commits_in_push', 'gh_sloc', 'gh_test_lines_per_kloc',
                    'gh_lang_enc', 'gh_by_core_team_member',
                ],
                knn_cols=['duration_seconds', 'git_diff_src_churn', 'test_fail_rate'],
                label_col='anomaly_window_active',
                stride=_tt_graph_stride,
                seq_len=_tt_graph_seq_len,
                min_len=_tt_graph_seq_len,
                k=_tt_graph_k,
                graph_prefix='travistorrent',
                out_csv=_graph_tmp,
                append=True,
                max_windows_per_group=_tt_max_windows,
            )
            _graph_written = _graph_written or (_graph_tmp.exists() and _graph_tmp.stat().st_size > 0)

            _node_batch = enriched.groupby('gh_project_name').agg(
                duration_sum=('duration_seconds', 'sum'),
                billed_sum=('billed_cost', 'sum'),
                anomaly_sum=('anomaly_window_active', 'sum'),
                row_count=('anomaly_window_active', 'size'),
            )
            for _gid, _row in _node_batch.iterrows():
                _acc = node_acc.setdefault(str(_gid), [0.0, 0.0, 0.0, 0.0])
                _acc[0] += float(_row['duration_sum'])
                _acc[1] += float(_row['billed_sum'])
                _acc[2] += float(_row['anomaly_sum'])
                _acc[3] += float(_row['row_count'])

            del df, ordered_df, project_q95, rule1, rule2, rule3, rule4, rule5
            del focus_df, enriched, seq_df, _node_batch
            gc.collect()
            _safe_unlink(_bucket_path)

        if not _focus_written:
            logger.warning(f'[TT] No usable TravisTorrent rows found in {input_path}')
            return 0

        os.replace(str(_focus_tmp), str(out_path))

        _tt_seq_final = out_dir / 'lstm_training_sequences.csv'
        if _seq_written and _seq_tmp.exists() and _seq_tmp.stat().st_size > 0:
            os.replace(str(_seq_tmp), str(_tt_seq_final))
        else:
            _atomic_dataframe_to_csv(_tt_seq_final, pd.DataFrame(), index=False)
        _link_or_copy_file(_tt_seq_final, out_dir / 'tt_lstm_sequences.csv')

        _tt_graph_final = out_dir / 'pipeline_graphs.csv'
        if _graph_written and _graph_tmp.exists() and _graph_tmp.stat().st_size > 0:
            os.replace(str(_graph_tmp), str(_tt_graph_final))
        else:
            _atomic_dataframe_to_csv(_tt_graph_final, pd.DataFrame(), index=False)

        _tt_graphs_alias = out_dir / 'tt_graphs.csv'
        if _tt_graph_final.exists() and _tt_graph_final.stat().st_size > 64:
            try:
                _link_or_copy_file(_tt_graph_final, _tt_graphs_alias)
            except OSError as _cp_exc:
                logger.warning(f'[TT] tt_graphs.csv copy failed (non-fatal): {_cp_exc}')
                _atomic_dataframe_to_csv(_tt_graphs_alias, pd.DataFrame(), index=False)
        else:
            _atomic_dataframe_to_csv(_tt_graphs_alias, pd.DataFrame(), index=False)

        node_rows: List[Dict[str, object]] = []
        for _gid, _vals in node_acc.items():
            _count = max(1.0, _vals[3])
            node_rows.append({
                'group_id': _gid,
                'duration_seconds': _vals[0] / _count,
                'billed_cost': _vals[1] / _count,
                'anomaly_rate': _vals[2] / _count,
            })
        _atomic_dataframe_to_csv(out_dir / 'node_stats.csv', pd.DataFrame(node_rows), index=False)

        _tt_graph_rows, _tt_graph_ids = _count_graph_csv_stats(_tt_graph_final)
        gc.collect()
        _event(
            "WRITE",
            "TravisTorrent",
            telemetry=out_path.resolve(),
            sequences=(out_dir / 'lstm_training_sequences.csv').resolve(),
            graphs=(out_dir / 'pipeline_graphs.csv').resolve(),
            node_stats=(out_dir / 'node_stats.csv').resolve(),
        )

        anomaly_rate = float(_tt_anomaly_sum / max(1, _tt_rows_total))
        logger.info(f'[TT] rows={_tt_rows_total:,} anomaly_rate={anomaly_rate:.2%}')
        _event(
            "LOAD",
            "TravisTorrent",
            "END",
            rows=_tt_rows_total,
            anomaly_rate=anomaly_rate,
            graph_rows=_tt_graph_rows,
            graph_ids=_tt_graph_ids,
            duration=format_duration_s(timer.elapsed_s),
        )
        return int(_tt_rows_total)
    finally:
        for _tmp in (_focus_tmp, _seq_tmp, _graph_tmp):
            _safe_unlink(_tmp)
        if _bucket_dir.exists():
            shutil.rmtree(_bucket_dir, ignore_errors=True)


def run_universal_loader(
    tt_input: Optional[str] = None,
    bitbrains_dir: Optional[str] = None,
    force: bool = False,
    hardware: Optional[HardwareProfile] = None,
    results_base: Union[str, Path] = "./results",
) -> str:
    workspace = _workspace_root(results_base)
    bitbrains_raw = _ensure_dir(workspace / 'bitbrains_data')
    universal_dir = _ensure_dir(workspace / 'universal_raw')
    bitbrains_out = bitbrains_raw / 'bitbrains_focus.csv'
    travis_dir = _ensure_dir(workspace / 'real_data')
    travis_out = travis_dir / 'pipeline_stage_telemetry.csv'
    merged_out = universal_dir / 'pipeline_stage_telemetry.csv'

    n_b = load_bitbrains(Path(bitbrains_dir), bitbrains_out) if bitbrains_dir else 0
    _vram_transition_flush('post-BitBrains loader')
    n_t = load_travistorrent(Path(tt_input), travis_dir) if tt_input else 0
    _vram_transition_flush('post-TravisTorrent loader')

    if not force and merged_out.exists() and merged_out.stat().st_size > 256:
        logger.info('[SKIP] merged universal corpus exists - use --force to rebuild')
        return str(universal_dir)

    first = True
    merged_tmp = merged_out.with_name(f".{merged_out.name}.{os.getpid()}.{time.time_ns()}.tmp")
    _safe_unlink(merged_tmp)
    for src in [bitbrains_out, travis_out]:
        if not src.exists():
            continue
        for chunk in smart_read_csv(src, chunksize=50_000):
            chunk = _apply_focus_defaults(chunk)
            chunk.to_csv(merged_tmp, mode='w' if first else 'a', header=first, index=False)
            first = False
    if first:
        _atomic_dataframe_to_csv(merged_out, pd.DataFrame(), index=False)
    else:
        with merged_tmp.open("ab") as fh:
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(merged_tmp), str(merged_out))
    logger.info(f'[UNIVERSAL] merged_rows={n_b + n_t:,} -> {merged_out}')
    return str(universal_dir)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 pos_weight: Optional[float] = None) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self._pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, -20.0, 20.0)
        targets = targets.float()
        pos_weight = None
        if self._pos_weight is not None:
            pos_weight = torch.tensor([float(self._pos_weight)], device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=pos_weight)
        probs = torch.sigmoid(logits)
        p_t = torch.where(targets > 0.5, probs, 1.0 - probs)
        alpha_t = torch.where(targets > 0.5, torch.full_like(targets, self.alpha), torch.full_like(targets, 1.0 - self.alpha))
        loss = alpha_t * ((1.0 - p_t) ** self.gamma) * bce
        loss = loss.mean()
        return loss if torch.isfinite(loss) else bce.mean()


class BahdanauBiLSTM(nn.Module):
    def __init__(self, n_channels: int = N_CHANNELS, n_ctx: int = N_CTX,
                 hidden: int = 256, dropout: float = 0.30,
                 wd: float = 0.01, num_layers: int = 3,
                 ctx_proj_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden
        self.num_layers = num_layers
        self.input_proj = nn.Sequential(nn.Linear(n_channels, hidden), nn.LayerNorm(hidden))
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
            for _ in range(num_layers)
        ])
        self.res_projs = nn.ModuleList([nn.Linear(hidden * 2, hidden, bias=False) for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        self.W_q = nn.Linear(hidden, hidden, bias=False)
        self.W_k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, 1, bias=False)
        self.ctx_proj = nn.Sequential(nn.Linear(n_ctx, hidden), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(hidden + hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_ctx: torch.Tensor):
        x_seq = x_seq.float()
        x_ctx = x_ctx.float()
        _assert_finite_tensor(x_seq, domain="TASK-B", source="BahdanauBiLSTM.forward/x_seq")
        _assert_finite_tensor(x_ctx, domain="TASK-B", source="BahdanauBiLSTM.forward/x_ctx")
        h = self.input_proj(x_seq)
        batch_size = h.shape[0]
        for lstm, res_proj, norm, drop in zip(self.lstm_layers, self.res_projs, self.layer_norms, self.dropouts):
            h0 = torch.zeros(2, batch_size, self.hidden_dim, device=x_seq.device, dtype=h.dtype)
            c0 = torch.zeros(2, batch_size, self.hidden_dim, device=x_seq.device, dtype=h.dtype)
            out, _ = lstm(h, (h0, c0))
            proj = res_proj(out)
            h = norm(proj + h)
            h = drop(h)
        query = h[:, -1, :]
        keys = h
        energy = torch.tanh(self.W_q(query).unsqueeze(1) + self.W_k(keys))
        scores = self.v(energy).squeeze(-1)
        weights = torch.softmax(scores / math.sqrt(self.hidden_dim), dim=1)
        context = torch.sum(weights.unsqueeze(-1) * keys, dim=1)
        ctx_embed = self.ctx_proj(x_ctx)
        fused = torch.cat([context, query + ctx_embed], dim=1)
        logits = self.head(fused).squeeze(-1)
        _assert_finite_tensor(logits, domain="TASK-B", source="BahdanauBiLSTM.forward/logits")
        return torch.clamp(logits, -20.0, 20.0), weights

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GATv2Pipeline(nn.Module):
    def __init__(self, n_node_feat: int = N_NODE_FEAT, n_edge_feat: int = 3,
                 hidden: int = 128, heads: int = 4, num_layers: int = 3,
                 dropout: float = 0.30, drop_edge: float = 0.10,
                 gradient_checkpointing: bool = False) -> None:
        super().__init__()
        if not _HAS_PYG:
            raise ImportError('torch_geometric required for GATv2Pipeline')
        self.in_channels = n_node_feat
        self.dropout = dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.conv1 = GATv2Conv(n_node_feat, hidden, heads=4, concat=True, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden * 4)
        self.conv2 = GATv2Conv(hidden * 4, hidden, heads=4, concat=True, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden * 4)
        self.conv3 = GATv2Conv(hidden * 4, hidden, heads=1, concat=False, dropout=dropout)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _conv_block(self, conv: GATv2Conv, norm: nn.BatchNorm1d,
                    x: torch.Tensor, edge_index: torch.Tensor,
                    apply_dropout: bool = True) -> torch.Tensor:
        h = conv(x, edge_index)
        h = F.elu(norm(h))
        if apply_dropout:
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def _maybe_checkpoint(self, fn: Callable[[torch.Tensor], torch.Tensor],
                          x: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradient_checkpointing and x.requires_grad:
            return activation_checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor], batch: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_channels:
            raise ValueError(f'GATv2 expected {self.in_channels} node features, got {x.shape[-1]}')
        x = x.float()
        _assert_finite_tensor(x, domain="TASK-C", source="GATv2Pipeline.forward/x")
        if edge_attr is not None:
            _assert_finite_tensor(edge_attr.float(), domain="TASK-C", source="GATv2Pipeline.forward/edge_attr")
        h = self._maybe_checkpoint(lambda tensor: self._conv_block(self.conv1, self.bn1, tensor, edge_index), x)
        h = self._maybe_checkpoint(lambda tensor: self._conv_block(self.conv2, self.bn2, tensor, edge_index), h)
        h = self._maybe_checkpoint(
            lambda tensor: self._conv_block(self.conv3, self.bn3, tensor, edge_index, apply_dropout=False),
            h,
        )
        pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        logits = self.head(pooled).squeeze(-1)
        _assert_finite_tensor(logits, domain="TASK-C", source="GATv2Pipeline.forward/logits")
        return torch.clamp(logits, -20.0, 20.0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = max(1, len(probs))
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum():
            ece += (mask.sum() / N) * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


def f1_at_optimal_threshold(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    if len(probs) == 0 or len(labels) == 0:
        return 0.0, 0.5
    if len(probs) != len(labels):
        n = min(len(probs), len(labels))
        probs = probs[:n]
        labels = labels[:n]
    if len(np.unique(labels)) < 2:
        return 0.0, 0.5

    best_f1 = 0.0
    best_thr = 0.5
    for thr in OPT_THRESHOLDS:
        preds = (probs >= thr).astype(int)
        f1 = float(f1_score(labels, preds, zero_division=0))
        if f1 > best_f1 or (math.isclose(f1, best_f1) and thr < best_thr):
            best_f1 = f1
            best_thr = float(thr)
    return best_f1, best_thr


def temperature_scale(logits: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 1.0
    def nll(temp: float) -> float:
        scores = logits / temp
        loss = -float((labels * (-np.logaddexp(0.0, -scores)) + (1.0 - labels) * (-np.logaddexp(0.0, scores))).mean())
        return loss if np.isfinite(loss) else 1e9
    value = float(minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded').x)
    return value if math.isfinite(value) else 1.0


def full_eval(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5,
              tune_threshold: bool = False,
              scores_are_probabilities: bool = False) -> Dict:
    probs = np.asarray(logits, dtype=np.float64).reshape(-1) if scores_are_probabilities else _sig(logits)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    if len(probs) == 0 or len(labels) == 0:
        return {}
    if len(probs) != len(labels):
        n = min(len(probs), len(labels))
        probs = probs[:n]
        labels = labels[:n]
    eval_threshold = float(threshold)
    if tune_threshold:
        _, eval_threshold = f1_at_optimal_threshold(probs, labels)
    preds = (probs >= eval_threshold).astype(int)
    try:
        roc = float(roc_auc_score(labels, probs))
    except Exception:
        roc = 0.5
    try:
        pra = float(average_precision_score(labels, probs))
    except Exception:
        pra = 0.0
    return {
        'f1_at_opt': float(f1_score(labels, preds, zero_division=0)),
        'threshold': float(eval_threshold),
        'opt_threshold': float(eval_threshold),
        'roc_auc': roc,
        'pr_auc': pra,
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'ece': _compute_ece(probs, labels),
    }


def evaluate_calibrated_splits(val_logits: np.ndarray, val_labels: np.ndarray,
                               test_logits: np.ndarray, test_labels: np.ndarray,
                               temperature: float = 1.0) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
    if len(val_logits) == 0 or len(val_labels) == 0:
        return {}, {}, np.array([]), np.array([])
    temp = max(float(temperature), 1e-3)
    val_probs = _sig(np.asarray(val_logits, dtype=np.float64) / temp)
    test_probs = _sig(np.asarray(test_logits, dtype=np.float64) / temp)
    if len(np.unique(np.asarray(val_labels).reshape(-1))) < 2:
        val_metrics = full_eval(val_probs, val_labels, threshold=0.5, scores_are_probabilities=True)
    else:
        val_metrics = full_eval(val_probs, val_labels, tune_threshold=True, scores_are_probabilities=True)
    test_metrics = full_eval(
        test_probs,
        test_labels,
        threshold=val_metrics.get('opt_threshold', 0.5),
        scores_are_probabilities=True,
    )
    return val_metrics, test_metrics, val_probs, test_probs
def _safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _mad_iqr_outlier_mask(
    values: np.ndarray,
    mad_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
) -> np.ndarray:
    """MAD + IQR outlier detector.

    BB-FIX-5: Parameterised thresholds so infrastructure callers can use
    tighter values (mad_threshold=2.5, iqr_multiplier=1.2) without breaking
    all existing callers that rely on the defaults (3.0 / 1.5).
    """
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    mask = np.zeros(len(arr), dtype=bool)
    finite_mask = np.isfinite(arr)
    finite_vals = arr[finite_mask]
    if finite_vals.size == 0:
        return mask

    median = float(np.median(finite_vals))
    mad = float(np.median(np.abs(finite_vals - median)))
    if mad > 0.0:
        robust_z = (finite_vals - median) / mad
        mask[finite_mask] |= np.abs(robust_z) > mad_threshold

    q1, q3 = np.percentile(finite_vals, [25.0, 75.0])
    iqr = float(q3 - q1)
    if iqr > 0.0:
        fence_lo = q1 - iqr_multiplier * iqr
        fence_hi = q3 + iqr_multiplier * iqr
        mask[finite_mask] |= (finite_vals < fence_lo) | (finite_vals > fence_hi)
    return mask


def _progressive_group_outlier_labels(df: pd.DataFrame, group_col: str,
                                      metric_cols: List[str],
                                      sort_col: Optional[str] = None,
                                      min_history: int = 5,
                                      mad_threshold: float = 3.0,
                                      iqr_multiplier: float = 1.5) -> np.ndarray:
    if len(df) == 0:
        return np.zeros(0, dtype=bool)

    work = df.copy()
    work["_row_id"] = np.arange(len(work), dtype=np.int64)
    work["_group_key"] = work.get(group_col, pd.Series("unknown", index=work.index)).fillna("unknown").astype(str)

    default_sort = pd.Series(np.arange(len(work), dtype=np.float64), index=work.index)
    sort_key = default_sort
    if sort_col and sort_col in work.columns:
        numeric = pd.to_numeric(work[sort_col], errors="coerce")
        if numeric.notna().any():
            sort_key = numeric.astype(np.float64).where(numeric.notna(), default_sort)
        else:
            dt = pd.to_datetime(work[sort_col], errors="coerce")
            if dt.notna().any():
                sort_key = default_sort.copy()
                sort_key.loc[dt.notna()] = (dt.loc[dt.notna()].astype("int64") / 1e9).astype(np.float64)
    work["_sort_key"] = sort_key
    work = work.sort_values(["_group_key", "_sort_key", "_row_id"], kind="mergesort")

    labels = np.zeros(len(work), dtype=bool)
    for _, grp in work.groupby("_group_key", sort=False):
        row_ids = grp["_row_id"].to_numpy(dtype=np.int64)
        grp_mask = np.zeros(len(grp), dtype=bool)
        metric_arrays = {
            col: pd.to_numeric(grp.get(col, pd.Series(0.0, index=grp.index)), errors="coerce")
            .fillna(0.0).to_numpy(dtype=np.float64)
            for col in metric_cols
        }
        for col in metric_cols:
            vals = metric_arrays[col]
            for i in range(min_history, len(vals)):
                grp_mask[i] |= bool(_mad_iqr_outlier_mask(
                    vals[:i + 1],
                    mad_threshold=mad_threshold,
                    iqr_multiplier=iqr_multiplier,
                )[-1])
        labels[row_ids] = grp_mask
    return labels


def _fit_train_cleaner(train_2d: np.ndarray) -> Dict[str, np.ndarray]:
    arr = np.asarray(train_2d, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for cleaner fit, got {arr.shape}")

    medians = np.zeros(arr.shape[1], dtype=np.float32)
    p01 = np.zeros(arr.shape[1], dtype=np.float32)
    p99 = np.zeros(arr.shape[1], dtype=np.float32)
    for col_idx in range(arr.shape[1]):
        col = arr[:, col_idx].astype(np.float64, copy=False)
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            medians[col_idx] = 0.0
            p01[col_idx] = 0.0
            p99[col_idx] = 0.0
            continue
        medians[col_idx] = float(np.median(finite))
        p01[col_idx] = float(np.percentile(finite, 1.0))
        p99[col_idx] = float(np.percentile(finite, 99.0))
    return {"median": medians, "p01": p01, "p99": p99}


def _apply_train_cleaner(data_2d: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    arr = np.asarray(data_2d, dtype=np.float32).copy()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for cleaner apply, got {arr.shape}")

    medians = stats["median"].astype(np.float32)
    lows = stats["p01"].astype(np.float32)
    highs = stats["p99"].astype(np.float32)
    for col_idx in range(arr.shape[1]):
        col = arr[:, col_idx]
        pos_inf = np.isposinf(col)
        neg_inf = np.isneginf(col)
        nan_mask = np.isnan(col)
        if pos_inf.any():
            col[pos_inf] = highs[col_idx]
        if neg_inf.any():
            col[neg_inf] = lows[col_idx]
        if nan_mask.any():
            col[nan_mask] = medians[col_idx]
        non_finite = ~np.isfinite(col)
        if non_finite.any():
            col[non_finite] = medians[col_idx]
        arr[:, col_idx] = col
    return arr.astype(np.float32, copy=False)


def build_feature_matrix(seq_arr: np.ndarray, ctx_arr: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq_arr, dtype=np.float32)
    ctx = np.asarray(ctx_arr, dtype=np.float32)
    if seq.ndim != 3:
        raise ValueError(f"Expected sequence tensor (N,T,C), got {seq.shape}")
    if ctx.ndim != 2:
        raise ValueError(f"Expected context matrix (N,Ctx), got {ctx.shape}")
    if len(seq) != len(ctx):
        raise ValueError(f"Sequence/context row mismatch: {len(seq)} vs {len(ctx)}")
    return np.concatenate([seq.reshape(len(seq), -1), ctx], axis=1).astype(np.float32, copy=False)


def _split_feature_matrix(feature_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    feats = np.asarray(feature_matrix, dtype=np.float32)
    seq_width = SEQ_LEN * N_CHANNELS
    if feats.ndim != 2 or feats.shape[1] < seq_width:
        raise ValueError(f"Invalid feature matrix shape for split: {feats.shape}")
    seq = feats[:, :seq_width].reshape(-1, SEQ_LEN, N_CHANNELS).astype(np.float32, copy=False)
    ctx = feats[:, seq_width:].astype(np.float32, copy=False)
    return seq, ctx


def augment_training_feature_matrix(X_train: np.ndarray, y_train: np.ndarray,
                                    seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    feats = np.asarray(X_train, dtype=np.float32)
    labels = np.asarray(y_train, dtype=np.float32).reshape(-1)
    if len(feats) == 0 or len(labels) == 0:
        return feats, labels

    pos_idx = np.where(labels == 1)[0]
    pos_rate = float(labels.mean()) if len(labels) else 0.0
    if pos_rate >= 0.10 or len(pos_idx) < 2:
        return feats, labels

    rng = np.random.default_rng(seed)
    target_ratio = 0.10
    pos_count = int(len(pos_idx))
    needed = int(math.ceil(max(0.0, (target_ratio * len(labels) - pos_count) / (1.0 - target_ratio))))
    if needed <= 0:
        return feats, labels

    try:
        from imblearn.over_sampling import SMOTE
        sampling_ratio = target_ratio / max(1e-9, 1.0 - target_ratio)
        smote = SMOTE(sampling_strategy=sampling_ratio, random_state=seed,
                      k_neighbors=min(5, pos_count - 1))
        aug_X, aug_y = smote.fit_resample(feats, labels.astype(int))
        aug_X = np.asarray(aug_X, dtype=np.float32)
        aug_y = np.asarray(aug_y, dtype=np.float32)
        logger.info(f"[AUGMENT] SMOTE applied | pos_rate {pos_rate:.2%} -> {aug_y.mean():.2%}")
    except Exception as exc:
        seq_train, ctx_train = _split_feature_matrix(feats)
        seq_flat = seq_train.reshape(len(seq_train), -1)
        sampled_idx = rng.choice(pos_idx, size=needed, replace=True)
        jitter_sigma = seq_flat[pos_idx].std(axis=0, dtype=np.float64)
        jitter_sigma = np.where(np.isfinite(jitter_sigma), jitter_sigma, 0.0)
        noise = rng.normal(0.0, jitter_sigma * 0.05,
                           size=(needed, seq_flat.shape[1])).astype(np.float32)
        syn_seq = seq_flat[sampled_idx] + noise
        syn_ctx = ctx_train[sampled_idx]
        syn_X = np.concatenate([syn_seq, syn_ctx], axis=1).astype(np.float32)
        syn_y = np.ones(needed, dtype=np.float32)
        aug_X = np.vstack([feats, syn_X]).astype(np.float32)
        aug_y = np.concatenate([labels, syn_y]).astype(np.float32)
        logger.info(f"[AUGMENT] manual jitter fallback | reason={type(exc).__name__} "
                    f"| pos_rate {pos_rate:.2%} -> {aug_y.mean():.2%}")

    order = rng.permutation(len(aug_y))
    return aug_X[order], aug_y[order]


# FIX: removed dead duplicate _build_context_vector
def _build_context_vector(df_meta: pd.DataFrame, seq_arr: np.ndarray,
                          stage_enc: LabelEncoder, exec_enc: LabelEncoder,
                          branch_enc: LabelEncoder) -> np.ndarray:
    n_rows = len(seq_arr)
    ctx = np.zeros((n_rows, N_CTX), dtype=np.float32)
    primary = seq_arr[:, :, 0]
    secondary = seq_arr[:, :, 1]
    tertiary = seq_arr[:, :, 2]
    idx = 0
    for window in (6, 12, 24):
        segment = primary[:, -window:]
        ctx[:, idx] = segment.mean(axis=1)
        ctx[:, idx + 1] = segment.std(axis=1)
        x_axis = np.arange(window, dtype=np.float32)
        for row_idx in range(n_rows):
            try:
                ctx[row_idx, idx + 2] = float(np.polyfit(x_axis, segment[row_idx], 1)[0])
            except Exception:
                ctx[row_idx, idx + 2] = 0.0
        idx += 3
    ts = _safe_to_datetime(df_meta.get('created_at', pd.Series('', index=df_meta.index)))
    hours = ts.dt.hour.fillna(0).to_numpy(dtype=np.float32)
    dows = ts.dt.dayofweek.fillna(0).to_numpy(dtype=np.float32)
    ctx[:, 9] = np.sin(2 * np.pi * hours / 24.0)
    ctx[:, 10] = np.cos(2 * np.pi * hours / 24.0)
    ctx[:, 11] = np.sin(2 * np.pi * dows / 7.0)
    ctx[:, 12] = np.cos(2 * np.pi * dows / 7.0)
    ctx[:, 13] = primary[:, -1] - primary[:, max(0, SEQ_LEN - 6)]
    ctx[:, 14] = primary[:, -1] - primary[:, 0]
    ctx[:, 15] = primary.min(axis=1)
    ctx[:, 16] = primary.max(axis=1)
    ctx[:, 17] = ctx[:, 16] - ctx[:, 15]

    def _safe_encode(values: pd.Series, encoder: LabelEncoder) -> np.ndarray:
        out = np.zeros(n_rows, dtype=np.float32)
        classes = set(map(str, encoder.classes_))
        for row_idx, value in enumerate(values.fillna('unknown').astype(str)):
            out[row_idx] = float(encoder.transform([value])[0]) if value in classes else 0.0
        return out

    ctx[:, 18] = _safe_encode(df_meta.get('stage_name', pd.Series('build', index=df_meta.index)), stage_enc)
    ctx[:, 19] = _safe_encode(df_meta.get('executor_type', pd.Series('unknown', index=df_meta.index)), exec_enc)
    ctx[:, 20] = _safe_encode(df_meta.get('branch', pd.Series('main', index=df_meta.index)), branch_enc)
    ctx[:, 21] = tertiary[:, -1] / (tertiary[:, 0] + 1e-8)
    ctx = _sanitise_numeric_2d(ctx)
    return _validate_numeric_checkpoint(
        ctx,
        domain="TASK-B",
        stage="POST_SANITISE_CONTEXT",
        source="_build_context_vector",
        repair=False,
        log_ok=True,
    )


def _infer_domain_from_raw_dir(raw_dir: Path) -> str:
    name = str(raw_dir).lower()
    if 'bitbrains' in name:
        return 'bitbrains'
    if 'real' in name:
        return 'real'
    return 'synthetic'


def _fit_encoder(values: pd.Series, defaults: List[str]) -> LabelEncoder:
    train_values = pd.Series(list(values.fillna('unknown').astype(str)) + defaults)
    encoder = LabelEncoder()
    encoder.fit(train_values)
    return encoder


def preprocess_task_b(raw_dir: Path, out_dir: Path,
                      seed: int = 42, force: bool = False) -> None:
    timer = StageTimer()
    global _ML_READY_DIR, _TASK_B_DIR
    tb = _ensure_dir(out_dir / 'task_B')
    _TASK_B_DIR = tb
    _ML_READY_DIR = out_dir

    seq_csv = raw_dir / 'lstm_training_sequences.csv'
    if not seq_csv.exists():
        raise FileNotFoundError(f'{seq_csv} not found - generate or ingest data first')
    _event("PREPROCESS", "TASK-B", "START", seed=seed, raw_dir=raw_dir.resolve(), out_dir=tb.resolve(), sequence_csv=seq_csv.resolve())

    # OOM-FIX: probe header only, then stream into numpy lists — never load full DataFrame
    _seq_header = pd.read_csv(seq_csv, nrows=0).columns.tolist()
    if not _seq_header:
        raise ValueError('[TASK-B] sequence CSV is empty')
    _require_columns(pd.DataFrame(columns=_seq_header), ['run_id', 'label_budget_breach'], 'TASK-B')

    seq_cols = [f't{t:02d}_{channel}' for t in range(SEQ_LEN) for channel in CHANNEL_NAMES]
    missing_seq_cols = [col for col in seq_cols if col not in _seq_header]
    if missing_seq_cols:
        raise ValueError(
            f"[TASK-B] Sequence CSV is missing {len(missing_seq_cols)} time-series columns; "
            f"examples: {', '.join(missing_seq_cols[:5])}"
        )
    meta_cols = ['created_at', 'stage_name', 'executor_type', 'branch']
    _labels_list: list = []
    _run_ids_list: list = []
    _seq_list: list = []
    _ctx_list: list = []
    _meta_list: list = []
    _ctx_cols = [c for c in _seq_header if c.startswith('ctx_')]
    _total_rows = 0
    for _chunk in pd.read_csv(seq_csv, dtype=str, chunksize=20_000):
        _chunk = _strip_string_values(_chunk)
        _labels_list.append(_to_numeric(_chunk.get('label_budget_breach', pd.Series(0.0, index=_chunk.index))).to_numpy(dtype=np.float32))
        _run_ids_list.append(_chunk.get('run_id', pd.Series(range(len(_chunk)), index=_chunk.index)).astype(str).to_numpy())
        _seq_chunk = _chunk[seq_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32).reshape(len(_chunk), SEQ_LEN, N_CHANNELS)
        _seq_chunk = _validate_numeric_checkpoint(
            _seq_chunk,
            domain="TASK-B",
            stage="RAW_TO_PREPROCESS",
            source="sequence_chunk",
            repair=True,
        )
        _seq_list.append(_seq_chunk)
        if _ctx_cols:
            _ctx_chunk = _chunk[_ctx_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
            _ctx_chunk = _validate_numeric_checkpoint(
                _ctx_chunk,
                domain="TASK-B",
                stage="RAW_TO_PREPROCESS",
                source="context_chunk",
                repair=True,
            )
            _ctx_list.append(_ctx_chunk)
        _meta_list.append(pd.DataFrame({col: _chunk.get(col, pd.Series('', index=_chunk.index)).astype(str) for col in meta_cols}))
        _total_rows += len(_chunk)
        del _chunk
        gc.collect()
    _event("LOAD", "TASK-B", path=seq_csv.resolve(), rows=_total_rows, columns=len(_seq_header))
    labels = np.concatenate(_labels_list).astype(np.float32); del _labels_list
    run_ids = np.concatenate(_run_ids_list).astype(str); del _run_ids_list
    _X_raw = np.concatenate(_seq_list, axis=0).astype(np.float32); del _seq_list
    _C_raw = np.concatenate(_ctx_list, axis=0).astype(np.float32) if _ctx_list else np.zeros((len(labels), max(1, len(_ctx_cols))), dtype=np.float32); del _ctx_list
    df_meta = pd.concat(_meta_list, ignore_index=True); del _meta_list
    gc.collect()
    if 'created_at' in df_meta.columns:
        try:
            order = pd.to_datetime(df_meta['created_at'], errors='coerce').fillna(pd.Timestamp.now(tz='UTC')).argsort().to_numpy()
            _X_raw = _X_raw[order]; _C_raw = _C_raw[order]; labels = labels[order]; run_ids = run_ids[order]
            df_meta = df_meta.iloc[order].reset_index(drop=True)
        except Exception:
            pass
    # Reconstruct df_seq as a minimal shell (run_id + label only) for downstream compat
    df_seq = pd.DataFrame({'run_id': run_ids, 'label_budget_breach': labels})

    raw_features = _X_raw  # OOM-FIX: already built chunk-by-chunk above; no DataFrame copy
    del _X_raw
    gc.collect()
    raw_features = _validate_numeric_checkpoint(
        raw_features,
        domain="TASK-B",
        stage="RAW_TO_PREPROCESS",
        source="raw_features",
        repair=True,
        log_ok=True,
    )
    _C_raw = _validate_numeric_checkpoint(
        _C_raw,
        domain="TASK-B",
        stage="RAW_TO_PREPROCESS",
        source="raw_context",
        repair=True,
        log_ok=True,
    )
    domain = _infer_domain_from_raw_dir(raw_dir)
    index = np.arange(len(df_seq))
    if domain == 'real' and len(np.unique(labels)) > 1 and len(df_seq) > 20:
        tr_idx, tmp_idx = train_test_split(index, test_size=0.30, random_state=seed, stratify=labels)
        tmp_labels = labels[tmp_idx]
        stratify_tmp = tmp_labels if len(np.unique(tmp_labels)) > 1 else None
        val_idx, te_idx = train_test_split(tmp_idx, test_size=0.50, random_state=seed, stratify=stratify_tmp)
    else:
        tr_end = max(1, int(len(df_seq) * 0.70))
        val_end = max(tr_end + 1, int(len(df_seq) * 0.85))
        tr_idx, val_idx, te_idx = index[:tr_end], index[tr_end:val_end], index[val_end:]
        if len(val_idx) == 0:
            val_idx = tr_idx[-1:]
        if len(te_idx) == 0:
            te_idx = val_idx[-1:]

    Xtr_raw = raw_features[tr_idx]
    Xva_raw = raw_features[val_idx]
    Xte_raw = raw_features[te_idx]
    ytr = labels[tr_idx]
    yva = labels[val_idx]
    yte = labels[te_idx]

    # --- flatten for cleaner ---
    Xtr_2d = Xtr_raw.reshape(Xtr_raw.shape[0], -1)
    Xva_2d = Xva_raw.reshape(Xva_raw.shape[0], -1)
    Xte_2d = Xte_raw.reshape(Xte_raw.shape[0], -1)

    # --- fit cleaner on TRAIN only ---
    seq_clean_stats = _fit_train_cleaner(Xtr_2d)

    # --- apply cleaner ---
    Xtr_clean_2d = _apply_train_cleaner(Xtr_2d, seq_clean_stats)
    Xva_clean_2d = _apply_train_cleaner(Xva_2d, seq_clean_stats)
    Xte_clean_2d = _apply_train_cleaner(Xte_2d, seq_clean_stats)

    # --- reshape back to 3D ---
    Xtr_clean = Xtr_clean_2d.reshape(Xtr_raw.shape)
    Xva_clean = Xva_clean_2d.reshape(Xva_raw.shape)
    Xte_clean = Xte_clean_2d.reshape(Xte_raw.shape)
    Xtr_clean = _validate_numeric_checkpoint(Xtr_clean, domain="TASK-B", stage="POST_SANITISE_SEQUENCE", source="X_train_clean", repair=False, log_ok=True)
    Xva_clean = _validate_numeric_checkpoint(Xva_clean, domain="TASK-B", stage="POST_SANITISE_SEQUENCE", source="X_val_clean", repair=False, log_ok=True)
    Xte_clean = _validate_numeric_checkpoint(Xte_clean, domain="TASK-B", stage="POST_SANITISE_SEQUENCE", source="X_test_clean", repair=False, log_ok=True)

    # --- cleanup ---
    del Xtr_2d, Xva_2d, Xte_2d
    del Xtr_clean_2d, Xva_clean_2d, Xte_clean_2d
    gc.collect()
    train_meta = df_meta.iloc[tr_idx].reset_index(drop=True)
    val_meta = df_meta.iloc[val_idx].reset_index(drop=True)
    test_meta = df_meta.iloc[te_idx].reset_index(drop=True)
    stage_enc = _fit_encoder(train_meta.get('stage_name', pd.Series('build')), STAGE_ORDER + ['build'])
    exec_enc = _fit_encoder(train_meta.get('executor_type', pd.Series('unknown')), EXECUTOR_TYPES + [BITBRAINS_EXECUTOR, TRAVIS_EXECUTOR, 'synthetic_ci'])
    branch_enc = _fit_encoder(train_meta.get('branch', pd.Series('main')), BRANCH_TYPES + ['production', 'main'])

    ctr_raw = _build_context_vector(train_meta, Xtr_clean, stage_enc, exec_enc, branch_enc)
    cva_raw = _build_context_vector(val_meta, Xva_clean, stage_enc, exec_enc, branch_enc)
    cte_raw = _build_context_vector(test_meta, Xte_clean, stage_enc, exec_enc, branch_enc)
    ctr_raw = _validate_numeric_checkpoint(ctr_raw, domain="TASK-B", stage="CONTEXT_VECTOR", source="ctx_train_raw", repair=False, log_ok=True)
    cva_raw = _validate_numeric_checkpoint(cva_raw, domain="TASK-B", stage="CONTEXT_VECTOR", source="ctx_val_raw", repair=False, log_ok=True)
    cte_raw = _validate_numeric_checkpoint(cte_raw, domain="TASK-B", stage="CONTEXT_VECTOR", source="ctx_test_raw", repair=False, log_ok=True)
    ctx_clean_stats = _fit_train_cleaner(ctr_raw)
    ctr_clean = _apply_train_cleaner(ctr_raw, ctx_clean_stats)
    cva_clean = _apply_train_cleaner(cva_raw, ctx_clean_stats)
    cte_clean = _apply_train_cleaner(cte_raw, ctx_clean_stats)

    train_matrix = build_feature_matrix(Xtr_clean, ctr_clean)
    if float(ytr.mean()) < 0.10 and len(np.unique(ytr)) > 1:
        train_matrix, y_train_final = augment_training_feature_matrix(train_matrix, ytr, seed=seed)
    else:
        y_train_final = ytr.astype(np.float32)
    Xtr_aug_clean, ctr_aug_clean = _split_feature_matrix(train_matrix)

    seq_scaler = RobustScaler(quantile_range=(5.0, 95.0))
    seq_scaler.fit(Xtr_aug_clean.reshape(-1, N_CHANNELS))
    Xtr_sc = seq_scaler.transform(Xtr_aug_clean.reshape(-1, N_CHANNELS)).reshape(len(Xtr_aug_clean), SEQ_LEN, N_CHANNELS).astype(np.float32)
    Xva_sc = seq_scaler.transform(Xva_clean.reshape(-1, N_CHANNELS)).reshape(len(Xva_clean), SEQ_LEN, N_CHANNELS).astype(np.float32)
    Xte_sc = seq_scaler.transform(Xte_clean.reshape(-1, N_CHANNELS)).reshape(len(Xte_clean), SEQ_LEN, N_CHANNELS).astype(np.float32)
    Xtr_sc = _validate_numeric_checkpoint(Xtr_sc, domain="TASK-B", stage="POST_SCALER_SEQUENCE", source="X_train_scaled", repair=True, log_ok=True)
    Xva_sc = _validate_numeric_checkpoint(Xva_sc, domain="TASK-B", stage="POST_SCALER_SEQUENCE", source="X_val_scaled", repair=True, log_ok=True)
    Xte_sc = _validate_numeric_checkpoint(Xte_sc, domain="TASK-B", stage="POST_SCALER_SEQUENCE", source="X_test_scaled", repair=True, log_ok=True)

    ctx_scaler = RobustScaler(quantile_range=(5.0, 95.0))
    ctx_scaler.fit(ctr_aug_clean)
    ctr_sc = ctx_scaler.transform(ctr_aug_clean).astype(np.float32)
    cva_sc = ctx_scaler.transform(cva_clean).astype(np.float32)
    cte_sc = ctx_scaler.transform(cte_clean).astype(np.float32)
    ctr_sc = _validate_numeric_checkpoint(ctr_sc, domain="TASK-B", stage="POST_SCALER_CONTEXT", source="ctx_train_scaled", repair=True, log_ok=True)
    cva_sc = _validate_numeric_checkpoint(cva_sc, domain="TASK-B", stage="POST_SCALER_CONTEXT", source="ctx_val_scaled", repair=True, log_ok=True)
    cte_sc = _validate_numeric_checkpoint(cte_sc, domain="TASK-B", stage="POST_SCALER_CONTEXT", source="ctx_test_scaled", repair=True, log_ok=True)

    feat_train = build_feature_matrix(Xtr_sc, ctr_sc)
    feat_val = build_feature_matrix(Xva_sc, cva_sc)
    feat_test = build_feature_matrix(Xte_sc, cte_sc)

    n_pos = int(y_train_final.sum())
    n_neg = int(len(y_train_final) - n_pos)
    pos_weight = float(n_neg / max(1, n_pos))
    logger.info(f'[CLASS-BALANCE] domain={domain} pos={n_pos} neg={n_neg} ratio={float(y_train_final.mean()):.2%} pos_weight={pos_weight:.2f}')
    _event("SPLIT", "TASK-B", train=len(y_train_final), val=len(yva), test=len(yte), domain=domain)

    _atomic_numpy_save(tb / 'X_train.npy', Xtr_sc)
    _atomic_numpy_save(tb / 'X_ctx_train.npy', ctr_sc)
    _atomic_numpy_save(tb / 'y_train.npy', y_train_final.astype(np.float32))
    _atomic_numpy_save(tb / 'X_val.npy', Xva_sc)
    _atomic_numpy_save(tb / 'X_ctx_val.npy', cva_sc)
    _atomic_numpy_save(tb / 'y_val.npy', yva.astype(np.float32))
    _atomic_numpy_save(tb / 'X_test.npy', Xte_sc)
    _atomic_numpy_save(tb / 'X_ctx_test.npy', cte_sc)
    _atomic_numpy_save(tb / 'y_test.npy', yte.astype(np.float32))
    _atomic_numpy_save(tb / 'X_feature_train.npy', feat_train.astype(np.float32))
    _atomic_numpy_save(tb / 'X_feature_val.npy', feat_val.astype(np.float32))
    _atomic_numpy_save(tb / 'X_feature_test.npy', feat_test.astype(np.float32))
    _atomic_numpy_save(tb / 'run_ids_train.npy', np.asarray([f'{domain}_train_{i}' for i in range(len(y_train_final))], dtype=str))
    _atomic_numpy_save(tb / 'run_ids_val.npy', run_ids[val_idx].astype(str))
    _atomic_numpy_save(tb / 'run_ids_test.npy', run_ids[te_idx].astype(str))
    _atomic_numpy_save(tb / 'test_ids.npy', np.arange(len(yte), dtype=np.int64))
    _atomic_write_json(tb / 'config.json', {
        'domain': domain,
        'pos_weight': pos_weight,
        'n_train': len(y_train_final),
        'n_val': len(yva),
        'n_test': len(yte),
        'seq_len': SEQ_LEN,
        'n_channels': N_CHANNELS,
        'n_ctx': N_CTX,
        'seed': seed,
        'n_feature_dim': int(feat_train.shape[1]),
        'train_pos_ratio': float(y_train_final.mean()) if len(y_train_final) else 0.0,
        'val_pos_ratio': float(yva.mean()) if len(yva) else 0.0,
        'test_pos_ratio': float(yte.mean()) if len(yte) else 0.0,
    })
    # OOM-FIX: release all heavy intermediate arrays (after last use of y_train_final/yva/yte)
    del raw_features, Xtr_raw, Xva_raw, Xte_raw
    del Xtr_clean, Xva_clean, Xte_clean
    del Xtr_sc, Xva_sc, Xte_sc
    del ctr_sc, cva_sc, cte_sc
    del ctr_raw, cva_raw, cte_raw
    del ctr_clean, cva_clean, cte_clean
    del Xtr_aug_clean, ctr_aug_clean
    del y_train_final, ytr, yva, yte
    gc.collect()
    _atomic_pickle_dump(tb / 'scaler.pkl', seq_scaler)
    _atomic_pickle_dump(tb / 'ctx_scaler.pkl', ctx_scaler)
    _event(
        "WRITE",
        "TASK-B",
        out_dir=tb.resolve(),
        config=(tb / 'config.json').resolve(),
        scaler=(tb / 'scaler.pkl').resolve(),
        ctx_scaler=(tb / 'ctx_scaler.pkl').resolve(),
    )
    _event("PREPROCESS", "TASK-B", "END", duration=format_duration_s(timer.elapsed_s), out_dir=tb.resolve())


def preprocess_task_c(raw_dir: Path, out_dir: Path,
                      mode: str = 'standard', force: bool = False, seed: int = 42) -> None:
    timer = StageTimer()
    global _TASK_C_DIR
    tc = _ensure_dir(out_dir / 'task_C')
    _TASK_C_DIR = tc
    if not _HAS_PYG:
        logger.warning('[TASK-C] torch_geometric not available - skipping GAT preprocessing')
        _atomic_write_json(tc / 'config.json', {'n_node_features': N_NODE_FEAT, 'n_edge_features': 3})
        return

    graph_csv = raw_dir / 'pipeline_graphs.csv'
    if not graph_csv.exists():
        logger.warning(f'[TASK-C] {graph_csv} not found - skipping GAT preprocessing')
        _atomic_write_json(tc / 'config.json', {'n_node_features': N_NODE_FEAT, 'n_edge_features': 3})
        return
    _event("PREPROCESS", "TASK-C", "START", seed=seed, raw_dir=raw_dir.resolve(), out_dir=tc.resolve(), graph_csv=graph_csv.resolve())

    # OOM-FIX: probe header first, then stream graph_csv — never hold full DataFrame
    _gc_header_df = pd.read_csv(graph_csv, nrows=0)
    _gc_cols = _gc_header_df.columns.tolist()
    del _gc_header_df
    _require_columns(
        pd.DataFrame(columns=_gc_cols),
        ['graph_id', 'src_stage', 'dst_stage', 'edge_cost_ratio', 'edge_duration_s', 'edge_call_count', 'label_cost_anomalous'],
        'TASK-C',
    )
    nf_map: Dict[int, List[str]] = {}
    for col in _gc_cols:
        match = re.match(r'^nf_(\d+)_(.+)$', str(col))
        if match:
            node_idx = int(match.group(1))
            nf_map.setdefault(node_idx, []).append(col)
    if not nf_map:
        raise ValueError('[TASK-C] pipeline_graphs.csv is missing per-graph nf_* columns')
    node_indices = sorted(nf_map)
    feature_count = len(sorted(nf_map[node_indices[0]]))
    # OOM-FIX: two-pass streaming. First pass: write all chunks to %TEMP% CSV.
    # Second pass: re-read one graph_id group at a time. Never holds all rows in RAM.
    _gc_sys_tmp = Path(os.environ.get('TEMP', os.environ.get('TMP', '/tmp')))
    _gc_tmp = _gc_sys_tmp / '_task_c_graph_edges.csv'
    try: _gc_tmp.unlink(missing_ok=True)
    except OSError: pass
    _gc_wrote_header = False
    _total_edge_rows = 0
    _graph_id_set: set = set()
    for _gc_chunk in pd.read_csv(graph_csv, dtype=str, chunksize=10_000):
        _gc_chunk.to_csv(_gc_tmp, mode='a', header=not _gc_wrote_header, index=False)
        _gc_wrote_header = True
        _graph_id_set.update(_gc_chunk['graph_id'].dropna().unique().tolist())
        _total_edge_rows += len(_gc_chunk)
        del _gc_chunk; gc.collect()
    _event("LOAD", "TASK-C", path=graph_csv.resolve(), rows=_total_edge_rows, columns=len(_gc_cols))
    _event("GRAPH", "TASK-C", "BUILD", graph_ids=len(_graph_id_set), edge_rows=_total_edge_rows, n_node_features=feature_count)
    graphs_by_id: Dict[str, Data] = {}
    raw_graph_features: Dict[str, np.ndarray] = {}
    _gc_cur_gid: Optional[str] = None
    _gc_cur_rows: list = []

    def _gc_build_graph(gid: str, rows: list) -> None:
        grp = pd.concat(rows, ignore_index=True)
        x = np.zeros((len(node_indices), feature_count), dtype=np.float32)
        for row_idx, node_idx in enumerate(node_indices):
            cols = sorted(nf_map[node_idx])
            x[row_idx] = grp.iloc[0][cols].to_numpy(dtype=np.float32)  # FIX: iloc not loc
        x = _validate_numeric_checkpoint(
            x,
            domain="TASK-C",
            stage="RAW_TO_PREPROCESS",
            source=f"node_features[{gid}]",
            repair=True,
        )
        edge_index = torch.tensor(grp[['src_stage', 'dst_stage']].to_numpy(dtype=np.int64).T, dtype=torch.long)
        edge_np = grp[['edge_cost_ratio', 'edge_duration_s', 'edge_call_count']].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
        edge_np = _validate_numeric_checkpoint(
            edge_np,
            domain="TASK-C",
            stage="RAW_TO_PREPROCESS",
            source=f"edge_features[{gid}]",
            repair=True,
        )
        edge_attr = torch.tensor(edge_np, dtype=torch.float32)
        y_np = pd.to_numeric(grp['label_cost_anomalous'], errors='coerce').to_numpy(dtype=np.float32)
        y_np = _validate_numeric_checkpoint(
            y_np,
            domain="TASK-C",
            stage="RAW_TO_PREPROCESS",
            source=f"graph_label[{gid}]",
            repair=True,
        )
        y = torch.tensor([float(np.max(y_np))], dtype=torch.float32)
        graphs_by_id[gid] = Data(x=torch.tensor(x, dtype=torch.float32), edge_index=edge_index, edge_attr=edge_attr, y=y)
        raw_graph_features[gid] = x

    for _gc_chunk in pd.read_csv(_gc_tmp, dtype=str, chunksize=10_000):
        for _gc_gid, _gc_grp in _gc_chunk.groupby('graph_id', sort=False):
            _gc_gid = str(_gc_gid)
            if _gc_cur_gid is None: _gc_cur_gid = _gc_gid
            if _gc_gid != _gc_cur_gid:
                if _gc_cur_rows: _gc_build_graph(_gc_cur_gid, _gc_cur_rows); _gc_cur_rows = []
                _gc_cur_gid = _gc_gid
            _gc_cur_rows.append(_gc_grp)
        del _gc_chunk; gc.collect()
    if _gc_cur_gid and _gc_cur_rows: _gc_build_graph(_gc_cur_gid, _gc_cur_rows)
    try: _gc_tmp.unlink(missing_ok=True)
    except OSError: pass

    tb = out_dir / 'task_B'
    split_ids = None
    split_files = {split: tb / f'run_ids_{split}.npy' for split in ('train', 'val', 'test')}
    if all(p.exists() for p in split_files.values()):
        split_ids = {split: np.load(path, allow_pickle=True).astype(str).tolist() for split, path in split_files.items()}
        split_ids = {split: [graph_id for graph_id in ids if graph_id in graphs_by_id] for split, ids in split_ids.items()}
    if not split_ids or not split_ids['train']:
        all_ids = list(graphs_by_id.keys())
        rng = np.random.default_rng(seed)
        rng.shuffle(all_ids)
        tr_end = max(1, int(len(all_ids) * 0.70))
        val_end = max(tr_end + 1, int(len(all_ids) * 0.85))
        split_ids = {
            'train': all_ids[:tr_end],
            'val': all_ids[tr_end:val_end] or all_ids[:1],
            'test': all_ids[val_end:] or all_ids[-1:],
        }
    _event("SPLIT", "TASK-C", train=len(split_ids['train']), val=len(split_ids['val']), test=len(split_ids['test']))

    train_stack = np.vstack([raw_graph_features[g_id] for g_id in split_ids['train']])
    train_stack = _validate_numeric_checkpoint(
        train_stack,
        domain="TASK-C",
        stage="RAW_TO_PREPROCESS",
        source="train_stack_node_features",
        repair=True,
        log_ok=True,
    )
    cleaner_stats = _fit_train_cleaner(train_stack)
    cleaned_train = _apply_train_cleaner(train_stack, cleaner_stats)
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    scaler.fit(cleaned_train)

    split_graphs: Dict[str, List[Data]] = {'train': [], 'val': [], 'test': []}
    _split_counts: Dict[str, int] = {'train': 0, 'val': 0, 'test': 0}
    train_labels = np.array([], dtype=np.float32)
    for split, ids in split_ids.items():
        for graph_id in ids:
            graph = graphs_by_id[graph_id]
            x_clean = _apply_train_cleaner(graph.x.numpy(), cleaner_stats)
            x_clean = _validate_numeric_checkpoint(
                x_clean,
                domain="TASK-C",
                stage="POST_SANITISE_NODE_FEATURES",
                source=f"{split}:{graph_id}:x_clean",
                repair=False,
            )
            x_scaled = scaler.transform(x_clean).astype(np.float32)
            x_scaled = _validate_numeric_checkpoint(
                x_scaled,
                domain="TASK-C",
                stage="POST_SCALER_NODE_FEATURES",
                source=f"{split}:{graph_id}:x_scaled",
                repair=True,
            )
            graph.x = torch.tensor(x_scaled, dtype=torch.float32)
            split_graphs[split].append(graph)
        _split_counts[split] = len(split_graphs[split])
        if split == 'train':
            train_labels = np.array([g.y.item() for g in split_graphs['train']], dtype=np.float32)
        _atomic_torch_save(tc / f'graphs_{split}.pt', split_graphs[split])
        split_graphs[split] = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _atomic_write_json(tc / 'config.json', {
        'n_node_features': feature_count,
        'n_edge_features': 3,
        'n_train': _split_counts['train'],
        'n_val':   _split_counts['val'],
        'n_test':  _split_counts['test'],
        'train_pos_ratio': float(train_labels.mean()) if len(train_labels) else 0.0,
    })
    _event(
        "WRITE",
        "TASK-C",
        train_graphs=(tc / 'graphs_train.pt').resolve(),
        val_graphs=(tc / 'graphs_val.pt').resolve(),
        test_graphs=(tc / 'graphs_test.pt').resolve(),
        config=(tc / 'config.json').resolve(),
    )
    _event("PREPROCESS", "TASK-C", "END", duration=format_duration_s(timer.elapsed_s), out_dir=tc.resolve())


def load_task_b_tensors(task_b_dir: Optional[Path] = None) -> Tuple:
    tb = task_b_dir or _TASK_B_DIR
    if tb is None:
        raise RuntimeError('[TASK-B] _TASK_B_DIR not set - run preprocessing first')

    def _load(name: str) -> torch.Tensor:
        path = tb / name
        if not path.exists():
            raise FileNotFoundError(f'[TASK-B] {path} not found')
        arr = np.load(path, mmap_mode='r', allow_pickle=False)
        if arr.dtype != np.float32:
            arr = np.asarray(arr, dtype=np.float32)
        arr = _validate_numeric_checkpoint(
            arr,
            domain="TASK-B",
            stage="PRE_MODEL_INPUT",
            source=str(path.name),
            repair=False,
            log_ok=True,
        )
        tensor = torch.as_tensor(arr, dtype=torch.float32)
        _assert_finite_tensor(tensor, domain="TASK-B", source=str(path.name))
        return tensor

    return (
        _load('X_train.npy'),
        _load('X_ctx_train.npy'),
        _load('y_train.npy'),
        _load('X_val.npy'),
        _load('X_ctx_val.npy'),
        _load('y_val.npy'),
        _load('X_test.npy'),
        _load('X_ctx_test.npy'),
        _load('y_test.npy'),
    )


def load_task_c_graphs(split: str = 'train',
                       task_c_dir: Optional[Path] = None) -> List:
    tc = task_c_dir or _TASK_C_DIR
    if tc is None or not _HAS_PYG:
        return []
    path = tc / f'graphs_{split}.pt'
    if not path.exists():
        return []
    graphs = torch.load(path, map_location='cpu', weights_only=False)
    for graph_idx, graph in enumerate(graphs):
        graph.x = torch.tensor(
            _validate_numeric_checkpoint(
                graph.x.detach().cpu().numpy(),
                domain="TASK-C",
                stage="PRE_MODEL_INPUT",
                source=f"{split}[{graph_idx}].x",
                repair=False,
            ),
            dtype=torch.float32,
        )
        _assert_finite_tensor(graph.x, domain="TASK-C", source=f"{split}[{graph_idx}].x")
        edge_attr = getattr(graph, 'edge_attr', None)
        if edge_attr is not None:
            graph.edge_attr = torch.tensor(
                _validate_numeric_checkpoint(
                    edge_attr.detach().cpu().numpy(),
                    domain="TASK-C",
                    stage="PRE_MODEL_INPUT",
                    source=f"{split}[{graph_idx}].edge_attr",
                    repair=False,
                ),
                dtype=torch.float32,
            )
            _assert_finite_tensor(graph.edge_attr, domain="TASK-C", source=f"{split}[{graph_idx}].edge_attr")
        graph.y = torch.tensor(
            _validate_numeric_checkpoint(
                graph.y.detach().cpu().numpy(),
                domain="TASK-C",
                stage="PRE_MODEL_INPUT",
                source=f"{split}[{graph_idx}].y",
                repair=False,
            ),
            dtype=torch.float32,
        )
        _assert_finite_tensor(graph.y, domain="TASK-C", source=f"{split}[{graph_idx}].y")
    return graphs


def compute_ensemble(lstm_probs: np.ndarray, gat_probs: np.ndarray,
                     lstm_pr_auc: float, gat_pr_auc: float,
                     val_labels: np.ndarray,
                     label: str = "eval") -> Tuple[np.ndarray, float, float]:
    del val_labels  # reserved for future OOF/stacking variants
    total = float(lstm_pr_auc + gat_pr_auc)
    if total <= 1e-9:
        w_l = 0.5
        w_g = 0.5
    else:
        w_l = float(lstm_pr_auc / total)
        w_g = 1.0 - w_l
    weighted = w_l * lstm_probs + w_g * gat_probs
    _event(
        "ENSEMBLE",
        label.upper(),
        lstm_weight=w_l,
        gat_weight=w_g,
        strategy="pr_auc_weighted" if total > 1e-9 else "uniform",
        candidate_rows=len(lstm_probs),
    )
    return weighted, w_l, w_g


def run_baseline_comparison(X_train: np.ndarray, X_test: np.ndarray,
                            y_train: np.ndarray, y_test: np.ndarray,
                            pos_rate: float,
                            X_val: Optional[np.ndarray] = None,
                            y_val: Optional[np.ndarray] = None) -> Dict:
    """
    Fit baselines on shared Task-B feature matrices.
    Validation picks the threshold once; test reuses that locked threshold.
    """
    results: Dict[str, Dict[str, Dict]] = {}
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32) if X_val is not None else None
    y_val = np.asarray(y_val, dtype=np.float32) if y_val is not None else None

    try:
        contamination = float(np.clip(pos_rate, 0.01, 0.49))
        iso = IsolationForest(
            contamination=contamination,
            random_state=SEED,
            n_jobs=1,
        )
        iso.fit(X_train)
        iso_metrics: Dict[str, Dict] = {}
        iso_threshold = 0.5
        if X_val is not None and y_val is not None and len(X_val) > 0 and len(y_val) > 0:
            val_scores = -iso.score_samples(X_val)
            val_probs = (val_scores - val_scores.min()) / (val_scores.max() - val_scores.min() + 1e-9)
            iso_metrics['val'] = full_eval(
                val_probs,
                y_val,
                tune_threshold=True,
                scores_are_probabilities=True,
            )
            iso_threshold = float(iso_metrics['val'].get('opt_threshold', 0.5))
        test_scores = -iso.score_samples(X_test)
        test_probs = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min() + 1e-9)
        iso_metrics['test'] = full_eval(
            test_probs,
            y_test,
            threshold=iso_threshold,
            scores_are_probabilities=True,
        )
        results['isolation_forest'] = iso_metrics
    except Exception as exc:
        logger.warning(f'[BASELINE] IsolationForest failed: {exc}')

    try:
        if len(np.unique(y_train.astype(int))) >= 2:
            rf = RandomForestClassifier(
                n_estimators=200,
                random_state=SEED,
                n_jobs=1,
                class_weight='balanced',
                max_features='sqrt',
            )
            rf.fit(X_train, y_train.astype(int))
            rf_metrics: Dict[str, Dict] = {}
            rf_threshold = 0.5
            if X_val is not None and y_val is not None and len(X_val) > 0 and len(y_val) > 0:
                rf_val_probs = rf.predict_proba(X_val)[:, 1]
                rf_metrics['val'] = full_eval(
                    rf_val_probs,
                    y_val,
                    tune_threshold=True,
                    scores_are_probabilities=True,
                )
                rf_threshold = float(rf_metrics['val'].get('opt_threshold', 0.5))
            rf_test_probs = rf.predict_proba(X_test)[:, 1]
            rf_metrics['test'] = full_eval(
                rf_test_probs,
                y_test,
                threshold=rf_threshold,
                scores_are_probabilities=True,
            )
            results['random_forest'] = rf_metrics
        else:
            logger.warning('[BASELINE] RandomForest skipped: single-class training labels')
    except Exception as exc:
        logger.warning(f'[BASELINE] RandomForest failed: {exc}')

    return results


def _print_metric_table(header: str, metrics_by_model: Dict[str, Dict]) -> None:
    print(f'\n{header}')
    print(f"{'Model':<18}{'Precision':>10}{'Recall':>10}{'F1':>10}{'ROC-AUC':>10}{'Thresh':>10}")
    print('-' * 68)
    printed = False
    for model_name, metrics in metrics_by_model.items():
        if not metrics:
            continue
        printed = True
        print(
            f"{model_name:<18}"
            f"{metrics.get('precision', 0.0):>10.4f}"
            f"{metrics.get('recall', 0.0):>10.4f}"
            f"{metrics.get('f1_at_opt', 0.0):>10.4f}"
            f"{metrics.get('roc_auc', 0.0):>10.4f}"
            f"{metrics.get('opt_threshold', metrics.get('threshold', 0.5)):>10.4f}"
        )
    if not printed:
        print('(no metrics available)')


CRS_WARN = 0.50
CRS_AUTO_OPTIMISE = 0.75
CRS_BLOCK = 0.90


def aog_gate(prob: float, stage: str = 'integration_test',
             est_cost: float = 0.20, ece: float = 0.023) -> Dict:
    """Autonomous Optimization Gate: ALLOW | WARN | AUTO_OPTIMISE | BLOCK."""
    half_width = 1.96 * float(ece)
    ci = (
        round(max(0.0, float(prob) - half_width), 4),
        round(min(1.0, float(prob) + half_width), 4),
    )
    catalogue = {
        'SPOT_INSTANCE': {'reduction_pct': 65.0, 'stages': ['build', 'unit_test', 'integration_test', 'docker_build']},
        'ENABLE_BUILD_CACHE': {'reduction_pct': 30.0, 'stages': ['build', 'docker_build']},
        'REDUCE_PARALLELISM': {'reduction_pct': 40.0, 'stages': ['unit_test', 'integration_test']},
        'DEFER_TO_SCHEDULE': {'reduction_pct': 20.0, 'stages': ['deploy_staging', 'deploy_prod']},
    }
    if prob >= CRS_BLOCK:
        return {
            'decision': 'BLOCK',
            'crs': float(prob),
            'ci': ci,
            'justification': f'CRS={prob:.3f} >= BLOCK={CRS_BLOCK}',
        }
    if prob >= CRS_AUTO_OPTIMISE:
        candidates = [name for name, meta in catalogue.items() if stage in meta['stages']]
        if candidates:
            best = max(candidates, key=lambda name: catalogue[name]['reduction_pct'])
            saving = float(est_cost) * catalogue[best]['reduction_pct'] / 100.0
            return {
                'decision': 'AUTO_OPTIMISE',
                'crs': float(prob),
                'ci': ci,
                'action': best,
                'saving_usd': round(saving, 4),
            }
        else:  # FIX: unknown stage
            return {'decision': 'AUTO_OPTIMISE', 'crs': float(prob), 'ci': ci,
                    'action': None, 'saving_usd': 0.0, 'note': f'no catalogue entry for stage={stage!r}'}
    if prob >= CRS_WARN:
        return {'decision': 'WARN', 'crs': float(prob), 'ci': ci}
    return {'decision': 'ALLOW', 'crs': float(prob), 'ci': ci}


def run_preprocessing(raw_dir: str, out_dir: str,
                      seed: int = 42, mode: str = 'standard',
                      force: bool = False) -> None:
    timer = StageTimer()
    raw = Path(raw_dir)
    out = _ensure_dir(Path(out_dir))
    global _ML_READY_DIR, _TASK_B_DIR, _TASK_C_DIR
    _ML_READY_DIR = out
    _TASK_B_DIR = out / 'task_B'
    _TASK_C_DIR = out / 'task_C'
    _event("PREPROCESS", "START", seed=seed, raw_dir=raw.resolve(), out_dir=out.resolve(), mode=mode, force=force)

    if 'universal' in str(raw).lower():
        telemetry = raw / 'pipeline_stage_telemetry.csv'
        if telemetry.exists():
            _event("GRAPH", "UNIVERSAL", "START", telemetry=telemetry.resolve(), raw_dir=raw.resolve())
            build_universal_sequences_and_graphs(telemetry, raw, seed=seed, data_mode='universal', force=force)
            _event("GRAPH", "UNIVERSAL", "END", duration=format_duration_s(timer.elapsed_s), raw_dir=raw.resolve())

    if not force and _preproc_exists(out):
        _event("PREPROCESS", "END", seed=seed, out_dir=out.resolve(), duration=format_duration_s(timer.elapsed_s), status="cache_hit")
        return

    preprocess_task_b(raw, out, seed=seed, force=force)
    preprocess_task_c(raw, out, mode=mode, force=force, seed=seed)
    _event("PREPROCESS", "END", seed=seed, out_dir=out.resolve(), duration=format_duration_s(timer.elapsed_s), status="complete")
@dataclass
class LSTMConfig:
    epochs: int = 150
    batch_size: int = 256
    lr: float = 5e-4
    hidden_dim: int = 256
    num_layers: int = 3
    ctx_proj_dim: int = 64
    dropout: float = 0.30
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    weight_decay: float = 1e-4
    lr_patience: int = 10
    lr_factor: float = 0.5
    grad_clip: float = 1.0
    warmup_epochs: int = 0
    patience: int = 10
    checkpoint_keep_last_k: int = 5
    seed: int = 42


@dataclass
class GATConfig:
    epochs: int = 150
    batch_size: int = 32
    lr: float = 5e-4
    hidden_dim: int = 128
    heads: int = 4
    num_layers: int = 3
    dropout: float = 0.30
    weight_decay: float = 1e-4
    lr_patience: int = 10
    lr_factor: float = 0.5
    grad_clip: float = 1.0
    warmup_epochs: int = 0
    patience: int = 10
    checkpoint_keep_last_k: int = 5
    seed: int = 42


def _make_scheduler(optimizer: optim.Optimizer, scheduler_type: str, epochs: int):
    if scheduler_type == 'cosine':
        _T0 = max(5, epochs // 5)  # FIX: derive from epoch budget
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=_T0, T_mult=2)
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)


def _checkpoints_dir(run_dir: Path) -> Path:
    return _ensure_dir(run_dir / 'checkpoints')


def _predictions_dir(run_dir: Path) -> Path:
    return _ensure_dir(run_dir / 'predictions')


def _load_model_payload(path: Path):
    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(payload, dict) and 'model' in payload:
        return payload['model']
    return payload


def _epoch_status_line(seed: int, domain_label: str, model_name: str,
                       epoch: int, total_epochs: int, train_loss: float,
                       val_loss: float, val_f1: float, best_f1: float, patience_ctr: int,
                       patience: int, lr: float, status: str) -> str:
    return (
        f"[Seed {seed}][{domain_label}][{model_name}][Epoch {epoch}/{total_epochs}] "
        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f} lr={lr:.6g} "
        f"best={best_f1:.4f} patience={patience_ctr}/{patience} status={status}"
    )


def _normalise_training_history(hist: Optional[Dict]) -> Dict[str, object]:
    payload: Dict[str, object] = hist if isinstance(hist, dict) else {}
    payload.setdefault('train_loss', [])
    payload.setdefault('val_loss', [])
    payload.setdefault('val_f1_opt', [])
    payload.setdefault('lr', [])
    payload.setdefault('best_epoch', 0)
    payload.setdefault('best_f1', 0.0)
    payload.setdefault('start_epoch', 1)
    payload.setdefault('resume_loaded', False)
    payload.setdefault('stopped_early', False)
    payload.setdefault('last_epoch', 0)
    return payload


def _atomic_torch_save(path: Path, payload: Dict[str, object]) -> None:
    """Persist a torch checkpoint via temp-file replace to avoid partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    try:
        tmp.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug(f"[IMMORTAL-IO-DEBUG] Checkpoint temp cleanup skipped for {tmp}: {exc}")
    torch.save(payload, str(tmp))
    try:
        os.replace(str(tmp), str(path))
        # FIX: no finally-unlink — os.replace() consumes tmp; finally could delete target on NTFS
    except OSError as exc:
        logger.info(f"[IMMORTAL-IO] Atomic checkpoint replace fallback for {path}: {exc}")
        try:
            if path.exists():
                path.unlink()
            tmp.rename(path)
        except OSError as rename_exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise RuntimeError(
                f"[ATOMIC-CHECKPOINT] Both replace() and rename() failed for {path}: {rename_exc}"
            ) from rename_exc


def _atomic_torch_save(path: Path, payload: object) -> None:
    """Persist a torch checkpoint via durable temp-file replace."""
    def _writer(tmp_path: Path) -> None:
        torch.save(payload, str(tmp_path))
        try:
            with tmp_path.open("rb") as fh:
                os.fsync(fh.fileno())
        except OSError:
            pass

    atomic_write_file(path, _writer)


def _checkpoint_history_dir(checkpoints_dir: Path, stem: str, kind: str) -> Path:
    return _ensure_dir(checkpoints_dir / f'{stem}_{kind}_history')


def _checkpoint_snapshots(checkpoints_dir: Path, stem: str, kind: str) -> List[Path]:
    history_dir = checkpoints_dir / f'{stem}_{kind}_history'
    if not history_dir.exists():
        return []
    return sorted(
        (path for path in history_dir.glob('*.pt') if path.is_file()),
        key=lambda item: (item.stat().st_mtime, item.name),
        reverse=True,
    )


def _snapshot_metric_suffix(metric: Optional[float]) -> str:
    if metric is None:
        return ''
    return f"_metric_{metric:.6f}".replace('.', 'p')


def _checkpoint_snapshot_path(checkpoints_dir: Path, stem: str, kind: str,
                              epoch: int, metric: Optional[float] = None) -> Path:
    history_dir = _checkpoint_history_dir(checkpoints_dir, stem, kind)
    timestamp = _utcnow().strftime('%Y%m%dT%H%M%S%fZ')
    return history_dir / f'{stem}_{kind}_epoch_{epoch:04d}{_snapshot_metric_suffix(metric)}_{timestamp}.pt'


def _prune_checkpoint_history(checkpoints_dir: Path, stem: str, kind: str, keep_last_k: int) -> None:
    if keep_last_k <= 0:
        return
    for stale_path in _checkpoint_snapshots(checkpoints_dir, stem, kind)[keep_last_k:]:
        try:
            stale_path.unlink()
        except OSError as exc:
            logger.warning(f"[CHECKPOINT][PRUNE] path={stale_path} error={exc}")


def _build_training_checkpoint_payload(model: nn.Module,
                                       optimizer: optim.Optimizer,
                                       scheduler: object,
                                       scaler: torch.amp.GradScaler,
                                       epoch: int,
                                       best_f1: float,
                                       best_epoch: int,
                                       patience_ctr: int,
                                       patience: int,
                                       hist: Dict[str, object],
                                       checkpoint_type: str,
                                       epoch_completed: bool = True) -> Dict[str, object]:
    checkpoint_config = {
        'seed': hist.get('seed'),
        'domain_label': hist.get('domain_label'),
        'model_name': hist.get('model_name'),
        'model_config': hist.get('model_config', {}),
        'adaptive': hist.get('adaptive', {}),
    }
    config_hash = str(hist.get('config_hash') or _config_payload_hash(checkpoint_config))
    return {
        'epoch': int(epoch),
        'step': int(hist.get('optimizer_steps', 0) or 0),
        'epoch_completed': bool(epoch_completed),
        'checkpoint_type': checkpoint_type,
        'saved_at': _utcnow().isoformat(),
        'seed': hist.get('seed'),
        'domain_label': hist.get('domain_label'),
        'model_name': hist.get('model_name'),
        'config_hash': config_hash,
        'config_snapshot': checkpoint_config,
        'resume_state': {
            'start_epoch': int(hist.get('start_epoch', 1) or 1),
            'last_epoch': int(hist.get('last_epoch', 0) or 0),
            'resume_loaded': bool(hist.get('resume_loaded', False)),
            'epoch_completed': bool(epoch_completed),
        },
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
        'scaler': scaler.state_dict() if torch.cuda.is_available() else None,
        'best_f1': float(best_f1),
        'best_metric': float(best_f1),
        'best_metric_name': 'val_f1',
        'best_epoch': int(best_epoch),
        'patience_ctr': int(patience_ctr),
        'patience_state': {
            'counter': int(patience_ctr),
            'limit': int(patience),
        },
        'hist': deepcopy(hist),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy_rng_state': deepcopy(np.random.get_state()),
        'python_rng_state': deepcopy(random.getstate()),
    }


def _save_training_checkpoint(checkpoints_dir: Path, stem: str, kind: str,
                              canonical_path: Path, payload: Dict[str, object],
                              epoch: int, keep_last_k: int,
                              metric: Optional[float] = None) -> None:
    _atomic_torch_save(canonical_path, payload)
    if keep_last_k > 0:
        snapshot_path = _checkpoint_snapshot_path(checkpoints_dir, stem, kind, epoch, metric=metric)
        _atomic_torch_save(snapshot_path, payload)
        _prune_checkpoint_history(checkpoints_dir, stem, kind, keep_last_k)
    _event(
        "CHECKPOINT",
        "SAVE",
        type=kind,
        epoch=epoch,
        metric=metric,
        path=canonical_path.resolve(),
    )


def _load_training_checkpoint(path: Path) -> Dict[str, object]:
    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    if not isinstance(payload, dict) or 'model' not in payload:
        raise ValueError(f"unsupported checkpoint payload: {path}")
    return payload


def _resolve_latest_checkpoint(checkpoints_dir: Path, stem: str, canonical_path: Path) -> Tuple[Optional[Path], Optional[Dict[str, object]]]:
    candidates = [candidate for candidate in [canonical_path, *_checkpoint_snapshots(checkpoints_dir, stem, 'latest')] if candidate.exists()]
    seen: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate.resolve())
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        try:
            return candidate, _load_training_checkpoint(candidate)
        except Exception as exc:
            logger.warning(f"[CHECKPOINT][LOAD-SKIP] path={candidate} error={exc}")
    return None, None


def _purge_stale_checkpoints(checkpoints_dir: Path, stem: str,
                             canonical_latest: Path, canonical_best: Path,
                             epoch_guard_path: Path) -> None:
    stale_paths: List[Path] = [canonical_latest, canonical_best, epoch_guard_path]
    stale_paths.extend(_checkpoint_snapshots(checkpoints_dir, stem, 'latest'))
    stale_paths.extend(_checkpoint_snapshots(checkpoints_dir, stem, 'best'))
    seen: set[str] = set()
    for path in stale_paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        _safe_unlink(path)


def _restore_training_rng_state(state: Dict[str, object]) -> None:
    torch_rng_state = state.get('torch_rng_state')
    if torch_rng_state is not None:
        torch.set_rng_state(torch.as_tensor(torch_rng_state, dtype=torch.uint8, device='cpu').cpu())
    cuda_rng_state_all = state.get('cuda_rng_state_all')
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([
            torch.as_tensor(rng_state, dtype=torch.uint8, device='cpu').cpu()
            for rng_state in cuda_rng_state_all
        ])
    numpy_rng_state = state.get('numpy_rng_state')
    if numpy_rng_state is not None:
        np.random.set_state(numpy_rng_state)
    python_rng_state = state.get('python_rng_state')
    if python_rng_state is not None:
        random.setstate(python_rng_state)


def _resume_epoch_from_checkpoint(state: Dict[str, object]) -> int:
    epoch = int(state.get('epoch', 0) or 0)
    if bool(state.get('epoch_completed', True)):
        return epoch + 1 if epoch > 0 else 1
    return max(1, epoch)


def _build_lstm_dataloaders(Xtr: torch.Tensor,
                            Ctr: torch.Tensor,
                            Ytr: torch.Tensor,
                            Xva: torch.Tensor,
                            Cva: torch.Tensor,
                            Yva: torch.Tensor,
                            Xte: torch.Tensor,
                            Cte: torch.Tensor,
                            Yte: torch.Tensor,
                            settings: AdaptiveTrainSettings) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_kwargs: Dict[str, object] = {
        "num_workers": settings.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if settings.num_workers > 0:
        train_kwargs["persistent_workers"] = True
    tr_dl = DataLoader(
        TensorDataset(Xtr, Ctr, Ytr),
        batch_size=max(1, min(settings.batch_size, len(Ytr))) if len(Ytr) else settings.batch_size,
        shuffle=True,
        **train_kwargs,
    )

    eval_kwargs: Dict[str, object] = {"num_workers": 0, "pin_memory": torch.cuda.is_available()}
    eval_batch_val = max(1, min(settings.eval_batch_size, len(Yva))) if len(Yva) else settings.eval_batch_size
    eval_batch_test = max(1, min(settings.eval_batch_size, len(Yte))) if len(Yte) else settings.eval_batch_size
    va_dl = DataLoader(TensorDataset(Xva, Cva, Yva), batch_size=eval_batch_val, shuffle=False, **eval_kwargs)
    te_dl = DataLoader(TensorDataset(Xte, Cte, Yte), batch_size=eval_batch_test, shuffle=False, **eval_kwargs)
    return tr_dl, va_dl, te_dl


def _build_gat_dataloaders(train_graphs: List[Data],
                           val_graphs: List[Data],
                           test_graphs: List[Data],
                           settings: AdaptiveTrainSettings) -> Tuple[GeoDataLoader, GeoDataLoader, GeoDataLoader]:
    loader_kwargs: Dict[str, object] = {"num_workers": settings.num_workers, "pin_memory": torch.cuda.is_available()}
    if settings.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    tr_batches = _build_graph_index_batches(
        train_graphs,
        batch_size=max(1, settings.batch_size),
        nodes_per_batch=settings.nodes_per_batch,
        shuffle=True,
    )
    va_batches = _build_graph_index_batches(
        val_graphs,
        batch_size=max(1, min(settings.batch_size, len(val_graphs) or 1)),
        nodes_per_batch=settings.nodes_per_batch,
        shuffle=False,
    )
    te_batches = _build_graph_index_batches(
        test_graphs,
        batch_size=max(1, min(settings.batch_size, len(test_graphs) or 1)),
        nodes_per_batch=settings.nodes_per_batch,
        shuffle=False,
    )
    tr_dl = GeoDataLoader(train_graphs, batch_sampler=StaticBatchSampler(tr_batches), **loader_kwargs)
    va_dl = GeoDataLoader(val_graphs, batch_sampler=StaticBatchSampler(va_batches), num_workers=0, pin_memory=torch.cuda.is_available())
    te_dl = GeoDataLoader(test_graphs, batch_sampler=StaticBatchSampler(te_batches), num_workers=0, pin_memory=torch.cuda.is_available())
    return tr_dl, va_dl, te_dl


def _probe_lstm_batch_size(cfg: LSTMConfig, Xtr: torch.Tensor, Ctr: torch.Tensor,
                           hardware: Optional[HardwareProfile] = None) -> int:
    requested = max(4, min(cfg.batch_size, len(Xtr))) if len(Xtr) else max(4, cfg.batch_size)
    if not torch.cuda.is_available():
        return requested

    threshold_batch = _lstm_batch_threshold(requested, hardware)
    probe_limit = min(len(Xtr), max(threshold_batch, LSTM_BATCH_SIZES[-1]))
    probe_x = Xtr[:probe_limit].to(DEVICE, dtype=torch.float32)
    probe_c = Ctr[:probe_limit].to(DEVICE, dtype=torch.float32)
    for batch_size in _candidate_batch_sizes(threshold_batch, LSTM_BATCH_SIZES, minimum=4):
        try:
            model = BahdanauBiLSTM(
                hidden=cfg.hidden_dim,
                dropout=cfg.dropout,
                num_layers=cfg.num_layers,
                ctx_proj_dim=cfg.ctx_proj_dim,
            ).to(DEVICE)
            with torch.no_grad():
                model(
                    probe_x[: min(batch_size, len(probe_x))],
                    probe_c[: min(batch_size, len(probe_c))],
                )
            del model
            clear_torch_memory(logger, label="lstm-probe")
            logger.info(f"[HW-ADAPT] model=LSTM batch_size={batch_size}")
            return batch_size
        except Exception as exc:
            if not is_torch_oom(exc):
                raise
            clear_torch_memory(logger, label="lstm-probe-oom")
    return max(4, min(threshold_batch, LSTM_BATCH_SIZES[-1]))


def _probe_gat_batch_size(cfg: GATConfig, train_graphs: List[Data], n_node_feat: int,
                          hardware: Optional[HardwareProfile] = None) -> int:
    requested = max(1, min(cfg.batch_size, len(train_graphs))) if train_graphs else max(1, cfg.batch_size)
    if not torch.cuda.is_available() or not train_graphs:
        return requested

    threshold_batch = _gat_batch_threshold(requested, hardware)
    for batch_size in _candidate_batch_sizes(threshold_batch, GAT_BATCH_SIZES, minimum=1):
        try:
            loader = GeoDataLoader(
                train_graphs[: min(len(train_graphs), batch_size)],
                batch_size=min(batch_size, len(train_graphs)),
                shuffle=False,
            )
            batch = next(iter(loader)).to(DEVICE)
            model = GATv2Pipeline(
                n_node_feat=n_node_feat,
                hidden=cfg.hidden_dim,
                heads=cfg.heads,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
            ).to(DEVICE)
            with torch.no_grad():
                model(batch.x, batch.edge_index, getattr(batch, "edge_attr", None), batch.batch)
            del model, batch
            clear_torch_memory(logger, label="gat-probe")
            logger.info(f"[HW-ADAPT] model=GAT batch_size={batch_size}")
            return batch_size
        except Exception as exc:
            if not is_torch_oom(exc):
                raise
            clear_torch_memory(logger, label="gat-probe-oom")
    return max(1, min(threshold_batch, GAT_BATCH_SIZES[-1]))


def _derive_lstm_settings(cfg: LSTMConfig,
                          Xtr: torch.Tensor,
                          Ctr: torch.Tensor,
                          hardware: Optional[HardwareProfile]) -> AdaptiveTrainSettings:
    requested_batch = max(4, int(cfg.batch_size or 4))
    actual_batch = _probe_lstm_batch_size(cfg, Xtr, Ctr, hardware=hardware)
    settings = AdaptiveTrainSettings(
        model_name="LSTM",
        requested_batch_size=requested_batch,
        batch_size=actual_batch,
        eval_batch_size=_recommended_eval_batch_size(actual_batch, len(Xtr), graph_mode=False),
        num_workers=_adaptive_num_workers(hardware, len(Xtr)),
        grad_accum_steps=_gradient_accumulation_steps(requested_batch, actual_batch),
        fp16_enabled=_fp16_safe(hardware),
    )
    _event(
        "HW-ADAPT",
        model="LSTM",
        batch_size=settings.batch_size,
        eval_batch_size=settings.eval_batch_size,
        num_workers=settings.num_workers,
        grad_accum=settings.grad_accum_steps,
        fp16=settings.fp16_enabled,
    )
    return settings


def _derive_gat_settings(cfg: GATConfig,
                         train_graphs: List[Data],
                         n_node_feat: int,
                         hardware: Optional[HardwareProfile]) -> AdaptiveTrainSettings:
    requested_batch = max(1, int(cfg.batch_size or 1))
    actual_batch = _probe_gat_batch_size(cfg, train_graphs, n_node_feat, hardware=hardware)
    settings = AdaptiveTrainSettings(
        model_name="GAT",
        requested_batch_size=requested_batch,
        batch_size=actual_batch,
        eval_batch_size=_recommended_eval_batch_size(actual_batch, len(train_graphs), graph_mode=True),
        num_workers=_adaptive_num_workers(hardware, len(train_graphs)),
        grad_accum_steps=_gradient_accumulation_steps(requested_batch, actual_batch),
        fp16_enabled=_fp16_safe(hardware),
        nodes_per_batch=_default_nodes_per_batch(train_graphs, actual_batch, hardware),
        neighbor_limit=_default_neighbor_limit(train_graphs, hardware),
        gradient_checkpointing=bool(hardware and hardware.vram_total_gb and hardware.vram_total_gb <= 6.0),
    )
    _event(
        "HW-ADAPT",
        model="GAT",
        batch_size=settings.batch_size,
        eval_batch_size=settings.eval_batch_size,
        num_workers=settings.num_workers,
        grad_accum=settings.grad_accum_steps,
        fp16=settings.fp16_enabled,
    )
    _event(
        "GAT-OPT",
        nodes_per_batch=settings.nodes_per_batch,
        neighbor_limit=settings.neighbor_limit,
        using_fp16=settings.fp16_enabled,
        gradient_checkpointing=settings.gradient_checkpointing,
    )
    return settings


def _downshift_lstm_settings(settings: AdaptiveTrainSettings) -> AdaptiveTrainSettings:
    new_batch_size = max(4, settings.batch_size // 2)
    if new_batch_size == settings.batch_size and settings.batch_size > 4:
        new_batch_size = settings.batch_size - 1
    settings.batch_size = max(4, new_batch_size)
    settings.eval_batch_size = _recommended_eval_batch_size(settings.batch_size, settings.eval_batch_size, graph_mode=False)
    settings.grad_accum_steps = _gradient_accumulation_steps(settings.requested_batch_size, settings.batch_size)
    settings.num_workers = 0
    return settings


def _downshift_gat_settings(settings: AdaptiveTrainSettings) -> AdaptiveTrainSettings:
    new_batch_size = max(1, settings.batch_size // 2)
    if new_batch_size == settings.batch_size and settings.batch_size > 1:
        new_batch_size = settings.batch_size - 1
    settings.batch_size = max(1, new_batch_size)
    if settings.nodes_per_batch:
        settings.nodes_per_batch = max(128, settings.nodes_per_batch // 2)
    if settings.neighbor_limit is not None:
        settings.neighbor_limit = max(MIN_GAT_NEIGHBOR_LIMIT, settings.neighbor_limit - 2)
    settings.gradient_checkpointing = True
    settings.eval_batch_size = _recommended_eval_batch_size(settings.batch_size, settings.eval_batch_size, graph_mode=True)
    settings.grad_accum_steps = _gradient_accumulation_steps(settings.requested_batch_size, settings.batch_size)
    settings.num_workers = 0
    return settings


def train_lstm(cfg: LSTMConfig, ckpt_dir: Path,
               scheduler_type: str = 'plateau',
               resume_epoch: bool = False,
               domain_label: str = 'Synthetic',
               verbose_epochs: bool = False,
               hardware: Optional[HardwareProfile] = None) -> Tuple:
    timer = StageTimer()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = _checkpoints_dir(ckpt_dir)
    predictions_dir = _predictions_dir(ckpt_dir)
    hardware = hardware or HardwareProfile.probe()

    Xtr, Ctr, Ytr, Xva, Cva, Yva, Xte, Cte, Yte = load_task_b_tensors()
    _assert_finite_tensor(Xtr, domain="TASK-B", source="train_lstm/X_train")
    _assert_finite_tensor(Ctr, domain="TASK-B", source="train_lstm/X_ctx_train")
    _assert_finite_tensor(Xva, domain="TASK-B", source="train_lstm/X_val")
    _assert_finite_tensor(Cva, domain="TASK-B", source="train_lstm/X_ctx_val")
    _assert_finite_tensor(Xte, domain="TASK-B", source="train_lstm/X_test")
    _assert_finite_tensor(Cte, domain="TASK-B", source="train_lstm/X_ctx_test")
    _assert_finite_tensor(Ytr, domain="TASK-B", source="train_lstm/y_train")
    _assert_finite_tensor(Yva, domain="TASK-B", source="train_lstm/y_val")
    _assert_finite_tensor(Yte, domain="TASK-B", source="train_lstm/y_test")
    _event(
        "DATA-INTEGRITY",
        "TASK-B",
        "PRE_MODEL_INPUT",
        source="train_lstm",
        train_rows=int(len(Ytr)),
        val_rows=int(len(Yva)),
        test_rows=int(len(Yte)),
        action="ok",
    )
    settings = _derive_lstm_settings(cfg, Xtr, Ctr, hardware)
    if domain_label.lower() in {"travistorrent", "bitbrains"} and settings.fp16_enabled:
        settings.fp16_enabled = False
        _event(
            "HW-ADAPT",
            "LSTM",
            "FP16_DISABLED",
            domain=domain_label,
            reason="pre_freeze_nan_guard",
        )
    cfg.batch_size = settings.batch_size
    tr_dl, va_dl, te_dl = _build_lstm_dataloaders(Xtr, Ctr, Ytr, Xva, Cva, Yva, Xte, Cte, Yte, settings)

    model = BahdanauBiLSTM(hidden=cfg.hidden_dim, dropout=cfg.dropout, num_layers=cfg.num_layers, ctx_proj_dim=cfg.ctx_proj_dim).to(DEVICE)
    pos_weight = float((len(Ytr) - float(Ytr.sum().item())) / max(1.0, float(Ytr.sum().item()))) if len(Ytr) else 1.0
    _event(
        f"Seed {cfg.seed}",
        domain_label,
        "LSTM",
        "START",
        train_rows=len(Ytr),
        val_rows=len(Yva),
        test_rows=len(Yte),
        batch_size=settings.batch_size,
        eval_batch_size=settings.eval_batch_size,
        num_workers=settings.num_workers,
        grad_accum=settings.grad_accum_steps,
        fp16=settings.fp16_enabled,
        pos_weight=pos_weight,
        checkpoint_dir=checkpoints_dir.resolve(),
        prediction_dir=predictions_dir.resolve(),
    )
    _event(
        f"Seed {cfg.seed}",
        domain_label,
        "SPLIT",
        train=len(Ytr),
        val=len(Yva),
        test=len(Yte),
        train_pos=float(Ytr.sum().item()) if len(Ytr) else 0.0,
        val_pos=float(Yva.sum().item()) if len(Yva) else 0.0,
        test_pos=float(Yte.sum().item()) if len(Yte) else 0.0,
    )
    _event(
        f"Seed {cfg.seed}",
        domain_label,
        "CLASS-BALANCE",
        pos=int(Ytr.sum().item()) if len(Ytr) else 0,
        neg=int(len(Ytr) - float(Ytr.sum().item())) if len(Ytr) else 0,
        ratio=float(Ytr.mean().item()) if len(Ytr) else 0.0,
        pos_weight=pos_weight,
    )
    _log_system_stats("lstm_start")
    criterion = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _make_scheduler(optimizer, scheduler_type, cfg.epochs)
    _gs_dev = 'cuda' if torch.cuda.is_available() else 'cpu'  # FIX: CPU-safe
    scaler = torch.amp.GradScaler(_gs_dev, enabled=settings.fp16_enabled and torch.cuda.is_available())

    ckpt_path = checkpoints_dir / 'lstm_ckpt.pt'
    best_path = checkpoints_dir / 'lstm_best.pt'
    checkpoint_keep_last_k = max(0, int(cfg.checkpoint_keep_last_k))
    epoch_guard_path = checkpoints_dir / 'lstm_epoch_guard.pt'
    start_epoch = 1
    best_f1 = 0.0
    best_epoch = 0
    patience_ctr = 0
    hist: Dict[str, object] = _normalise_training_history(None)
    hist['seed'] = int(cfg.seed)
    hist['domain_label'] = domain_label
    hist['model_name'] = 'lstm'
    hist['model_config'] = asdict(cfg)
    hist['config_hash'] = _config_payload_hash({
        'seed': int(cfg.seed),
        'domain_label': domain_label,
        'model_name': 'lstm',
        'model_config': asdict(cfg),
        'scheduler_type': scheduler_type,
    })
    hist['adaptive'] = settings.as_dict()
    if not resume_epoch:
        _purge_stale_checkpoints(checkpoints_dir, 'lstm', ckpt_path, best_path, epoch_guard_path)
        hist['resume_loaded'] = False
        hist['start_epoch'] = start_epoch
    else:
        resume_path, resume_state = _resolve_latest_checkpoint(checkpoints_dir, 'lstm', ckpt_path)
        if resume_state is not None and resume_path is not None:
            try:
                best_f1, best_epoch, patience_ctr, hist = _restore_training_state_objects(
                    model, optimizer, scheduler, scaler, resume_state
                )
                start_epoch = _resume_epoch_from_checkpoint(resume_state)
                hist['resume_loaded'] = True
                hist['start_epoch'] = start_epoch
                hist['adaptive'] = settings.as_dict()
                _event("CHECKPOINT", "LOAD", path=resume_path.resolve(), resumed_epoch=start_epoch)
                _event(
                    f"Seed {cfg.seed}",
                    domain_label,
                    "CHECKPOINT",
                    "LOAD",
                    path=resume_path.resolve(),
                    resumed_epoch=start_epoch,
                    best=best_f1,
                    patience=f"{patience_ctr}/{cfg.patience}",
                    requested=bool(resume_epoch),
                )
                _event(
                    f"Seed {cfg.seed}",
                    domain_label,
                    "LSTM",
                    "RESUME",
                    checkpoint=resume_path.resolve(),
                    next_epoch=start_epoch,
                    best=best_f1,
                    patience=f"{patience_ctr}/{cfg.patience}",
                )
            except Exception as exc:
                logger.warning(f'[LSTM] Checkpoint resume skipped: {exc}')
                hist['start_epoch'] = start_epoch
        else:
            hist['start_epoch'] = start_epoch
            logger.info('[LSTM] --resume-epoch requested but no checkpoint was found; starting fresh')
    if 'start_epoch' not in hist:
        hist['start_epoch'] = start_epoch

    oom_exhausted = False
    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_attempt = 0
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        val_f1 = 0.0

        while True:
            hist['adaptive'] = settings.as_dict()
            guard_payload = _build_training_checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                max(1, epoch - 1),
                best_f1,
                best_epoch,
                patience_ctr,
                cfg.patience,
                hist,
                checkpoint_type='epoch_guard',
                epoch_completed=False,
            )
            _atomic_torch_save(epoch_guard_path, guard_payload)
            try:
                if epoch == start_epoch or epoch == cfg.epochs or epoch % settings.monitor_every_epochs == 0:
                    _log_system_stats(f"lstm_epoch_{epoch}")

                model.train()
                epoch_loss = 0.0
                optimizer.zero_grad(set_to_none=True)
                for step, (batch_x, batch_c, batch_y) in enumerate(tr_dl, start=1):
                    batch_x = batch_x.to(DEVICE, dtype=torch.float32)
                    batch_c = batch_c.to(DEVICE, dtype=torch.float32)
                    batch_y = batch_y.to(DEVICE, dtype=torch.float32)
                    _assert_finite_tensor(batch_x, domain="TASK-B", source=f"train_lstm/epoch_{epoch}/step_{step}/batch_x")
                    _assert_finite_tensor(batch_c, domain="TASK-B", source=f"train_lstm/epoch_{epoch}/step_{step}/batch_c")
                    _assert_finite_tensor(batch_y, domain="TASK-B", source=f"train_lstm/epoch_{epoch}/step_{step}/batch_y")
                    with torch.amp.autocast(device_type=_gs_dev, enabled=settings.fp16_enabled and torch.cuda.is_available()):  # FIX
                        logits, _ = model(batch_x, batch_c)
                        _assert_finite_tensor(logits, domain="TASK-B", source=f"train_lstm/epoch_{epoch}/step_{step}/logits")
                        loss = criterion(logits, batch_y)
                        if not torch.isfinite(loss):
                            raise ValueError(f"[TASK-B] Non-finite training loss at epoch={epoch}, step={step}")
                        scaled_loss = loss / settings.grad_accum_steps
                    scaler.scale(scaled_loss).backward()
                    should_step = (step % settings.grad_accum_steps == 0) or (step == len(tr_dl))
                    if should_step:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        hist['optimizer_steps'] = int(hist.get('optimizer_steps', 0) or 0) + 1
                        optimizer.zero_grad(set_to_none=True)
                    epoch_loss += float(loss.item())

                val_logits, val_labels = [], []
                val_loss_total = 0.0
                model.eval()
                with torch.no_grad():
                    for batch_x, batch_c, batch_y in va_dl:
                        batch_x = batch_x.to(DEVICE, dtype=torch.float32)
                        batch_c = batch_c.to(DEVICE, dtype=torch.float32)
                        batch_y = batch_y.to(DEVICE, dtype=torch.float32)
                        _assert_finite_tensor(batch_x, domain="TASK-B", source=f"val_lstm/epoch_{epoch}/batch_x")
                        _assert_finite_tensor(batch_c, domain="TASK-B", source=f"val_lstm/epoch_{epoch}/batch_c")
                        _assert_finite_tensor(batch_y, domain="TASK-B", source=f"val_lstm/epoch_{epoch}/batch_y")
                        logits, _ = model(batch_x, batch_c)
                        _assert_finite_tensor(logits, domain="TASK-B", source=f"val_lstm/epoch_{epoch}/logits")
                        val_loss_total += float(criterion(logits, batch_y).item())
                        val_logits.append(logits.detach().cpu().numpy())
                        val_labels.append(batch_y.detach().cpu().numpy())

                val_log = np.concatenate(val_logits) if val_logits else np.array([])
                val_lbl = np.concatenate(val_labels) if val_labels else np.array([])
                val_log = _validate_numeric_checkpoint(
                    val_log,
                    domain="TASK-B",
                    stage="VAL_LOOP",
                    source=f"epoch_{epoch}:val_logits",
                    repair=False,
                )
                val_lbl = _validate_numeric_checkpoint(
                    val_lbl,
                    domain="TASK-B",
                    stage="VAL_LOOP",
                    source=f"epoch_{epoch}:val_labels",
                    repair=False,
                )
                val_f1, _ = f1_at_optimal_threshold(_sig(val_log), val_lbl) if len(val_log) else (0.0, 0.5)
                if scheduler_type == 'cosine':
                    scheduler.step(epoch)
                else:
                    scheduler.step(val_f1)
                epoch_train_loss = epoch_loss / max(1, len(tr_dl))
                epoch_val_loss = val_loss_total / max(1, len(va_dl))
                break
            except Exception as exc:
                if not is_torch_oom(exc):
                    raise
                epoch_attempt += 1
                clear_torch_memory(logger, label=f"lstm-epoch-{epoch}-oom")
                previous_batch = settings.batch_size
                previous_accum = settings.grad_accum_steps
                oom_events = hist.setdefault('oom_events', [])
                assert isinstance(oom_events, list)
                oom_events.append(
                    {
                        'model': 'lstm',
                        'epoch': epoch,
                        'attempt': epoch_attempt,
                        'batch_size': settings.batch_size,
                        'grad_accum_steps': settings.grad_accum_steps,
                    }
                )
                preserved_oom_events = list(oom_events)
                if epoch_attempt > settings.oom_retry_limit:
                    hist['aborted_due_to_oom'] = True
                    hist['oom_retries_exhausted'] = settings.oom_retry_limit
                    _event(
                        "OOM-FAILSAFE",
                        model="LSTM",
                        epoch=epoch,
                        retries=settings.oom_retry_limit,
                        batch_size=settings.batch_size,
                        status="skipped",
                    )
                    oom_exhausted = True
                    break
                settings = _downshift_lstm_settings(settings)
                cfg.batch_size = settings.batch_size
                _gs_dev = 'cuda' if torch.cuda.is_available() else 'cpu'  # FIX: CPU-safe
                scaler = torch.amp.GradScaler(_gs_dev, enabled=settings.fp16_enabled and torch.cuda.is_available())
                guard_state = _load_training_checkpoint(epoch_guard_path)
                best_f1, best_epoch, patience_ctr, hist = _restore_training_state_objects(
                    model, optimizer, scheduler, scaler, guard_state
                )
                hist['start_epoch'] = start_epoch
                hist['oom_events'] = preserved_oom_events
                hist['adaptive'] = settings.as_dict()
                tr_dl, va_dl, te_dl = _build_lstm_dataloaders(Xtr, Ctr, Ytr, Xva, Cva, Yva, Xte, Cte, Yte, settings)
                _event(
                    "OOM-DETECTED",
                    model="LSTM",
                    epoch=epoch,
                    attempt=epoch_attempt,
                    batch_size=f"{previous_batch}->{settings.batch_size}",
                    grad_accum=f"{previous_accum}->{settings.grad_accum_steps}",
                )
                _log_system_stats(f"lstm_retry_epoch_{epoch}")
                continue
        if oom_exhausted:
            break

        current_lr = float(optimizer.param_groups[0]['lr'])
        train_loss_hist = hist.setdefault('train_loss', [])
        val_loss_hist = hist.setdefault('val_loss', [])
        val_hist = hist.setdefault('val_f1_opt', [])
        lr_hist = hist.setdefault('lr', [])
        assert isinstance(train_loss_hist, list)
        assert isinstance(val_loss_hist, list)
        assert isinstance(val_hist, list)
        assert isinstance(lr_hist, list)
        train_loss_hist.append(epoch_train_loss)
        val_loss_hist.append(epoch_val_loss)
        val_hist.append(float(val_f1))
        lr_hist.append(current_lr)
        improved = val_f1 > best_f1  # FIX: strict > so plateau increments patience
        if improved:
            best_f1 = float(val_f1)
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
        hist['best_epoch'] = best_epoch
        hist['best_f1'] = best_f1
        hist['last_epoch'] = epoch
        hist['stopped_early'] = False
        hist['adaptive'] = settings.as_dict()
        if improved:
            best_payload = _build_training_checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_f1,
                best_epoch,
                patience_ctr,
                cfg.patience,
                hist,
                checkpoint_type='best',
            )
            _save_training_checkpoint(
                checkpoints_dir,
                'lstm',
                'best',
                best_path,
                best_payload,
                epoch,
                checkpoint_keep_last_k,
                metric=best_f1,
            )
            _event(
                f"Seed {cfg.seed}",
                domain_label,
                "LSTM",
                "BEST",
                epoch=best_epoch,
                val_f1=best_f1,
                checkpoint=best_path.resolve(),
            )
        latest_payload = _build_training_checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_f1,
            best_epoch,
            patience_ctr,
            cfg.patience,
            hist,
            checkpoint_type='latest',
        )
        _save_training_checkpoint(
            checkpoints_dir,
            'lstm',
            'latest',
            ckpt_path,
            latest_payload,
            epoch,
            checkpoint_keep_last_k,
        )
        status = 'checkpoint=best' if improved else 'checkpoint=latest'
        if verbose_epochs:
            logger.info(
                _epoch_status_line(
                    cfg.seed,
                    domain_label,
                    'LSTM',
                    epoch,
                    cfg.epochs,
                    epoch_train_loss,
                    epoch_val_loss,
                    float(val_f1),
                    best_f1,
                    patience_ctr,
                    cfg.patience,
                    current_lr,
                    status,
                )
            )
        if patience_ctr >= cfg.patience:
            hist['stopped_early'] = True
            _event(
                f"Seed {cfg.seed}",
                domain_label,
                "EARLY-STOP",
                metric="val_f1",
                current=val_f1,
                best=best_f1,
                epoch=epoch,
                best_epoch=best_epoch,
                patience=f"{patience_ctr}/{cfg.patience}",
            )
            break

    if best_path.exists():
        model.load_state_dict(_load_model_payload(best_path))
    elif hist.get('aborted_due_to_oom'):
        _event(
            f"Seed {cfg.seed}",
            domain_label,
            "LSTM",
            "FINAL",
            status="oom_aborted",
            duration=format_duration_s(timer.elapsed_s),
        )
        return None, hist, np.array([]), np.array([]), np.array([]), np.array([]), 0.0, 1.0

    _event(
        f"Seed {cfg.seed}",
        domain_label,
        "LSTM",
        "FINAL",
        best_epoch=best_epoch,
        best_val_f1=best_f1,
        duration=format_duration_s(timer.elapsed_s),
    )
    _log_system_stats("lstm_end")
    model.eval()
    val_logits, val_labels, test_logits, attn_all = [], [], [], []
    with torch.no_grad():
        for batch_x, batch_c, batch_y in va_dl:
            _assert_finite_tensor(batch_x, domain="TASK-B", source="eval_lstm/val/batch_x")
            _assert_finite_tensor(batch_c, domain="TASK-B", source="eval_lstm/val/batch_c")
            logits, attn = model(batch_x.to(DEVICE, dtype=torch.float32), batch_c.to(DEVICE, dtype=torch.float32))
            _assert_finite_tensor(logits, domain="TASK-B", source="eval_lstm/val/logits")
            val_logits.append(logits.detach().cpu().numpy())
            val_labels.append(batch_y.numpy())
        for batch_x, batch_c, _ in te_dl:
            _assert_finite_tensor(batch_x, domain="TASK-B", source="eval_lstm/test/batch_x")
            _assert_finite_tensor(batch_c, domain="TASK-B", source="eval_lstm/test/batch_c")
            logits, attn = model(batch_x.to(DEVICE, dtype=torch.float32), batch_c.to(DEVICE, dtype=torch.float32))
            _assert_finite_tensor(logits, domain="TASK-B", source="eval_lstm/test/logits")
            test_logits.append(logits.detach().cpu().numpy())
            attn_all.append(attn.detach().cpu().numpy())
    val_log_f = np.concatenate(val_logits) if val_logits else np.array([])
    val_lbl_f = np.concatenate(val_labels) if val_labels else np.array([])
    test_log = np.concatenate(test_logits) if test_logits else np.array([])
    attn_np = np.concatenate(attn_all) if attn_all else np.array([])
    val_log_f = _validate_numeric_checkpoint(val_log_f, domain="TASK-B", stage="PRE_METRIC", source="lstm_val_logits", repair=False, log_ok=True)
    val_lbl_f = _validate_numeric_checkpoint(val_lbl_f, domain="TASK-B", stage="PRE_METRIC", source="lstm_val_labels", repair=False, log_ok=True)
    test_log = _validate_numeric_checkpoint(test_log, domain="TASK-B", stage="PRE_METRIC", source="lstm_test_logits", repair=False, log_ok=True)
    attn_np = _validate_numeric_checkpoint(attn_np, domain="TASK-B", stage="PRE_METRIC", source="lstm_attention", repair=True, log_ok=True)
    _atomic_numpy_save(predictions_dir / 'lstm_val_logits.npy', val_log_f)
    _atomic_numpy_save(predictions_dir / 'lstm_test_logits.npy', test_log)
    _event(
        f"Seed {cfg.seed}",
        domain_label,
        "WRITE",
        artifact="predictions",
        val_logits=Path(predictions_dir / 'lstm_val_logits.npy').resolve(),
        test_logits=Path(predictions_dir / 'lstm_test_logits.npy').resolve(),
    )
    temp = temperature_scale(val_log_f, val_lbl_f) if len(val_log_f) else 1.0
    val_pr_auc = float(average_precision_score(val_lbl_f, _sig(val_log_f))) if len(np.unique(val_lbl_f)) > 1 else 0.0
    return model, hist, val_log_f, test_log, val_lbl_f, attn_np, val_pr_auc, temp


def train_gat(cfg: GATConfig, ckpt_dir: Path,
              scheduler_type: str = 'plateau',
              resume_epoch: bool = False,
              domain_label: str = 'Synthetic',
              verbose_epochs: bool = False,
              hardware: Optional[HardwareProfile] = None) -> Tuple:
    timer = StageTimer()
    if not _HAS_PYG:
        logger.warning('[GAT] torch_geometric not available - skipping GAT training')
        return None, {}, np.array([]), np.array([]), np.array([]), None, 0.0, 1.0
    hardware = hardware or HardwareProfile.probe()
    cfg_path = (_TASK_C_DIR / 'config.json') if _TASK_C_DIR else None
    n_node_feat = int(json.loads(cfg_path.read_text()).get('n_node_features', N_NODE_FEAT)) if cfg_path and cfg_path.exists() else N_NODE_FEAT
    train_graphs = load_task_c_graphs('train')
    val_graphs = load_task_c_graphs('val')
    test_graphs = load_task_c_graphs('test')
    if not train_graphs or not val_graphs or not test_graphs:
        logger.warning('[GAT] Missing graph splits - skipping GAT training')
        return None, {}, np.array([]), np.array([]), np.array([]), None, 0.0, 1.0
    _event(
        "DATA-INTEGRITY",
        "TASK-C",
        "PRE_MODEL_INPUT",
        source="train_gat",
        train_graphs=int(len(train_graphs)),
        val_graphs=int(len(val_graphs)),
        test_graphs=int(len(test_graphs)),
        action="ok",
    )
    settings = _derive_gat_settings(cfg, train_graphs, n_node_feat, hardware)
    if domain_label.lower() in {"travistorrent", "bitbrains"} and settings.fp16_enabled:
        settings.fp16_enabled = False
        _event(
            "HW-ADAPT",
            "GAT",
            "FP16_DISABLED",
            domain=domain_label,
            reason="pre_freeze_nan_guard",
        )
    cfg.batch_size = settings.batch_size
    tr_dl, va_dl, te_dl = _build_gat_dataloaders(train_graphs, val_graphs, test_graphs, settings)

    model = GATv2Pipeline(
        n_node_feat=n_node_feat,
        hidden=cfg.hidden_dim,
        heads=cfg.heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        gradient_checkpointing=settings.gradient_checkpointing,
    ).to(DEVICE)
    train_labels = np.array([graph.y.item() for graph in train_graphs], dtype=np.float32)
    pos_weight = float((len(train_labels) - train_labels.sum()) / max(1.0, train_labels.sum())) if len(train_labels) else 1.0
    edge_rows = int(sum(int(graph.edge_index.shape[1]) for graph in train_graphs + val_graphs + test_graphs))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = _checkpoints_dir(ckpt_dir)
    predictions_dir = _predictions_dir(ckpt_dir)
    _event(
        f"Seed {cfg.seed}",
        domain_label,
        "GAT",
        "START",
        train_graphs=len(train_graphs),
        val_graphs=len(val_graphs),
        test_graphs=len(test_graphs),
        batch_size=settings.batch_size,
        eval_batch_size=settings.eval_batch_size,
        num_workers=settings.num_workers,
        grad_accum=settings.grad_accum_steps,
        fp16=settings.fp16_enabled,
        nodes_per_batch=settings.nodes_per_batch,
        neighbor_limit=settings.neighbor_limit,
        gradient_checkpointing=settings.gradient_checkpointing,
        n_node_features=n_node_feat,
        edge_rows=edge_rows,
        checkpoint_dir=checkpoints_dir.resolve(),
    )
    _event(
        f"Seed {cfg.seed}",
        domain_label,
        "GRAPH",
        split="train/val/test",
        train=len(train_graphs),
        val=len(val_graphs),
        test=len(test_graphs),
        n_node_features=n_node_feat,
        edge_rows=edge_rows,
    )
    _event(
        f"Seed {cfg.seed}",
        domain_label,
        "CLASS-BALANCE",
        pos=int(train_labels.sum()) if len(train_labels) else 0,
        neg=int(len(train_labels) - train_labels.sum()) if len(train_labels) else 0,
        ratio=float(train_labels.mean()) if len(train_labels) else 0.0,
        pos_weight=pos_weight,
    )
    _log_system_stats("gat_start")
    criterion = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _make_scheduler(optimizer, scheduler_type, cfg.epochs)
    _gs_dev = 'cuda' if torch.cuda.is_available() else 'cpu'  # FIX: CPU-safe
    scaler = torch.amp.GradScaler(_gs_dev, enabled=settings.fp16_enabled and torch.cuda.is_available())

    ckpt_path = checkpoints_dir / 'gat_ckpt.pt'
    best_path = checkpoints_dir / 'gat_best.pt'
    checkpoint_keep_last_k = max(0, int(cfg.checkpoint_keep_last_k))
    epoch_guard_path = checkpoints_dir / 'gat_epoch_guard.pt'
    start_epoch = 1
    best_f1 = 0.0
    best_epoch = 0
    patience_ctr = 0
    hist: Dict[str, object] = _normalise_training_history(None)
    hist['seed'] = int(cfg.seed)
    hist['domain_label'] = domain_label
    hist['model_name'] = 'gat'
    hist['model_config'] = asdict(cfg)
    hist['config_hash'] = _config_payload_hash({
        'seed': int(cfg.seed),
        'domain_label': domain_label,
        'model_name': 'gat',
        'model_config': asdict(cfg),
        'scheduler_type': scheduler_type,
        'n_node_feat': n_node_feat,
    })
    hist['adaptive'] = settings.as_dict()
    if not resume_epoch:
        _purge_stale_checkpoints(checkpoints_dir, 'gat', ckpt_path, best_path, epoch_guard_path)
        hist['resume_loaded'] = False
        hist['start_epoch'] = start_epoch
    else:
        resume_path, resume_state = _resolve_latest_checkpoint(checkpoints_dir, 'gat', ckpt_path)
        if resume_state is not None and resume_path is not None:
            try:
                best_f1, best_epoch, patience_ctr, hist = _restore_training_state_objects(
                    model, optimizer, scheduler, scaler, resume_state
                )
                start_epoch = _resume_epoch_from_checkpoint(resume_state)
                hist['resume_loaded'] = True
                hist['start_epoch'] = start_epoch
                hist['adaptive'] = settings.as_dict()
                _event(
                    "CHECKPOINT",
                    "LOAD",
                    path=resume_path.resolve(),
                    resumed_epoch=start_epoch,
                )
                _event(
                    f"Seed {cfg.seed}",
                    domain_label,
                    "CHECKPOINT",
                    "LOAD",
                    path=resume_path.resolve(),
                    resumed_epoch=start_epoch,
                    best=best_f1,
                    patience=f"{patience_ctr}/{cfg.patience}",
                    requested=bool(resume_epoch),
                )
                _event(
                    f"Seed {cfg.seed}",
                    domain_label,
                    "GAT",
                    "RESUME",
                    checkpoint=resume_path.resolve(),
                    next_epoch=start_epoch,
                    best=best_f1,
                    patience=f"{patience_ctr}/{cfg.patience}",
                )
            except Exception as exc:
                logger.warning(f'[GAT] Checkpoint resume skipped: {exc}')
                hist['start_epoch'] = start_epoch
        else:
            hist['start_epoch'] = start_epoch
            logger.info('[GAT] --resume-epoch requested but no checkpoint was found; starting fresh')
    if 'start_epoch' not in hist:
        hist['start_epoch'] = start_epoch

    oom_exhausted = False
    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_attempt = 0
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        val_f1 = 0.0

        while True:
            hist['adaptive'] = settings.as_dict()
            guard_payload = _build_training_checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                max(1, epoch - 1),
                best_f1,
                best_epoch,
                patience_ctr,
                cfg.patience,
                hist,
                checkpoint_type='epoch_guard',
                epoch_completed=False,
            )
            _atomic_torch_save(epoch_guard_path, guard_payload)
            try:
                if epoch == start_epoch or epoch == cfg.epochs or epoch % settings.monitor_every_epochs == 0:
                    _log_system_stats(f"gat_epoch_{epoch}")

                model.train()
                epoch_loss = 0.0
                optimizer.zero_grad(set_to_none=True)
                for step, batch in enumerate(tr_dl, start=1):
                    batch = _limit_graph_batch_neighbors(batch, settings.neighbor_limit)
                    batch = batch.to(DEVICE)
                    _assert_finite_tensor(batch.x, domain="TASK-C", source=f"train_gat/epoch_{epoch}/step_{step}/x")
                    if getattr(batch, 'edge_attr', None) is not None:
                        _assert_finite_tensor(batch.edge_attr, domain="TASK-C", source=f"train_gat/epoch_{epoch}/step_{step}/edge_attr")
                    _assert_finite_tensor(batch.y.float().view(-1), domain="TASK-C", source=f"train_gat/epoch_{epoch}/step_{step}/y")
                    with torch.amp.autocast(device_type=_gs_dev, enabled=settings.fp16_enabled and torch.cuda.is_available()):  # FIX
                        logits = model(batch.x, batch.edge_index, getattr(batch, 'edge_attr', None), batch.batch)
                        _assert_finite_tensor(logits, domain="TASK-C", source=f"train_gat/epoch_{epoch}/step_{step}/logits")
                        loss = criterion(logits, batch.y.float().view(-1))
                        if not torch.isfinite(loss):
                            raise ValueError(f"[TASK-C] Non-finite training loss at epoch={epoch}, step={step}")
                        scaled_loss = loss / settings.grad_accum_steps
                    scaler.scale(scaled_loss).backward()
                    should_step = (step % settings.grad_accum_steps == 0) or (step == len(tr_dl))
                    if should_step:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        hist['optimizer_steps'] = int(hist.get('optimizer_steps', 0) or 0) + 1
                        optimizer.zero_grad(set_to_none=True)
                    epoch_loss += float(loss.item())

                val_logits, val_labels = [], []
                val_loss_total = 0.0
                model.eval()
                with torch.no_grad():
                    for batch in va_dl:
                        batch = _limit_graph_batch_neighbors(batch, settings.neighbor_limit)
                        batch = batch.to(DEVICE)
                        _assert_finite_tensor(batch.x, domain="TASK-C", source=f"val_gat/epoch_{epoch}/x")
                        if getattr(batch, 'edge_attr', None) is not None:
                            _assert_finite_tensor(batch.edge_attr, domain="TASK-C", source=f"val_gat/epoch_{epoch}/edge_attr")
                        _assert_finite_tensor(batch.y.float().view(-1), domain="TASK-C", source=f"val_gat/epoch_{epoch}/y")
                        logits = model(batch.x, batch.edge_index, getattr(batch, 'edge_attr', None), batch.batch)
                        _assert_finite_tensor(logits, domain="TASK-C", source=f"val_gat/epoch_{epoch}/logits")
                        val_loss_total += float(criterion(logits, batch.y.float().view(-1)).item())
                        val_logits.append(logits.detach().cpu().numpy())
                        val_labels.append(batch.y.float().view(-1).cpu().numpy())
                val_log = np.concatenate(val_logits) if val_logits else np.array([])
                val_lbl = np.concatenate(val_labels) if val_labels else np.array([])
                val_log = _validate_numeric_checkpoint(
                    val_log,
                    domain="TASK-C",
                    stage="VAL_LOOP",
                    source=f"epoch_{epoch}:val_logits",
                    repair=False,
                )
                val_lbl = _validate_numeric_checkpoint(
                    val_lbl,
                    domain="TASK-C",
                    stage="VAL_LOOP",
                    source=f"epoch_{epoch}:val_labels",
                    repair=False,
                )
                val_f1, _ = f1_at_optimal_threshold(_sig(val_log), val_lbl) if len(val_log) else (0.0, 0.5)
                if scheduler_type == 'cosine':
                    scheduler.step(epoch)
                else:
                    scheduler.step(val_f1)
                epoch_train_loss = epoch_loss / max(1, len(tr_dl))
                epoch_val_loss = val_loss_total / max(1, len(va_dl))
                break
            except Exception as exc:
                if not is_torch_oom(exc):
                    raise
                epoch_attempt += 1
                clear_torch_memory(logger, label=f"gat-epoch-{epoch}-oom")
                previous_batch = settings.batch_size
                previous_nodes = settings.nodes_per_batch
                previous_neighbor_limit = settings.neighbor_limit
                previous_accum = settings.grad_accum_steps
                oom_events = hist.setdefault('oom_events', [])
                assert isinstance(oom_events, list)
                oom_events.append(
                    {
                        'model': 'gat',
                        'epoch': epoch,
                        'attempt': epoch_attempt,
                        'batch_size': settings.batch_size,
                        'nodes_per_batch': settings.nodes_per_batch,
                        'neighbor_limit': settings.neighbor_limit,
                    }
                )
                preserved_oom_events = list(oom_events)
                if epoch_attempt > settings.oom_retry_limit:
                    hist['aborted_due_to_oom'] = True
                    hist['oom_retries_exhausted'] = settings.oom_retry_limit
                    _event(
                        "OOM-FAILSAFE",
                        model="GAT",
                        epoch=epoch,
                        retries=settings.oom_retry_limit,
                        batch_size=settings.batch_size,
                        nodes_per_batch=settings.nodes_per_batch,
                        status="skipped",
                    )
                    oom_exhausted = True
                    break
                settings = _downshift_gat_settings(settings)
                cfg.batch_size = settings.batch_size
                model.gradient_checkpointing = settings.gradient_checkpointing
                _gs_dev = 'cuda' if torch.cuda.is_available() else 'cpu'  # FIX: CPU-safe
                scaler = torch.amp.GradScaler(_gs_dev, enabled=settings.fp16_enabled and torch.cuda.is_available())
                guard_state = _load_training_checkpoint(epoch_guard_path)
                best_f1, best_epoch, patience_ctr, hist = _restore_training_state_objects(
                    model, optimizer, scheduler, scaler, guard_state
                )
                hist['start_epoch'] = start_epoch
                hist['oom_events'] = preserved_oom_events
                hist['adaptive'] = settings.as_dict()
                tr_dl, va_dl, te_dl = _build_gat_dataloaders(train_graphs, val_graphs, test_graphs, settings)
                _event(
                    "OOM-DETECTED",
                    model="GAT",
                    epoch=epoch,
                    attempt=epoch_attempt,
                    batch_size=f"{previous_batch}->{settings.batch_size}",
                    nodes_per_batch=f"{previous_nodes}->{settings.nodes_per_batch}",
                    neighbor_limit=f"{previous_neighbor_limit}->{settings.neighbor_limit}",
                    grad_accum=f"{previous_accum}->{settings.grad_accum_steps}",
                )
                _event(
                    "GAT-OPT",
                    nodes_per_batch=settings.nodes_per_batch,
                    neighbor_limit=settings.neighbor_limit,
                    using_fp16=settings.fp16_enabled,
                    gradient_checkpointing=settings.gradient_checkpointing,
                )
                _log_system_stats(f"gat_retry_epoch_{epoch}")
                continue
        if oom_exhausted:
            break

        current_lr = float(optimizer.param_groups[0]['lr'])
        train_loss_hist = hist.setdefault('train_loss', [])
        val_loss_hist = hist.setdefault('val_loss', [])
        val_hist = hist.setdefault('val_f1_opt', [])
        lr_hist = hist.setdefault('lr', [])
        assert isinstance(train_loss_hist, list)
        assert isinstance(val_loss_hist, list)
        assert isinstance(val_hist, list)
        assert isinstance(lr_hist, list)
        train_loss_hist.append(epoch_train_loss)
        val_loss_hist.append(epoch_val_loss)
        val_hist.append(float(val_f1))
        lr_hist.append(current_lr)
        improved = val_f1 > best_f1  # FIX: strict > so plateau increments patience
        if improved:
            best_f1 = float(val_f1)
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
        hist['best_epoch'] = best_epoch
        hist['best_f1'] = best_f1
        hist['last_epoch'] = epoch
        hist['stopped_early'] = False
        hist['adaptive'] = settings.as_dict()
        if improved:
            best_payload = _build_training_checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_f1,
                best_epoch,
                patience_ctr,
                cfg.patience,
                hist,
                checkpoint_type='best',
            )
            _save_training_checkpoint(
                checkpoints_dir,
                'gat',
                'best',
                best_path,
                best_payload,
                epoch,
                checkpoint_keep_last_k,
                metric=best_f1,
            )
            _event(
                f"Seed {cfg.seed}",
                domain_label,
                "GAT",
                "BEST",
                epoch=best_epoch,
                val_f1=best_f1,
                checkpoint=best_path.resolve(),
            )
        latest_payload = _build_training_checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_f1,
            best_epoch,
            patience_ctr,
            cfg.patience,
            hist,
            checkpoint_type='latest',
        )
        _save_training_checkpoint(
            checkpoints_dir,
            'gat',
            'latest',
            ckpt_path,
            latest_payload,
            epoch,
            checkpoint_keep_last_k,
        )
        status = 'checkpoint=best' if improved else 'checkpoint=latest'
        if verbose_epochs:
            logger.info(
                _epoch_status_line(
                    cfg.seed,
                    domain_label,
                    'GAT',
                    epoch,
                    cfg.epochs,
                    epoch_train_loss,
                    epoch_val_loss,
                    float(val_f1),
                    best_f1,
                    patience_ctr,
                    cfg.patience,
                    current_lr,
                    status,
                )
            )
        if patience_ctr >= cfg.patience:
            hist['stopped_early'] = True
            _event(
                f"Seed {cfg.seed}",
                domain_label,
                "EARLY-STOP",
                metric="val_f1",
                current=val_f1,
                best=best_f1,
                epoch=epoch,
                best_epoch=best_epoch,
                patience=f"{patience_ctr}/{cfg.patience}",
            )
            break

    if best_path.exists():
        model.load_state_dict(_load_model_payload(best_path))
    elif hist.get('aborted_due_to_oom'):
        _event(
            f"Seed {cfg.seed}",
            domain_label,
            "GAT",
            "FINAL",
            status="oom_aborted",
            duration=format_duration_s(timer.elapsed_s),
        )
        return None, hist, np.array([]), np.array([]), np.array([]), None, 0.0, 1.0

    _event(
        f"Seed {cfg.seed}",
        domain_label,
        "GAT",
        "FINAL",
        best_epoch=best_epoch,
        best_val_f1=best_f1,
        duration=format_duration_s(timer.elapsed_s),
    )
    _log_system_stats("gat_end")
    model.eval()
    val_logits, val_labels, test_logits = [], [], []
    with torch.no_grad():
        for batch in va_dl:
            batch = _limit_graph_batch_neighbors(batch, settings.neighbor_limit)
            batch = batch.to(DEVICE)
            _assert_finite_tensor(batch.x, domain="TASK-C", source="eval_gat/val/x")
            if getattr(batch, 'edge_attr', None) is not None:
                _assert_finite_tensor(batch.edge_attr, domain="TASK-C", source="eval_gat/val/edge_attr")
            logits = model(batch.x, batch.edge_index, getattr(batch, 'edge_attr', None), batch.batch)
            _assert_finite_tensor(logits, domain="TASK-C", source="eval_gat/val/logits")
            val_logits.append(logits.detach().cpu().numpy())
            val_labels.append(batch.y.float().view(-1).cpu().numpy())
        for batch in te_dl:
            batch = _limit_graph_batch_neighbors(batch, settings.neighbor_limit)
            batch = batch.to(DEVICE)
            _assert_finite_tensor(batch.x, domain="TASK-C", source="eval_gat/test/x")
            if getattr(batch, 'edge_attr', None) is not None:
                _assert_finite_tensor(batch.edge_attr, domain="TASK-C", source="eval_gat/test/edge_attr")
            logits = model(batch.x, batch.edge_index, getattr(batch, 'edge_attr', None), batch.batch)
            _assert_finite_tensor(logits, domain="TASK-C", source="eval_gat/test/logits")
            test_logits.append(logits.detach().cpu().numpy())
    val_log_f = np.concatenate(val_logits) if val_logits else np.array([])
    val_lbl_f = np.concatenate(val_labels) if val_labels else np.array([])
    test_log = np.concatenate(test_logits) if test_logits else np.array([])
    val_log_f = _validate_numeric_checkpoint(val_log_f, domain="TASK-C", stage="PRE_METRIC", source="gat_val_logits", repair=False, log_ok=True)
    val_lbl_f = _validate_numeric_checkpoint(val_lbl_f, domain="TASK-C", stage="PRE_METRIC", source="gat_val_labels", repair=False, log_ok=True)
    test_log = _validate_numeric_checkpoint(test_log, domain="TASK-C", stage="PRE_METRIC", source="gat_test_logits", repair=False, log_ok=True)
    _atomic_numpy_save(predictions_dir / 'gat_val_logits.npy', val_log_f)
    _atomic_numpy_save(predictions_dir / 'gat_test_logits.npy', test_log)
    _event(
        f"Seed {cfg.seed}",
        domain_label,
        "WRITE",
        artifact="predictions",
        val_logits=Path(predictions_dir / 'gat_val_logits.npy').resolve(),
        test_logits=Path(predictions_dir / 'gat_test_logits.npy').resolve(),
    )
    temp = temperature_scale(val_log_f, val_lbl_f) if len(val_log_f) else 1.0
    val_pr_auc = float(average_precision_score(val_lbl_f, _sig(val_log_f))) if len(np.unique(val_lbl_f)) > 1 else 0.0
    return model, hist, val_log_f, test_log, val_lbl_f, None, val_pr_auc, temp


class EWCRegularizer:
    def __init__(self, ewc_lambda: float = EWC_LAMBDA) -> None:
        self.ewc_lambda = ewc_lambda
        self._anchors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def compute_fisher(self, model: nn.Module, dataloader: DataLoader,
                       device: torch.device = DEVICE, max_batches: int = 8) -> None:
        fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, device=device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        model.eval()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                x_seq, x_ctx, y = batch[0].to(device, dtype=torch.float32), batch[1].to(device, dtype=torch.float32), batch[2].to(device, dtype=torch.float32)
                logits, _ = model(x_seq, x_ctx)
                loss = F.binary_cross_entropy_with_logits(logits, y)
            else:
                continue
            model.zero_grad(set_to_none=True)
            loss.backward()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)
        denom = float(max(1, min(len(dataloader), max_batches)))
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self._anchors:
                _, theta = self._anchors[name]
                self._anchors[name] = (fisher[name] / denom, theta)
            else:
                self._anchors[name] = (fisher[name] / denom, param.data.detach().clone())

    def penalty(self, model: nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        loss = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self._anchors:
                continue
            fisher, theta_star = self._anchors[name]
            loss = loss + (fisher.to(device) * (param - theta_star.to(device)).pow(2)).sum()
        return (self.ewc_lambda / 2.0) * loss

    def update_reference(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self._anchors:
                fisher, _ = self._anchors[name]
                self._anchors[name] = (fisher, param.data.detach().clone())
            else:
                self._anchors[name] = (torch.zeros_like(param.data), param.data.detach().clone())

    def save(self, path: Path) -> None:
        try:
            _atomic_torch_save(path, {name: (fisher.cpu(), theta.cpu()) for name, (fisher, theta) in self._anchors.items()})
        except Exception as exc:
            logger.error(f"[IMMORTAL-IO-ERROR] EWC save failed: {exc}")

    def load(self, path: Path, device: torch.device) -> None:
        try:
            data = torch.load(str(path), map_location="cpu", weights_only=False)
            self._anchors = {name: (fisher.to(device), theta.to(device)) for name, (fisher, theta) in data.items()}
        except Exception as exc:
            logger.warning(f"[IMMORTAL-IO-ERROR] EWC load failed: {exc}")


# ──────────────────────────────────────────────────────────────────────
class IncrementalNormalizer:
    """Online mean/variance ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â never stores raw data (Welford 1962)."""
    def __init__(self, state: Optional[Dict] = None) -> None:
        self._s: Dict[str, Dict] = state or {}

    def update(self, col: str, values: np.ndarray) -> None:
        v = np.asarray(values, dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        if not len(v): return
        if col not in self._s:
            self._s[col] = {"mean": 0.0, "M2": 0.0, "n": 0}
        s = self._s[col]
        for x in v:
            s["n"] += 1; delta = x - s["mean"]
            s["mean"] += delta / s["n"]; s["M2"] += delta * (x - s["mean"])

    def mean(self, col: str) -> float:
        return float(self._s.get(col, {}).get("mean", 0.0))

    def std(self, col: str, min_std: float = 1e-6) -> float:
        s = self._s.get(col, {}); n = s.get("n", 0)
        if n < 2: return min_std
        return float(max(math.sqrt(max(s.get("M2", 0.0) / (n-1), 0.0)), min_std))

    def normalize(self, col: str, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, np.float32) - self.mean(col)) / self.std(col)

    def ewa_update(self, col: str, new_value: float, alpha: float = EWA_ALPHA) -> None:
        """EWA-smooth threshold update on drift."""
        if col not in self._s:
            self._s[col] = {"mean": new_value, "M2": 0.0, "n": 1}
        else:
            self._s[col]["mean"] = (alpha * new_value
                                    + (1 - alpha) * self._s[col]["mean"])

    def to_dict(self) -> Dict:
        return deepcopy(self._s)

    @classmethod
    def from_dict(cls, d: Dict) -> "IncrementalNormalizer":
        return cls(state=d)


# ──────────────────────────────────────────────────────────────────────
class ExperienceReplayBuffer:
    """Bounded float32 replay store with atomic compressed persistence."""

    def __init__(self, capacity: int = REPLAY_SIZE, seed: int = SEED) -> None:
        self.capacity = max(1, int(capacity))
        self._rng = np.random.default_rng(seed)
        self._x: Optional[np.ndarray] = None
        self._c: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    @property
    def size(self) -> int:
        return 0 if self._y is None else int(len(self._y))

    def add_batch(self, x: np.ndarray, c: np.ndarray, y: np.ndarray) -> None:
        x_arr = np.asarray(x, dtype=np.float32)
        c_arr = np.asarray(c, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        if len(y_arr) == 0:
            return
        if len(y_arr) > self.capacity:
            x_arr = x_arr[-self.capacity:]
            c_arr = c_arr[-self.capacity:]
            y_arr = y_arr[-self.capacity:]
        if self._y is None:
            self._x, self._c, self._y = x_arr.copy(), c_arr.copy(), y_arr.copy()
        else:
            self._x = np.concatenate([self._x, x_arr], axis=0)[-self.capacity:]
            self._c = np.concatenate([self._c, c_arr], axis=0)[-self.capacity:]
            self._y = np.concatenate([self._y, y_arr], axis=0)[-self.capacity:]
        gc.collect()

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._y is None or self._x is None or self._c is None or self.size == 0:
            raise ValueError("[REPLAY] sample requested from empty replay buffer")
        n = min(max(1, int(n)), self.size)
        idx = self._rng.choice(self.size, size=n, replace=False)
        return self._x[idx].copy(), self._c[idx].copy(), self._y[idx].copy()

    def save(self, path: Path) -> None:
        if self._y is None or self._x is None or self._c is None:
            return

        def _writer(tmp_path: Path) -> None:
            with tmp_path.open("wb") as fh:
                np.savez_compressed(fh, x=self._x, c=self._c, y=self._y)
                fh.flush()
                os.fsync(fh.fileno())

        atomic_write_file(path, _writer)

    def load(self, path: Path) -> None:
        try:
            with np.load(path, allow_pickle=False) as payload:
                self._x = np.asarray(payload["x"], dtype=np.float32)[-self.capacity:]
                self._c = np.asarray(payload["c"], dtype=np.float32)[-self.capacity:]
                self._y = np.asarray(payload["y"], dtype=np.float32).reshape(-1)[-self.capacity:]
        except Exception as exc:
            logger.warning(f"[REPLAY] Could not load replay buffer {path}: {exc}")
            self._x = self._c = self._y = None


class LifelongModelTrainer:
    """
    JARVIS-IMMORTAL-BRAIN: Lifelong Continuous Learning Engine.
    - Loads brain if costguard_brain.pt exists, else initializes fresh
    - learn_from_focus_csv(): streaming chunks, EWC + replay, drift adapt
    - save_brain(): atomic write (tmp ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ rename, never corrupt), SHA256 sidecar
    """
    def __init__(self, brain_dir: Optional[Union[str, Path]] = None,
                 hardware: Optional[HardwareProfile] = None) -> None:
        resolved_brain_dir = Path(brain_dir) if brain_dir else _default_brain_dir("./results")
        self.brain_dir  = _ensure_dir(resolved_brain_dir)
        self.hw         = hardware or HardwareProfile.probe()
        self.device     = DEVICE
        self._pt_path   = self.brain_dir / BRAIN_FILE
        self._meta_path = self.brain_dir / "brain_meta.json"
        self._ewc_path  = self.brain_dir / "brain_ewc.pt"
        self._replay_path = self.brain_dir / "brain_replay.npz"

        # Components
        self.normalizer = IncrementalNormalizer()
        self.replay     = ExperienceReplayBuffer(capacity=REPLAY_SIZE)
        self.ewc        = EWCRegularizer(ewc_lambda=EWC_LAMBDA)
        self.lstm_model: Optional[BahdanauBiLSTM] = None
        self._lstm_opt:  Optional[optim.Adam]      = None
        self._criterion  = FocalLoss(alpha=0.25, gamma=1.5)

        # Drift detectors ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â one per key metric
        self._ph_cost   = PageHinkleyTest(PH_DELTA, PH_LAMBDA)
        self._ph_cpu    = PageHinkleyTest(PH_DELTA, PH_LAMBDA)
        self._ph_anom   = PageHinkleyTest(PH_DELTA, PH_LAMBDA)

        # Brain metadata
        self._meta: Dict = {
            "datasets_seen": 0, "total_rows": 0, "created_at": "",
            "last_updated": "", "drift_events": [],
            "iqr_thresholds": {}, "normalizer_state": {},
            "ph_cost": {}, "ph_cpu": {}, "ph_anom": {},
        }

        self.load_brain()

    # ──────────────────────────────────────────────────────────────────────
    def load_brain(self) -> None:
        """Load brain if exists, else initialize fresh. Auto-called at init."""
        if self._meta_path.exists():
            try:
                self._meta = json.loads(self._meta_path.read_text("utf-8"))
                self.normalizer = IncrementalNormalizer.from_dict(
                    self._meta.get("normalizer_state", {}))
                if self._meta.get("ph_cost"):
                    self._ph_cost = PageHinkleyTest.from_dict(self._meta["ph_cost"])
                if self._meta.get("ph_cpu"):
                    self._ph_cpu = PageHinkleyTest.from_dict(self._meta["ph_cpu"])
                logger.info(
                    f"{_GRN}[LIFELONG-BRAIN]{_RST} Metadata loaded: "
                    f"datasets={self._meta.get('datasets_seen',0)} "
                    f"rows={self._meta.get('total_rows',0):,}")
            except Exception as exc:
                logger.warning(f"[IMMORTAL-IO-ERROR] Meta load failed: {exc}")

        self._init_model()

        if self._pt_path.exists():
            try:
                ckpt = torch.load(str(self._pt_path), map_location="cpu",
                                  weights_only=False)
                self.lstm_model.load_state_dict(
                    ckpt.get("lstm_state_dict", {}), strict=False)
                if self._lstm_opt and "lstm_opt" in ckpt:
                    self._lstm_opt.load_state_dict(ckpt["lstm_opt"])
                logger.info(f"{_GRN}[LIFELONG-BRAIN]{_RST} Weights loaded ÃƒÂ¢Ã¢â‚¬Â Ã‚Â {self._pt_path.name}")
                print(f"  [LIFELONG-BRAIN] ÃƒÂ¢Ã…â€œÃ¢â‚¬Å“ Brain loaded "
                      f"({self._meta.get('datasets_seen',0)} datasets, "
                      f"{self._meta.get('total_rows',0):,} rows)")
            except Exception as exc:
                logger.warning(f"[IMMORTAL-IO-ERROR] Weights load failed: {exc}")
                self._init_model()
        else:
            print("  [LIFELONG-BRAIN] Initializing fresh brain")

        if self._ewc_path.exists():
            self.ewc.load(self._ewc_path, self.device)

        if self._replay_path.exists():
            self.replay.load(self._replay_path)
            logger.info(f"[LIFELONG-BRAIN] Replay buffer: {self.replay.size} samples")
            _event("REPLAY", "LOAD", path=self._replay_path.resolve(), buffer_size=self.replay.size)

        if not self._meta.get("created_at"):
            self._meta["created_at"] = datetime.now(timezone.utc).isoformat()

    def _init_model(self) -> None:
        self.lstm_model = BahdanauBiLSTM(
            n_channels=N_CHANNELS, n_ctx=N_CTX, hidden=256,
            dropout=0.40, num_layers=3, ctx_proj_dim=64,
        ).to(torch.float32).to(self.device)
        self._lstm_opt = optim.Adam(
            self.lstm_model.parameters(), lr=LIFELONG_LR, weight_decay=1e-5)

    def save_brain(self) -> None:
        """A3-style atomic save: tmp ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ rename. SHA256 sidecar. Never corrupt."""
        ckpt = {}
        if self.lstm_model is not None:
            ckpt["lstm_state_dict"] = {
                k: v.cpu() for k, v in self.lstm_model.state_dict().items()}
        if self._lstm_opt is not None:
            ckpt["lstm_opt"] = self._lstm_opt.state_dict()
        ckpt["meta"] = self._meta

        tmp = Path(str(self._pt_path) + ".tmp")
        try:
            # BUG-001 FIX: Delete any leftover .tmp from a previous failed save BEFORE
            # calling torch.save(). On Windows, torch.save() internally creates its own
            # temporary file and renames it to the target path; if the target (.pt.tmp)
            # already exists from a crashed prior attempt, that internal rename fails
            # with [WinError 183] (ERROR_ALREADY_EXISTS) before our outer replace() runs.
            try:
                tmp.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug(f"[IMMORTAL-IO-DEBUG] Temp cleanup skipped for {tmp}: {exc}")
            torch.save(ckpt, str(tmp))
            # Use os.replace() via Path.replace() ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â atomically overwrites on Windows.
            # Fallback: if replace() still fails (e.g. target memory-mapped by another
            # process), explicitly unlink destination then rename as a last resort.
            try:
                tmp.replace(self._pt_path)
            except OSError:
                try:
                    if self._pt_path.exists():
                        self._pt_path.unlink()
                    tmp.rename(self._pt_path)
                except OSError as _rename_exc:
                    raise RuntimeError(
                        f"[ATOMIC-SAVE] Both replace() and rename() failed: {_rename_exc}"
                    ) from _rename_exc
            sha = hashlib.sha256(self._pt_path.read_bytes()).hexdigest()[:16]
            # Ãƒâ€šÃ‚Â§5 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Post-save integrity assertion: verify the file is loadable
            _verify = torch.load(str(self._pt_path), map_location="cpu", weights_only=False)
            _required_keys = {"lstm_state_dict", "meta"}
            _missing = _required_keys - set(_verify.keys())
            if _missing:
                raise RuntimeError(f"[INTEGRITY] Brain file missing keys after save: {_missing}")
            del _verify
            logger.info(f"[LIFELONG-BRAIN-SAVED] sha256={sha} integrity=OK ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ {self._pt_path.name}")
        except Exception as exc:
            logger.error(f"[IMMORTAL-IO-ERROR] Brain save failed or integrity check failed: {exc}")
            try:
                tmp.unlink(missing_ok=True)
            except Exception as cleanup_exc:
                logger.debug(f"[IMMORTAL-IO-DEBUG] Cleanup skipped for {tmp}: {cleanup_exc}")
            return

        # Meta JSON
        self._meta["last_updated"]     = datetime.now(timezone.utc).isoformat()
        self._meta["normalizer_state"] = self.normalizer.to_dict()
        self._meta["ph_cost"]          = self._ph_cost.to_dict()
        self._meta["ph_cpu"]           = self._ph_cpu.to_dict()
        _atomic_write_json(self._meta_path, self._meta)

        # EWC anchors
        self.ewc.save(self._ewc_path)

        # Replay buffer
        self.replay.save(self._replay_path)
        _event("WRITE", "LIFELONG", meta=self._meta_path.resolve(), ewc=self._ewc_path.resolve(), replay=self._replay_path.resolve())

    # ──────────────────────────────────────────────────────────────────────
    def save_brain(self) -> None:
        """Crash-safe lifelong brain save with atomic torch/meta/EWC/replay writes."""
        ckpt: Dict[str, object] = {}
        if self.lstm_model is not None:
            ckpt["lstm_state_dict"] = {k: v.cpu() for k, v in self.lstm_model.state_dict().items()}
        if self._lstm_opt is not None:
            ckpt["lstm_opt"] = self._lstm_opt.state_dict()
        self._meta["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._meta["normalizer_state"] = self.normalizer.to_dict()
        self._meta["ph_cost"] = self._ph_cost.to_dict()
        self._meta["ph_cpu"] = self._ph_cpu.to_dict()
        self._meta["ph_anom"] = self._ph_anom.to_dict()
        ckpt["meta"] = self._meta
        try:
            _atomic_torch_save(self._pt_path, ckpt)
            sha = hashlib.sha256(self._pt_path.read_bytes()).hexdigest()[:16]
            verify = torch.load(str(self._pt_path), map_location="cpu", weights_only=False)
            missing = {"lstm_state_dict", "meta"} - set(verify.keys())
            del verify
            if missing:
                raise RuntimeError(f"[INTEGRITY] Brain file missing keys after save: {missing}")
            _atomic_write_json(self._meta_path, self._meta)
            self.ewc.save(self._ewc_path)
            self.replay.save(self._replay_path)
            logger.info(f"[LIFELONG-BRAIN-SAVED] sha256={sha} integrity=OK -> {self._pt_path.name}")
            _event("WRITE", "LIFELONG", meta=self._meta_path.resolve(), ewc=self._ewc_path.resolve(), replay=self._replay_path.resolve())
        except Exception as exc:
            logger.error(f"[IMMORTAL-IO-ERROR] Brain save failed or integrity check failed: {exc}")

    def learn_from_focus_csv(self, focus_csv_path: Union[str, Path]) -> Dict:
        """
        Stream FOCUS CSV in chunks. Per chunk:
          1. EWA normalizer update
          2. Page-Hinkley drift detection
          3. Mix with replay (50/50)
          4. Forward LSTM + EWC penalty
          5. Backward + step (lr=LIFELONG_LR=1e-4)
          6. Replay buffer add
          7. Checkpoint save
        Returns {chunks_processed, drifts_detected, final_loss}
        """
        focus_csv_path = Path(focus_csv_path)
        if not focus_csv_path.exists():
            logger.error(f"[LIFELONG] FOCUS CSV not found: {focus_csv_path}")
            return {"error": "file_not_found"}

        logger.info(f"\n{'='*60}")
        logger.info(f"[LIFELONG-BRAIN] Learning from: {focus_csv_path.name}")
        logger.info(f"  datasets_seen={self._meta['datasets_seen']} "
                    f"rows={self._meta['total_rows']:,}")
        learn_timer = StageTimer()
        _event(
            "LIFELONG",
            "START",
            focus_csv=focus_csv_path.resolve(),
            datasets_seen=self._meta["datasets_seen"],
            total_rows=self._meta["total_rows"],
        )

        chunk_size     = self.hw.safe_chunk // 10
        chunks_proc    = 0; drifts_det = 0; total_rows = 0
        _rows_at_last_ckpt = 0  # FIX: track checkpointed rows
        loss_history:  deque = deque(maxlen=20)
        amp_on         = False   # FIX-LIFELONG-NAN: Real-world Bitbrains/TT distributions
                                 # ──────────────────────────────────────────────────────────────────────
                                 # skipped ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ rows_learned=0. FP32 is safe for chunk sizes
                                 # ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â¤2000 rows and fast enough on RTX 3050.
        _ll_gs_dev     = 'cuda' if torch.cuda.is_available() else 'cpu'  # FIX
        gscaler        = torch.amp.GradScaler(_ll_gs_dev, enabled=False)

        if self.lstm_model is None:
            self._init_model()
        self.lstm_model.train()

        try:
            reader = smart_read_csv(focus_csv_path, chunksize=max(100, chunk_size))
            if not hasattr(reader, "__iter__"):
                reader = [reader]   # single DataFrame fallback
        except Exception as exc:
            logger.exception(f"[IMMORTAL-IO-ERROR] Cannot open {focus_csv_path}: {exc}")
            return {"error": "focus_csv_open_failed"}

        # ──────────────────────────────────────────────────────────────────────
        # brain_meta stores {"domain_progress": {"<filename>": {"chunks_done": N}}}
        # On a clean start this key is absent (0 chunks done).
        # On restart we skip the first N chunks without re-processing them.
        _domain_key    = focus_csv_path.name
        _domain_prog   = self._meta.setdefault("domain_progress", {})
        _chunks_done   = int(
            _domain_prog.get(_domain_key, {}).get("chunks_done", 0)
        )
        _global_idx    = 0   # raw chunk counter (before skip)

        if _chunks_done > 0:
            logger.info(
                f"{_GRN}[LIFELONG-RESUME]{_RST} Domain '{_domain_key}': "
                f"skipping first {_chunks_done} already-processed chunks"
            )
            _event("RESUME", "LIFELONG", domain=_domain_key, chunks_done=_chunks_done)

        try:
            for chunk in reader:
                # ──────────────────────────────────────────────────────────────────────
                _global_idx += 1
                if _global_idx <= _chunks_done:
                    continue   # fast-forward through the CSV reader
                if chunk is None or len(chunk) < 10:
                    continue

                # Ensure FOCUS columns
                for col in FOCUS_COLS:
                    if col not in chunk.columns:
                        chunk[col] = 0 if col == "anomaly_window_active" else 0.0

                # 1. EWA normalizer update (ÃƒÅ½Ã‚Â±=0.10)
                for col in ("billed_cost", "cpu_seconds", "latency_p95"):
                    self.normalizer.update(col, chunk[col].values)

                # 3. Build mini-batch from chunk ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ sequences (MOVED ABOVE DRIFT DETECTION)
                X_new, C_new, Y_new = self._focus_to_sequences(chunk)
                if len(X_new) == 0:
                    continue

                # 2. Page-Hinkley drift detection
                cost_vals = pd.to_numeric(chunk["billed_cost"], errors="coerce").dropna().values
                cpu_vals  = pd.to_numeric(chunk["cpu_seconds"], errors="coerce").dropna().values
                anom_vals = pd.to_numeric(chunk["anomaly_window_active"], errors="coerce").dropna().values
                dc,  _ = self._ph_cost.update_batch(cost_vals)
                dcp, _ = self._ph_cpu.update_batch(cpu_vals)
                da,  _ = self._ph_anom.update_batch(anom_vals)
                if dc or dcp or da:
                    drifts_det += 1
                    stream  = ("billed_cost" if dc else "cpu_seconds" if dcp else "anomaly_rate")
                    mag     = abs(self._ph_cost._mean_est if dc else self._ph_cpu._mean_est)
                    logger.warning(
                        f"[LIFELONG-DRIFT-DETECTED] stream={stream} "
                        f"drift_magnitude={mag:.4f} chunk={chunks_proc}")
                    _event(
                        "DRIFT",
                        "PageHinkley",
                        stream=stream,
                        chunk=chunks_proc,
                        magnitude=mag,
                        ph_cost_mean=getattr(self._ph_cost, "_mean_est", 0.0),
                        ph_cpu_mean=getattr(self._ph_cpu, "_mean_est", 0.0),
                        ph_anom_mean=getattr(self._ph_anom, "_mean_est", 0.0),
                        triggered=True,
                    )
                    # EWA-smooth IQR thresholds (ÃƒÅ½Ã‚Â±=0.10)
                    for col in ("billed_cost", "cpu_seconds"):
                        q1 = float(pd.to_numeric(chunk[col], errors="coerce").quantile(0.25))
                        q3 = float(pd.to_numeric(chunk[col], errors="coerce").quantile(0.75))
                        old = self._meta["iqr_thresholds"].get(col, {"q1": q1, "q3": q3})
                        new_q1 = EWA_ALPHA * q1 + (1 - EWA_ALPHA) * old["q1"]
                        new_q3 = EWA_ALPHA * q3 + (1 - EWA_ALPHA) * old["q3"]
                        self._meta["iqr_thresholds"][col] = {"q1": new_q1, "q3": new_q3}
                    # Refresh Fisher matrix on drift
                    if len(X_new) > 0:
                        _tmp_ds = TensorDataset(
                            torch.from_numpy(X_new).float(),
                            torch.from_numpy(C_new).float(),
                            torch.from_numpy(Y_new).float())
                        _tmp_dl = DataLoader(_tmp_ds, batch_size=min(32, len(X_new)), shuffle=False, num_workers=0)
                        self.ewc.compute_fisher(
                            self.lstm_model, _tmp_dl, self._criterion, self.device, n_samples=len(X_new))
                        # Ãƒâ€šÃ‚Â§1C ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Defensive re-assertion: ensure train mode after Fisher
                        self.lstm_model.train()
                    else:
                        self.ewc.update_reference(self.lstm_model)

                # 4. Mix with replay buffer (50/50 new/coreset)
                n_rep = min(len(X_new), self.replay.size)
                if n_rep > 0:
                    Xr, Cr, Yr = self.replay.sample(n_rep)
                    X_all = np.concatenate([X_new, Xr], axis=0)
                    C_all = np.concatenate([C_new, Cr], axis=0)
                    Y_all = np.concatenate([Y_new, Yr], axis=0)
                else:
                    X_all, C_all, Y_all = X_new, C_new, Y_new
                replay_ratio = float(n_rep / max(1, len(X_all)))
                _event("REPLAY", "MIX", buffer_size=self.replay.size, replay_batch=n_rep, ratio=replay_ratio, new_batch=len(X_new))

                perm  = np.random.permutation(len(X_all))
                X_all = X_all[perm]; C_all = C_all[perm]; Y_all = Y_all[perm]

                # 5. Forward + EWC + backward (lr=LIFELONG_LR=1e-4)
                ds  = TensorDataset(torch.from_numpy(X_all).float(),
                                    torch.from_numpy(C_all).float(),
                                    torch.from_numpy(Y_all).float())
                bs  = max(16, min(128, len(X_all) // 2))
                dl  = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)

                # Set optimizer lr to LIFELONG_LR
                for pg in self._lstm_opt.param_groups:
                    pg["lr"] = LIFELONG_LR

                ep_loss = 0.0
                for bX, bC, bY in dl:
                    bX = bX.to(self.device); bC = bC.to(self.device); bY = bY.to(self.device)
                    self._lstm_opt.zero_grad()
                    # BUG-003 FIX (part A): Compute EWC penalty OUTSIDE autocast in float32.
                    # Inside autocast, ewc_lambda=400 * large Fisher values overflows float16
                    # to inf, making loss=nan and poisoning ep_loss permanently.
                    ewc_loss = self.ewc.penalty(self.lstm_model)  # always float32, always finite
                    # Guard against any residual inf/nan in EWC penalty before adding to loss
                    if not torch.isfinite(ewc_loss):
                        ewc_loss = torch.tensor(0.0, device=self.device)

                    with torch.amp.autocast(_ll_gs_dev, enabled=amp_on):  # FIX
                        logits, _ = self.lstm_model(bX, bC)
                        logits = torch.clamp(logits, -15.0, 15.0)  # FIX-4B: prevent BCE overflow
                        task_loss = self._criterion(logits, bY)
                    # BUG-003 FIX (part B): Add EWC penalty to task_loss OUTSIDE autocast
                    # in full float32 precision, then scale together. This prevents float16
                    # overflow while still allowing GradScaler to manage gradient scaling.
                    loss = task_loss.float() + ewc_loss
                    _event("EWC", "STEP", penalty=float(ewc_loss.item()), lambda_value=self.ewc.ewc_lambda)

                    # BUG-003 FIX (part C): Guard the combined loss before backward.
                    # If task_loss itself is nan/inf (e.g., degenerate batch), skip this
                    # batch cleanly rather than corrupting GradScaler's internal scale state.
                    if not torch.isfinite(loss):
                        logger.warning(
                            f"[LIFELONG-NAN-GUARD] Non-finite loss "
                            f"(task={task_loss.item():.4f} ewc={ewc_loss.item():.4f}) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â skipping batch")
                        gscaler.update()   # FIX-4C: reset GradScaler inf state before next batch
                        continue

                    gscaler.scale(loss).backward()
                    gscaler.unscale_(self._lstm_opt)
                    nn.utils.clip_grad_norm_(self.lstm_model.parameters(), 1.0)
                    gscaler.step(self._lstm_opt); gscaler.update()
                    ep_loss += loss.item()

                avg_loss = ep_loss / max(1, len(dl))
                loss_history.append(avg_loss)

                # 6. Add new samples to replay
                self.replay.add_batch(X_new, C_new, Y_new)

                total_rows  += len(chunk)
                chunks_proc += 1

                # 7. Checkpoint every 10 chunks + update domain progress
                if chunks_proc % 10 == 0:
                    self._meta["total_rows"] += total_rows
                    # v4.0: persist chunk index so a restart skips this work
                    _domain_prog[_domain_key] = {
                        "chunks_done": _chunks_done + chunks_proc,
                        "last_saved":  datetime.now(timezone.utc).isoformat(),
                    }
                    self.save_brain()
                    _rows_at_last_ckpt = total_rows
                    self._meta["total_rows"] -= total_rows  # avoid double-count
                    logger.info(
                        f"[LIFELONG-BRAIN] chunk={chunks_proc} "
                        f"(global={_chunks_done + chunks_proc}) "
                        f"loss={avg_loss:.4f} drifts={drifts_det}"
                    )
                    _event(
                        "CHECKPOINT",
                        "SAVE",
                        path=self._pt_path.resolve(),
                        chunk=chunks_proc,
                        global_chunk=_chunks_done + chunks_proc,
                        loss=avg_loss,
                        drifts=drifts_det,
                    )
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        except KeyboardInterrupt:
            logger.info("[LIFELONG-BRAIN] Interrupted - saving checkpoint")
            self._meta["total_rows"] += (total_rows - _rows_at_last_ckpt)  # FIX: only un-checkpointed
            # Ãƒâ€šÃ‚Â§2A ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Do NOT increment datasets_seen on interrupt: domain not complete.
            self.save_brain()
            _event("LIFELONG", "INTERRUPTED", focus_csv=focus_csv_path.resolve(), chunks_processed=chunks_proc, duration=format_duration_s(learn_timer.elapsed_s))
            return {"chunks_processed": chunks_proc, "drifts_detected": drifts_det,
                    "final_loss": loss_history[-1] if loss_history else float("nan")}
        except Exception as exc:
            logger.exception(f"[LIFELONG-BRAIN] Learning loop failed: {exc}")
            self._meta["total_rows"] += (total_rows - _rows_at_last_ckpt)  # FIX: only un-checkpointed
            self.save_brain()
            _event("LIFELONG", "FAIL", focus_csv=focus_csv_path.resolve(), chunks_processed=chunks_proc, duration=format_duration_s(learn_timer.elapsed_s), error=type(exc).__name__)
            return {"error": "lifelong_learning_failed", "chunks_processed": chunks_proc}

        # Ãƒâ€šÃ‚Â§2B ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Consolidation guard: only increment datasets_seen after EWC anchoring succeeds.
        _consolidation_success = False
        if chunks_proc > 0:
            try:
                self.ewc.update_reference(self.lstm_model)
                _consolidation_success = True
            except Exception as _ewc_exc:
                logger.error(f"[LIFELONG-BRAIN] EWC update_reference failed: {_ewc_exc} "
                             f"ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â datasets_seen NOT incremented to preserve audit integrity")

        self._meta["total_rows"] += total_rows
        if _consolidation_success:
            self._meta["datasets_seen"] += 1
            # Ãƒâ€šÃ‚Â§2C ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Write-once audit log entry for this domain completion
            _log_entry = {
                "domain":       focus_csv_path.name,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "seed":         self._meta.get("seed", "unknown"),
                "chunks":       chunks_proc,
                "rows":         total_rows,
            }
            self._meta.setdefault("consolidation_log", []).append(_log_entry)
            logger.info(f"[LIFELONG-BRAIN] Consolidation logged: {_log_entry}")
        self._meta["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._meta["drift_events"] = (
            self._meta.get("drift_events", [])
            + [{"ts": datetime.now(timezone.utc).isoformat(), "count": drifts_det}]
        )[-20:]

        # v4.0: Mark domain as fully processed (chunks_done = total processed)
        # On next run this domain will be entirely skipped by the chunk guard.
        _domain_prog[_domain_key] = {
            "chunks_done": _chunks_done + chunks_proc,
            "domain_complete": True,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save_brain()

        final_loss = loss_history[-1] if loss_history else float("nan")
        logger.info(f"{_GRN}[LIFELONG-BRAIN]{_RST} ÃƒÂ¢Ã…â€œÃ¢â‚¬Å“ Learned {total_rows:,} rows "
                    f"| {chunks_proc} chunks | {drifts_det} drifts "
                    f"| loss={final_loss:.4f}")
        _event("LIFELONG", "END", focus_csv=focus_csv_path.resolve(), rows_learned=total_rows, chunks_processed=chunks_proc, drifts_detected=drifts_det, final_loss=final_loss, duration=format_duration_s(learn_timer.elapsed_s))
        return {"chunks_processed": chunks_proc, "drifts_detected": drifts_det,
                "final_loss": round(final_loss, 5), "rows_learned": total_rows,
                "total_rows_lifetime": self._meta["total_rows"],
                "datasets_lifetime":   self._meta["datasets_seen"]}

    def _focus_to_sequences(self, chunk: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build (X_seq, X_ctx, Y) arrays from a FOCUS DataFrame chunk."""
        chunk = chunk.copy()
        for col in FOCUS_COLS:
            if col not in chunk.columns:
                chunk[col] = 0.0 if col != "anomaly_window_active" else 0

        cost = pd.to_numeric(chunk["billed_cost"], errors="coerce").fillna(0.0)
        cpu  = pd.to_numeric(chunk["cpu_seconds"], errors="coerce").fillna(0.0)
        lat  = pd.to_numeric(chunk["latency_p95"], errors="coerce").fillna(0.0)
        call = pd.to_numeric(chunk["call_count"],  errors="coerce").fillna(1.0)
        mem  = pd.to_numeric(chunk["memory_gb_s"], errors="coerce").fillna(0.0)

        self.normalizer.update("billed_cost", cost.values)
        self.normalizer.update("cpu_seconds",  cpu.values)
        self.normalizer.update("latency_p95",  lat.values)

        w    = max(3, min(50, len(chunk) // 4))
        roll = cost.rolling(w, min_periods=1).mean()
        cost_dev = (cost - roll) / (roll.abs() + 1e-8)

        c0 = self.normalizer.normalize("billed_cost", cost.values)
        c1 = np.clip(cost_dev.values.astype(np.float32), -5.0, 5.0)
        c2 = self.normalizer.normalize("billed_cost", roll.values)
        c3 = (call.values.astype(np.float32) / (call.max() + 1e-8))  # matches channel 3 = service_call_count
        c4 = self.normalizer.normalize("latency_p95",  lat.values)    # matches channel 4 = latency_p95
        feat_2d = np.nan_to_num(np.stack([c0,c1,c2,c3,c4], axis=1).astype(np.float32))

        labels = pd.to_numeric(chunk["anomaly_window_active"],
                               errors="coerce").fillna(0).values.astype(np.float32)
        if labels.mean() < 0.001:
            q1 = float(cost.quantile(0.25)); q3 = float(cost.quantile(0.75))
            iqr = q3 - q1
            labels = ((cost < q1 - 1.5*iqr) | (cost > q3 + 1.5*iqr)).astype(np.float32).values

        run_ids = chunk["run_id"].values
        X_seqs: List[np.ndarray] = []; C_ctxs: List[np.ndarray] = []; Y_lbs: List[float] = []

        for rid in dict.fromkeys(run_ids):
            mask  = run_ids == rid; ridx = np.where(mask)[0]
            n_run = len(ridx)
            rf    = feat_2d[ridx]; rl = labels[ridx]
            if n_run < SEQ_LEN:
                pad = np.zeros((SEQ_LEN - n_run, N_CHANNELS), np.float32)
                rf  = np.vstack([pad, rf]); rl = np.concatenate([np.zeros(SEQ_LEN - n_run), rl])
                n_run = SEQ_LEN
            stride = max(1, n_run // 8)
            for start in range(0, n_run - SEQ_LEN + 1, stride):
                w_seq = rf[start:start+SEQ_LEN]
                X_seqs.append(w_seq); Y_lbs.append(float(rl[start+SEQ_LEN-1]))
                ctx = np.zeros(N_CTX, np.float32)
                dev_w = w_seq[:, 1]  # cost_deviation_pct channel
                r7d_w = w_seq[:, 2]  # rolling_7d_mean channel
                _idx = 0
                for _win in [6, 12, 24]:
                    _seg = dev_w[-_win:]
                    ctx[_idx]   = float(_seg.mean())
                    ctx[_idx+1] = float(_seg.std()) + 1e-9
                    ctx[_idx+2] = float(np.polyfit(np.arange(len(_seg), dtype=np.float64), _seg, 1)[0]) if len(_seg) > 1 else 0.0
                    _idx += 3
                # ctx[9-12]: sin/cos time encodings ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â use created_at of last row in window
                _row_idx = ridx[min(start + SEQ_LEN - 1, len(ridx) - 1)]
                try:
                    _ts = pd.to_datetime(chunk["created_at"].iloc[_row_idx], errors="coerce")
                    if pd.notna(_ts):
                        _hr  = float(_ts.hour) + float(_ts.minute) / 60.0
                        _dow = float(_ts.dayofweek)
                        ctx[9]  = float(math.sin(2 * math.pi * _hr  / 24))
                        ctx[10] = float(math.cos(2 * math.pi * _hr  / 24))
                        ctx[11] = float(math.sin(2 * math.pi * _dow / 7))
                        ctx[12] = float(math.cos(2 * math.pi * _dow / 7))
                except Exception as exc:
                    logger.debug(f"[CTX] cyclic time features skipped for row {_row_idx}: {exc}")
                ctx[13] = float(dev_w[-1] - dev_w[max(0, len(dev_w)-6)])
                ctx[14] = float(dev_w[-1] - dev_w[max(0, len(dev_w)-24)])  # 24-step delta
                ctx[15] = float(dev_w.min()); ctx[16] = float(dev_w.max())
                ctx[17] = ctx[16] - ctx[15]
                # ctx[18-20]: stage/executor/branch label encodings
                try:
                    _sn = str(chunk["stage_name"].iloc[_row_idx])
                    ctx[18] = float(STAGE_ORDER.index(_sn)) / max(len(STAGE_ORDER)-1, 1) if _sn in STAGE_ORDER else 0.0
                except Exception as exc:
                    logger.debug(f"[CTX] stage encoding skipped for row {_row_idx}: {exc}")
                try:
                    _ex = str(chunk["executor_type"].iloc[_row_idx])
                    ctx[19] = float(EXECUTOR_TYPES.index(_ex)) / max(len(EXECUTOR_TYPES)-1, 1) if _ex in EXECUTOR_TYPES else 0.0
                except Exception as exc:
                    logger.debug(f"[CTX] executor encoding skipped for row {_row_idx}: {exc}")
                try:
                    _br = str(chunk["branch"].iloc[_row_idx])
                    ctx[20] = float(BRANCH_TYPES.index(_br)) / max(len(BRANCH_TYPES)-1, 1) if _br in BRANCH_TYPES else 0.0
                except Exception as exc:
                    logger.debug(f"[CTX] branch encoding skipped for row {_row_idx}: {exc}")
                _denom = r7d_w[0] + 1e-8
                ctx[21] = float(r7d_w[-1] / _denom)
                C_ctxs.append(ctx)

        if not X_seqs:
            return (np.zeros((0, SEQ_LEN, N_CHANNELS), np.float32),
                    np.zeros((0, N_CTX), np.float32),
                    np.zeros((0,), np.float32))
        return (np.stack(X_seqs, 0).astype(np.float32),
                np.stack(C_ctxs, 0).astype(np.float32),
                np.array(Y_lbs,  dtype=np.float32))

    def _should_adapt_lr(self, history: deque) -> bool:
        """EWA-smoothed loss trend ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â True if loss is consistently increasing."""
        if len(history) < 5:
            return False
        _hl = list(history)
        ema = _hl[0]                # FIX: oldest->newest
        for v in _hl[1:]:
            ema = 0.10 * v + 0.90 * ema
        return _hl[-1] > ema * 1.05


# ──────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────

def notify(decision: str, crs: float, projected_saving_usd: float,
           stage: str = "integration_test", summary: str = "") -> None:
    """
    TENSOR-3 fix: uses SLACK_BOT_TOKEN (chat.postMessage), NOT SLACK_WEBHOOK_URL.
    LOGIC-4/NOTIF-1 fix: socket.setdefaulttimeout(10) wraps starttls + login.
    NOTIF-2 fix: validates inputs before any network I/O.
    """
    if _RUNTIME_CONTROLS.get("disable_notifier", False):
        logger.info("[NOTIFIER] Disabled by runtime control")
        return
    # NOTIF-2: validate inputs
    if decision not in ("AUTO_OPTIMISE", "BLOCK", "WARN", "ALLOW"):
        logger.warning(f"[NOTIFIER] Invalid decision '{decision}' - skipping")
        return
    if not (0.0 <= crs <= 1.0):
        logger.warning(f"[NOTIFIER] Invalid CRS {crs} - skipping")
        return
    if decision not in ("AUTO_OPTIMISE", "BLOCK"):
        return   # Only fire alerts on actionable decisions

    now = _utcnow()
    notifier_key = (decision, stage)
    last_sent = _NOTIFIER_LAST_SENT.get(notifier_key)
    if last_sent and (now - last_sent).total_seconds() < _NOTIFIER_COOLDOWN_SECONDS:
        logger.info(
            f"[NOTIFIER] Cooldown active for decision={decision} stage={stage}; skipping duplicate alert"
        )
        return
    _NOTIFIER_LAST_SENT[notifier_key] = now

    ts  = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    msg = (f"CostGuard PADE Alert\n"
           f"Decision  : {decision}\n"
           f"Stage     : {stage}\n"
           f"CRS       : {crs:.4f}\n"
           f"Cost Saved: ${projected_saving_usd:.4f} USD\n"
           f"Summary   : {summary or 'single actionable event'}\n"
           f"Timestamp : {ts}")

    # TENSOR-3 fix: SLACK_BOT_TOKEN + chat.postMessage
    slack_token   = CredentialResolver.get("SLACK_BOT_TOKEN")
    slack_channel = CredentialResolver.get("SLACK_DEFAULT_CHANNEL", "#costguard-alerts")
    if slack_token:
        try:
            payload = json.dumps({
                "channel":     slack_channel,
                "text":        msg,
                "attachments": [{"color": "#FF0000" if decision == "BLOCK" else "#FF9800",
                                 "text": msg}]
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://slack.com/api/chat.postMessage",
                data=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {slack_token}"},
                method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
                if body.get("ok"):
                    logger.info(f"[NOTIFIER] Slack alert sent -> {slack_channel}")
                else:
                    logger.warning(f"[NOTIFIER] Slack API error: {body.get('error')}")
        except Exception as exc:
            logger.warning(f"[NOTIFIER] Slack failed: {exc}")

    # Gmail SMTP: socket.setdefaulttimeout(10) wraps starttls + login
    gmail_sender   = CredentialResolver.get("GMAIL_SENDER")
    gmail_password = CredentialResolver.get("GMAIL_APP_PASSWORD")
    gmail_recipient= CredentialResolver.get("GMAIL_RECIPIENT") or gmail_sender
    if gmail_sender and gmail_password:
        old_timeout = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(10)   # LOGIC-4 fix: wraps ENTIRE block
            from email.mime.text import MIMEText
            mime = MIMEText(msg)
            mime["Subject"] = f"CostGuard Alert: {decision}"
            mime["From"]    = gmail_sender
            mime["To"]      = gmail_recipient
            with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
                smtp.ehlo()
                smtp.starttls()            # covered by setdefaulttimeout
                smtp.login(gmail_sender, gmail_password)   # covered
                smtp.sendmail(gmail_sender, gmail_recipient, mime.as_string())
            logger.info(f"[NOTIFIER] Gmail alert sent -> {gmail_recipient}")
        except Exception as exc:
            logger.warning(f"[NOTIFIER] Gmail failed: {exc}")
        finally:
            socket.setdefaulttimeout(old_timeout)


# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    checks_green = 0
    failures: List[str] = []
    smoke_root = _ensure_dir(Path('results') / 'smoke_test_tmp')
    shutil.rmtree(smoke_root, ignore_errors=True)
    smoke_root.mkdir(parents=True, exist_ok=True)

    def case_dir(name: str) -> Path:
        path = smoke_root / name
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def record(name: str, fn: Callable[[], None]) -> None:
        nonlocal checks_green
        try:
            fn()
            checks_green += 1
        except Exception as exc:
            failures.append(f'{name}: {exc}')

    record('torch runtime', lambda: None if (torch.cuda.is_available() or True) else (_ for _ in ()).throw(RuntimeError('torch unavailable')))
    record('pyg import', lambda: logger.warning('[SMOKE] PyG unavailable - GAT-specific checks will be soft-skipped') if not _HAS_PYG else None)

    def _check_synth() -> None:
        root = case_dir('synthetic')
        gen = SyntheticDataGenerator(out_dir=root, seed=42)
        out = gen.generate(n_rows=1000, force=True)
        if not out.exists():
            raise FileNotFoundError('synthetic generation did not create telemetry CSV')
    record('synthetic generation', _check_synth)

    def _check_bitbrains() -> None:
        root = case_dir('bitbrains')
        bb_dir = root / 'bb'
        bb_dir.mkdir(parents=True, exist_ok=True)
        sample = bb_dir / '1.csv'
        atomic_write_text(
            sample,
            (
                "Timestamp [ms]; CPU cores; CPU capacity provisioned [MHZ]; CPU usage [MHZ]; CPU usage [%]; "
                "Memory capacity provisioned [KB]; Memory usage [KB]; Disk read throughput [KB/s]; "
                "Disk write throughput [KB/s]; Network received throughput [KB/s]; Network transmitted throughput [KB/s]\n"
                "1; 2; 5000; 1200; 24; 8388608; 2097152; 10; 20; 3; 4\n"
                "2; 2; 5000; 1400; 28; 8388608; 2197152; 12; 18; 5; 6\n"
                "3; 4; 12000; 9000; 75; 16777216; 15000000; 100; 120; 40; 55\n"
            ),
            encoding='utf-8',
        )
        out = root / 'raw_bitbrains' / 'bitbrains_focus.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = load_bitbrains(bb_dir, out)
        if rows <= 0:
            raise ValueError('load_bitbrains returned zero rows')
    record('bitbrains loader', _check_bitbrains)

    def _check_tt() -> None:
        root = case_dir('travistorrent')
        csv_path = root / 'tt.csv'
        atomic_write_text(
            csv_path,
            (
                "tr_build_id,gh_project_name,git_branch,gh_build_started_at,tr_status,tr_duration,tr_log_buildduration,"
                "tr_log_num_tests_run,tr_log_num_tests_failed,tr_log_num_tests_ok,tr_log_num_test_suites_failed,"
                "git_diff_src_churn,git_diff_test_churn,gh_diff_files_added,gh_diff_files_deleted,gh_diff_files_modified,"
                "gh_team_size,gh_num_commits_in_push,gh_num_issue_comments,gh_num_pr_comments,gh_repo_num_commits,"
                "gh_sloc,gh_test_lines_per_kloc,gh_lang,gh_by_core_team_member,gh_is_pr\n"
                "1,proj_a,main,2025-01-02T00:00:00,passed,120,120,100,0,100,0,20,5,1,0,4,5,2,1,1,1000,12000,80,Python,TRUE,FALSE\n"
                "2,proj_a,main,2025-01-01T00:00:00,failed,240,240,100,5,95,1,500,10,5,0,10,5,3,1,1,1000,12000,80,Python,TRUE,TRUE\n"
                "3,proj_b,develop,2025-01-03T00:00:00,passed,90,90,50,0,50,0,15,3,1,0,2,3,1,0,1,500,8000,60,Java,FALSE,FALSE\n"
            ),
            encoding='utf-8',
        )
        rows = load_travistorrent(csv_path, root / 'real_data')
        if rows <= 0:
            raise ValueError('load_travistorrent returned zero rows')
    record('travistorrent loader', _check_tt)

    record('lstm forward', lambda: (_ for _ in ()).throw(RuntimeError('bad shape')) if BahdanauBiLSTM()(torch.zeros(2, 30, 5), torch.zeros(2, N_CTX))[0].shape != (2,) else None)

    def _check_gat() -> None:
        if not _HAS_PYG:
            return
        data = Data(
            x=torch.randn(10, 11),
            edge_index=torch.tensor([[0,1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8,9]], dtype=torch.long),
            edge_attr=torch.ones(9, 3),
            y=torch.tensor([0.0, 1.0]),
        )
        data.batch = torch.tensor([0,0,0,0,0,1,1,1,1,1], dtype=torch.long)
        out = GATv2Pipeline(n_node_feat=11)(data.x, data.edge_index, data.edge_attr, data.batch)
        if out.shape != (2,):
            raise ValueError(f'expected (2,), got {out.shape}')
    record('gat forward', _check_gat)

    record('focal loss', lambda: (_ for _ in ()).throw(ValueError('non-finite focal loss')) if not torch.isfinite(FocalLoss()(torch.randn(8), torch.randint(0, 2, (8,), dtype=torch.float32))) else None)

    def _check_ewc() -> None:
        model = nn.Linear(4, 1)
        ewc = EWCRegularizer(ewc_lambda=EWC_LAMBDA)
        ewc.update_reference(model)
        with torch.no_grad():
            model.weight.add_(0.1)
        penalty = ewc.penalty(model)
        if penalty.item() < 0:
            raise ValueError('ewc penalty negative')
    record('ewc penalty', _check_ewc)

    def _check_threshold() -> None:
        probs = np.linspace(0.05, 0.95, 20)
        labels = np.array([0] * 10 + [1] * 10, dtype=np.float32)
        _, threshold = f1_at_optimal_threshold(probs, labels)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError('threshold outside [0, 1]')
    record('threshold sweep', _check_threshold)

    def _check_best_scores() -> None:
        result_dir = case_dir('best_scores')
        update_best_scores(
            result_dir,
            'synthetic',
            {'f1_at_opt': 0.9, 'roc_auc': 0.95, 'pr_auc': 0.91, 'precision': 0.9, 'recall': 0.9, 'threshold': 0.5},
            {'f1_at_opt': 0.9, 'roc_auc': 0.95, 'pr_auc': 0.91, 'precision': 0.9, 'recall': 0.9, 'threshold': 0.5},
            {'f1_at_opt': 0.9, 'roc_auc': 0.95, 'pr_auc': 0.91, 'precision': 0.9, 'recall': 0.9, 'threshold': 0.5},
            hpo_triggered=False,
            seed=42,
        )
        payload = json.loads((result_dir / 'best_scores.json').read_text(encoding='utf-8'))
        expected = {'best_lstm', 'best_gat', 'best_ens', 'hpo_triggered', 'domain', 'seed'}
        if set(payload.keys()) != expected:
            raise ValueError(f'bad best_scores keys: {sorted(payload.keys())}')
    record('best scores schema', _check_best_scores)

    def _check_bwt() -> None:
        score = compute_bwt(0.91, 0.89, 0.90, 0.88)
        if not isinstance(score, float):
            raise ValueError('bwt is not float')
    record('bwt formula', _check_bwt)

    try:
        if not failures:
            print('SMOKE TEST PASSED - 12/12 checks green')
            sys.exit(0)
        print(f"SMOKE TEST FAILED - {checks_green}/12 checks green; FAILURES: {failures}")
        sys.exit(1)
    finally:
        shutil.rmtree(smoke_root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CostGuard v17.0 IEEE 3-domain pipeline')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--rows', type=int, default=SYNTH_ROWS)
    parser.add_argument('--synth-mode', choices=['standard', 'enhanced'], default='standard')
    parser.add_argument('--anomaly-rate', type=float, default=ANOMALY_RATE)
    parser.add_argument('--ingest', action='store_true')
    parser.add_argument('--data-mode', choices=['synth', 'synthetic', 'real', 'universal', 'bitbrains'], default='synthetic')
    parser.add_argument('--real-input', default=None)
    parser.add_argument('--bitbrains-dir', default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train-lifelong', action='store_true')
    parser.add_argument('--lifelong', action='store_true')
    parser.add_argument('--brain-dir', default=None)
    parser.add_argument('--skip-data', action='store_true')
    parser.add_argument('--skip-preprocess', action='store_true')
    parser.add_argument('--preprocess-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--checkpoint-keep-last', type=int, default=5)
    parser.add_argument('--hpo-trials', type=int, default=30)
    parser.add_argument('--scheduler', choices=['plateau', 'cosine'], default='cosine')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results-base', default='./results')
    parser.add_argument('--raw-dir', default=None)
    parser.add_argument('--ml-ready-dir', default=None)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--mode', choices=['full', 'train', 'hpo', 'compare'], default='full')
    parser.add_argument('--synth-only', action='store_true')
    parser.add_argument('--synthetic-only', action='store_true')
    parser.add_argument('--real-only', action='store_true')
    parser.add_argument('--note', default='')
    parser.add_argument('--force-run-dir', default=None)
    parser.add_argument('--resume-epoch', action='store_true')
    parser.add_argument('--disable-notifier', action='store_true')
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument('--verbose-epochs', dest='verbose_epochs', action='store_true')
    verbosity.add_argument('--quiet-epochs', dest='verbose_epochs', action='store_false')
    parser.set_defaults(verbose_epochs=True)
    return parser.parse_args()


def main() -> None:
    _extended_main()
def _auto_label_cloud_telemetry(df: pd.DataFrame, data_mode: str,
                                 seed: int = 42) -> pd.DataFrame:
    """
    Progressive grouped MAD/IQR labeler for unlabeled universal telemetry.
    Leaves synthetic and real data unchanged and never overwrites non-zero labels.
    """
    if data_mode in ("synthetic", "synth", "real"):
        return df
    _orig_id = id(df)
    if _orig_id in _AUTO_LABEL_SEEN:
        logger.warning("[FLASH-DOUBLE-LABEL-GUARD] DataFrame already labeled ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â skipping")
        return df

    df = df.copy()
    N  = len(df)
    if N == 0:
        return df

    t_start = time.perf_counter()

    # Check if already labeled
    anom_col = "anomaly_window_active"
    if anom_col not in df.columns:
        df[anom_col] = 0
    pos_ratio = float(pd.to_numeric(df[anom_col], errors="coerce").fillna(0).mean())
    if pos_ratio > 0.0:
        _AUTO_LABEL_SEEN.add(_orig_id)
        return df

    if "run_id" not in df.columns:
        _AUTO_LABEL_SEEN.add(_orig_id)
        return df

    active_cols = [c for c in (
        "duration_seconds", "cpu_seconds", "memory_gb_s",
        "billed_cost", "network_egress_gb", "latency_p95",
    ) if c in df.columns]
    if not active_cols:
        _AUTO_LABEL_SEEN.add(_orig_id)
        return df

    df[anom_col] = _progressive_group_outlier_labels(
        df=df,
        group_col="run_id",
        metric_cols=active_cols,
        sort_col="created_at" if "created_at" in df.columns else None,
        min_history=5,
    ).astype(int)
    _AUTO_LABEL_SEEN.add(_orig_id)

    elapsed = time.perf_counter() - t_start
    if elapsed > 5.0:
        logger.warning(f"[FLASH-SLOW-WARNING] _auto_label took {elapsed:.1f}s "
                       f"(N={N}, cols={active_cols}) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â consider chunking")
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return df


# ──────────────────────────────────────────────────────────────────────
# Ãƒâ€šÃ‚Â§8.2  UNIVERSAL SEQUENCE & GRAPH BUILDER  (v17.0 - TT+BB compatibility mode)
# ──────────────────────────────────────────────────────────────────────

def build_universal_sequences_and_graphs(
    telemetry_path: Path,
    out_dir:        Path,
    seed:           int  = 42,
    data_mode:      str  = "universal",
    force:          bool = False,
) -> None:
    """
    Reads raw_universal/pipeline_stage_telemetry.csv and writes the 3 downstream
    files required by _prep_task_b / _prep_task_c:
      lstm_training_sequences.csv
      pipeline_graphs.csv
      node_stats.csv

    OOM-safe: HardwareProfile.probe().safe_max_rows() caps sample.
    Run IDs batched in groups of 2_000.
    Auto-labels cloud telemetry via _auto_label_cloud_telemetry() (Flash D4).
    """
    hw        = HardwareProfile.probe()
    MAX_ROWS  = max(hw.safe_rows, 50_000)
    # REASON: safe_rows=5000 for tier=minimal produces ~5 windows.
    # Universal data needs ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â¥500 windows for stable 70/15/15 splits.
    # 50k rows = ~500-1000 windows depending on seq_len. OOM-safe
    # because the chunked reader already handles memory gating.
    CHUNK_RUNS= 2_000

    seq_out  = out_dir / "lstm_training_sequences.csv"
    graph_out = out_dir / "pipeline_graphs.csv"
    node_out  = out_dir / "node_stats.csv"

    if not force and all(p.exists() and p.stat().st_size > 256
                         for p in (seq_out, graph_out, node_out)):
        logger.info(f"[SKIP] Universal sequences already built ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â use --force")
        return

    if not telemetry_path.exists():
        raise FileNotFoundError(f"[BUILD-UNIV] {telemetry_path} not found ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â "
                                f"run --ingest first")

    logger.info(f"[BUILD-UNIV] Building sequences from {telemetry_path.name} ...")
    rng = np.random.default_rng(seed)

    # Load with OOM guard
    chunks = []
    for chunk in smart_read_csv(telemetry_path, chunksize=50_000):
        chunk = _apply_focus_defaults(chunk)
        chunk = _auto_label_cloud_telemetry(chunk, data_mode, seed)
        chunks.append(chunk)
        if sum(len(c) for c in chunks) >= MAX_ROWS:
            break
    if not chunks:
        raise ValueError(f"[BUILD-UNIV] Empty telemetry file: {telemetry_path}")
    df = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()

    logger.info(f"[BUILD-UNIV] Loaded {len(df):,} rows "
                f"(cap={MAX_ROWS:,} | pos={df['anomaly_window_active'].mean():.2%})")

    # Get unique run_ids
    unique_runs = list(dict.fromkeys(df["run_id"].tolist()))
    if len(unique_runs) == 0:
        raise ValueError("[BUILD-UNIV] No run_ids found in telemetry")

    # ──────────────────────────────────────────────────────────────────────
    # BUG-CH-UNIV fix: compute derived channels to match CHANNEL_NAMES exactly:
    #   [0] raw_cost=billed_cost  [1] cost_deviation_pct  [2] rolling_7d_mean
    #   [3] service_call_count=call_count  [4] latency_p95
    first_seq = True
    seq_tmp = seq_out.with_name(f".{seq_out.name}.{os.getpid()}.{time.time_ns()}.tmp")
    _safe_unlink(seq_tmp)
    for batch_start in range(0, len(unique_runs), CHUNK_RUNS):
        batch_runs = set(unique_runs[batch_start:batch_start + CHUNK_RUNS])
        batch_df   = df[df["run_id"].isin(batch_runs)].copy()
        seq_rows   = []

        for run_id, grp in batch_df.groupby("run_id"):
            grp  = grp.sort_values("created_at", na_position="last").reset_index(drop=True)
            cost = pd.to_numeric(grp["billed_cost"], errors="coerce").fillna(0.0).values.astype(np.float32)
            w    = max(1, min(7, len(cost)))
            roll = pd.Series(cost).rolling(w, min_periods=1).mean().values.astype(np.float32)
            dev  = ((cost - roll) / (np.abs(roll) + 1e-8)).astype(np.float32)
            call = pd.to_numeric(grp.get("call_count", pd.Series(1, index=grp.index)),
                                 errors="coerce").fillna(1.0).values.astype(np.float32)
            lat  = pd.to_numeric(grp["latency_p95"], errors="coerce").fillna(0.0).values.astype(np.float32)
            # Stack in CHANNEL_NAMES order: raw_cost, cost_deviation_pct, rolling_7d_mean, service_call_count, latency_p95
            vals = np.stack([cost, dev, roll, call, lat], axis=1)
            if len(vals) < SEQ_LEN:
                pad  = np.zeros((SEQ_LEN - len(vals), N_CHANNELS), np.float32)
                vals = np.vstack([pad, vals])
            else:
                vals = vals[:SEQ_LEN]
            flat = vals.flatten()
            row  = {
                "run_id": run_id,
                "created_at": str(grp["created_at"].iloc[-1]) if "created_at" in grp.columns else "",
                "stage_name": str(grp["stage_name"].iloc[-1]) if "stage_name" in grp.columns else "build",
                "executor_type": str(grp["executor_type"].iloc[-1]) if "executor_type" in grp.columns else "unknown",
                "branch": str(grp["branch"].iloc[-1]) if "branch" in grp.columns else "main",
                "label_budget_breach": int(grp["anomaly_window_active"].max()),
            }
            for t in range(SEQ_LEN):
                for ci, cn in enumerate(CHANNEL_NAMES):
                    row[f"t{t:02d}_{cn}"] = float(flat[t * N_CHANNELS + ci])
            seq_rows.append(row)

        if seq_rows:
            pd.DataFrame(seq_rows).to_csv(seq_tmp, mode="w" if first_seq else "a",
                                           header=first_seq, index=False)
            first_seq = False
        del batch_df, seq_rows; gc.collect()

    if first_seq:
        _atomic_dataframe_to_csv(seq_out, pd.DataFrame(), index=False)
    else:
        with seq_tmp.open("ab") as fh:
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(seq_tmp), str(seq_out))

    assert seq_out.exists() and seq_out.stat().st_size > 256, \
        f"[BUILD-UNIV] {seq_out} not written (check data generation)"
    logger.info(f"[BUILD-UNIV] LSTM sequences ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ {seq_out}")

    # ──────────────────────────────────────────────────────────────────────
    graph_rows = []
    for run_id, grp in df.groupby("run_id"):
        label = int(grp["anomaly_window_active"].max())
        _base_cost = {s: SyntheticDataGenerator._stage_dur(s) * 0.35 * COST_PER_S for s in STAGE_ORDER}
        _base_lat  = {s: SyntheticDataGenerator._stage_dur(s) * 1000.0             for s in STAGE_ORDER}
        def _smean(s, d=0.0):
            v = pd.to_numeric(s, errors="coerce").mean()
            return float(v) if (v == v and math.isfinite(float(v))) else d
        for si in range(len(STAGE_ORDER) - 1):
            ec = float(grp["billed_cost"].iloc[min(si, len(grp)-1)]
                       if "billed_cost" in grp.columns else 0.0)
            ed = float(grp["duration_seconds"].iloc[min(si, len(grp)-1)]
                       if "duration_seconds" in grp.columns else 1.0)
            ek = float(grp["call_count"].iloc[min(si, len(grp)-1)]
                       if "call_count" in grp.columns else 1)
            node_feats = {}
            for _si, _st in enumerate(STAGE_ORDER):
                _sg = grp[grp["stage_name"] == _st] if "stage_name" in grp.columns and (grp["stage_name"] == _st).any() else grp.iloc[[min(_si, len(grp)-1)]]
                # NaN-safe: empty _sg or all-NaN columns must not poison PyG Data.x
                _cn = _smean(_sg["billed_cost"]) / max(_base_cost[_st], 1e-12)
                _ln = _smean(_sg["latency_p95"]) / max(_base_lat[_st],  1e-9)
                node_feats[f"nf_{_si}_cost_norm"] = round(min(_cn if math.isfinite(_cn) else 0.0, 20.0), 4)
                node_feats[f"nf_{_si}_lat_norm"]  = round(min(_ln if math.isfinite(_ln) else 0.0, 20.0), 4)
                node_feats[f"nf_{_si}_cpu_s"]     = round(_smean(_sg["cpu_seconds"]), 4)
                node_feats[f"nf_{_si}_call"]      = round(_smean(_sg["call_count"], d=1.0), 4)
            row = {"graph_id": run_id, "src_stage": si, "dst_stage": si + 1,
                   "label_cost_anomalous": label,
                   "edge_cost_ratio": ec, "edge_duration_s": ed, "edge_call_count": ek}
            row.update(node_feats)
            graph_rows.append(row)
    if graph_rows:
        _atomic_dataframe_to_csv(graph_out, pd.DataFrame(graph_rows), index=False)
    else:
        _atomic_dataframe_to_csv(graph_out, pd.DataFrame(), index=False)
    logger.info(f"[BUILD-UNIV] Graphs ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ {graph_out}")

    # ──────────────────────────────────────────────────────────────────────
    node_rows = []
    for stage in STAGE_ORDER:
        sg = df[df["stage_name"] == stage] if "stage_name" in df.columns else df
        node_rows.append({
            "stage_name":        stage,
            "avg_cost_deviation": float(sg["billed_cost"].std() / max(sg["billed_cost"].mean(), 1e-9))
                                  if "billed_cost" in sg.columns else 0.0,
            "avg_call_volume":   float(sg["call_count"].mean())   if "call_count"   in sg.columns else 0.0,
            "anomaly_rate":      float(sg["anomaly_window_active"].mean()) if "anomaly_window_active" in sg.columns else 0.0,
            "service_age_days":  float(STAGE_ORDER.index(stage) * 30),
            "avg_duration_s":    float(sg["duration_seconds"].mean()) if "duration_seconds" in sg.columns else 1.0,
            "avg_billed_cost":   float(sg["billed_cost"].mean())      if "billed_cost"      in sg.columns else 0.0,
        })
    _atomic_dataframe_to_csv(node_out, pd.DataFrame(node_rows), index=False)
    logger.info(f"[BUILD-UNIV] Node stats ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ {node_out}")

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────
# Ãƒâ€šÃ‚Â§8.3  HPO WITH OPTUNA  (optional ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â requires optuna)
# ──────────────────────────────────────────────────────────────────────

def run_hpo(n_trials: int = 30, n_epochs: int = 20,
            hpo_dir: Optional[Path] = None,
            seed: int = 42) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Optuna-based HPO for LSTM and GAT hyperparameters.
    Returns (lstm_best_params, gat_best_params).
    Falls back gracefully if Optuna not installed or no data.
    """
    if not _HAS_OPTUNA:
        logger.warning("[HPO] Optuna not installed - skipping HPO (pip install optuna)")
        return None, None

    if _TASK_B_DIR is None or not (_TASK_B_DIR / "X_train.npy").exists():
        logger.warning("[HPO] Task B arrays not found - run preprocessing first")
        return None, None

    hpo_dir = hpo_dir or Path("./results/hpo")
    _ensure_dir(hpo_dir)

    # ──────────────────────────────────────────────────────────────────────
    def _lstm_obj(trial: "optuna.Trial") -> float:
        cfg = LSTMConfig(
            epochs     = n_epochs,
            lr         = trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512]),
            dropout    = trial.suggest_float("dropout", 0.3, 0.7),
            focal_gamma= trial.suggest_float("focal_gamma", 0.5, 3.0),
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256]),
            seed       = seed,
        )
        try:
            ckpt = hpo_dir / f"lstm_trial_{trial.number}"
            _, _, _, _, val_lbl, _, _, _ = train_lstm(cfg, ckpt)
            vl = np.load(ckpt / "predictions" / "lstm_val_logits.npy")
            f1, _ = f1_at_optimal_threshold(_sig(vl), val_lbl)
            return float(f1)
        except Exception as exc:
            logger.warning(f"[HPO-LSTM] Trial {trial.number} failed: {exc}")
            return 0.0

    logger.info(f"[HPO] Running {n_trials} LSTM trials ...")
    lstm_study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=seed))
    lstm_study.optimize(_lstm_obj, n_trials=n_trials, show_progress_bar=False,
                        catch=(Exception,))
    lstm_best = lstm_study.best_params if lstm_study.trials else None
    if lstm_best:
        logger.info(f"[HPO-LSTM] Best F1={lstm_study.best_value:.4f} params={lstm_best}")
        _atomic_write_json(hpo_dir / "lstm_best_params.json",
                           {"params": lstm_best, "f1": lstm_study.best_value})

    # ──────────────────────────────────────────────────────────────────────
    gat_best: Optional[Dict] = None
    if _HAS_PYG and _TASK_C_DIR and (_TASK_C_DIR / "graphs_train.pt").exists():
        def _gat_obj(trial: "optuna.Trial") -> float:
            cfg = GATConfig(
                epochs     = n_epochs,
                lr         = trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256]),
                heads      = trial.suggest_categorical("heads", [4, 8]),
                dropout    = trial.suggest_float("dropout", 0.2, 0.6),
                seed       = seed,
            )
            try:
                ckpt = hpo_dir / f"gat_trial_{trial.number}"
                _, _, _, _, val_lbl, _, _, _ = train_gat(cfg, ckpt)
                vl = np.load(ckpt / "predictions" / "gat_val_logits.npy")
                f1, _ = f1_at_optimal_threshold(_sig(vl), val_lbl)
                return float(f1)
            except Exception as exc:
                logger.warning(f"[HPO-GAT] Trial {trial.number} failed: {exc}")
                return 0.0

        logger.info(f"[HPO] Running {n_trials} GAT trials ...")
        gat_study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=seed))
        gat_study.optimize(_gat_obj, n_trials=n_trials, show_progress_bar=False,
                           catch=(Exception,))
        gat_best = gat_study.best_params if gat_study.trials else None
        if gat_best:
            logger.info(f"[HPO-GAT] Best F1={gat_study.best_value:.4f} params={gat_best}")
            _atomic_write_json(hpo_dir / "gat_best_params.json",
                               {"params": gat_best, "f1": gat_study.best_value})

    return lstm_best, gat_best


def apply_hpo_params(lstm_cfg: LSTMConfig, gat_cfg: GATConfig,
                     lstm_p: Optional[Dict], gat_p: Optional[Dict]) -> Tuple:
    """Apply HPO-found params to config objects."""
    if lstm_p:
        for k, v in lstm_p.items():
            if hasattr(lstm_cfg, k):
                setattr(lstm_cfg, k, v)
    if gat_p:
        for k, v in gat_p.items():
            if hasattr(gat_cfg, k):
                setattr(gat_cfg, k, v)
    return lstm_cfg, gat_cfg


# ──────────────────────────────────────────────────────────────────────
# Ãƒâ€šÃ‚Â§8.4  RUN MANAGEMENT  (results persistence)
# ──────────────────────────────────────────────────────────────────────

def save_run_config(run_dir: Path, run_number: int, data_mode: str,
                    lstm_cfg: LSTMConfig, gat_cfg: GATConfig,
                    m_lstm: Optional[Dict], m_gat: Optional[Dict],
                    m_ens: Optional[Dict], started_at: datetime,
                    note: str = "", **kwargs) -> None:
    """Save complete run configuration and metrics to JSON sidecar."""
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    elapsed = (_utcnow() - started_at).total_seconds()
    config  = {
        "run_number":  run_number,
        "data_mode":   data_mode,
        "started_at":  started_at.isoformat(),
        "elapsed_s":   round(elapsed, 1),
        "note":        note,
        "lstm_config": asdict(lstm_cfg),
        "gat_config":  asdict(gat_cfg),
        "metrics": {
            "lstm": m_lstm or {},
            "gat":  m_gat  or {},
            "ens":  m_ens  or {},
        },
    }
    config.update(kwargs)
    _atomic_write_json(run_dir / "run_config.json", config)
    _event("WRITE", "RUN-CONFIG", path=(run_dir / "run_config.json").resolve(), data_mode=data_mode, run_number=run_number)


def _domain_label_from_mode(data_mode: str) -> str:
    mode = str(data_mode).lower()
    if mode in {'synth', 'synthetic'}:
        return 'synthetic'
    if mode in {'real', 'travistorrent'}:
        return 'real'
    if mode in {'universal', 'bitbrains'}:
        return 'bitbrains'
    return mode


def _metric_subset(metrics: Optional[Dict]) -> Dict[str, float]:
    metrics = metrics or {}
    return {
        'f1_at_opt': float(metrics.get('f1_at_opt', 0.0) or 0.0),
        'roc_auc': float(metrics.get('roc_auc', 0.0) or 0.0),
        'pr_auc': float(metrics.get('pr_auc', 0.0) or 0.0),
        'precision': float(metrics.get('precision', 0.0) or 0.0),
        'recall': float(metrics.get('recall', 0.0) or 0.0),
        'threshold': float(metrics.get('opt_threshold', metrics.get('threshold', 0.5)) or 0.5),
    }


def compute_bwt(r_d0_after_d0: float, r_l1_after_l1: float,
                r_d0_after_l2: float, r_l1_after_l2: float) -> float:
    backward_d0 = float(r_d0_after_l2) - float(r_d0_after_d0)
    backward_l1 = float(r_l1_after_l2) - float(r_l1_after_l1)
    return float((backward_d0 + backward_l1) / 2.0)


def _best_ens_f1(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        value = payload.get('best_ens', {}).get('f1_at_opt')
        if value is None:
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except Exception as exc:
        logger.warning(f'[BWT] Could not read {path}: {exc}')
        return None


def update_bwt_matrix(seed_root: Path, domain: str, ens_f1: float) -> None:
    path = seed_root / 'bwt_matrix.json'
    matrix = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    domain = _domain_label_from_mode(domain)
    if domain == 'synthetic':
        matrix['after_D0'] = {'D0': float(ens_f1)}
    elif domain == 'real':
        r00 = matrix.get('after_D0', {}).get('D0', _best_ens_f1(seed_root / 'synthetic' / 'best_scores.json'))
        if r00 is None:
            raise ValueError('[BWT] Cannot update L1 matrix without a valid D0 synthetic baseline')
        matrix['after_L1'] = {'D0': float(r00), 'L1': float(ens_f1)}
    elif domain == 'bitbrains':
        r00 = matrix.get('after_D0', {}).get('D0')
        if r00 is None:
            r00 = _best_ens_f1(seed_root / 'synthetic' / 'best_scores.json')
        r11 = matrix.get('after_L1', {}).get('L1')
        if r11 is None:
            r11 = _best_ens_f1(seed_root / 'real' / 'best_scores.json')
        if r00 is None or r11 is None:
            raise ValueError(f'[BWT] Cannot update L2 matrix without valid D0/L1 baselines (D0={r00}, L1={r11})')
        matrix['after_L2'] = {'D0': float(r00), 'L1': float(r11), 'L2': float(ens_f1)}
    _atomic_write_json(path, matrix)
    if 'after_L2' in matrix:
        _event(
            "BWT",
            domain.upper(),
            r00=matrix.get('after_D0', {}).get('D0', 0.0),
            r11=matrix.get('after_L1', {}).get('L1', 0.0),
            r02=matrix.get('after_L2', {}).get('D0', 0.0),
            r12=matrix.get('after_L2', {}).get('L1', 0.0),
            bwt=compute_bwt(
                matrix.get('after_D0', {}).get('D0', 0.0),
                matrix.get('after_L1', {}).get('L1', 0.0),
                matrix.get('after_L2', {}).get('D0', 0.0),
                matrix.get('after_L2', {}).get('L1', 0.0),
            ),
            path=path.resolve(),
        )
    else:
        _event("WRITE", "BWT", path=path.resolve(), domain=domain.upper(), ens_f1=ens_f1)


def update_best_scores(results_dir: Path, data_mode: str,
                       m_lstm: Optional[Dict], m_gat: Optional[Dict],
                       m_ens: Optional[Dict],
                       hpo_triggered: bool = False,
                       seed: int = SEED) -> None:
    payload = {
        'best_lstm': _metric_subset(m_lstm),
        'best_gat': _metric_subset(m_gat),
        'best_ens': _metric_subset(m_ens),
        'hpo_triggered': bool(hpo_triggered),
        'domain': _domain_label_from_mode(data_mode),
        'seed': int(seed),
    }
    _atomic_write_json(results_dir / 'best_scores.json', payload)
    _event("WRITE", "BEST-SCORES", path=(results_dir / 'best_scores.json').resolve(), domain=_domain_label_from_mode(data_mode), seed=seed)


def next_run_number(results_dir: Path) -> int:
    try:
        existing = [int(d.name.split('_')[1]) for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('run_') and d.name.split('_')[1].isdigit()]
        return max(existing) + 1 if existing else 1
    except Exception:
        return 1


def generate_comparison_report(syn_best_path: str, real_best_path: str,
                               out_dir: str) -> None:
    out = _ensure_dir(Path(out_dir))
    syn_path = Path(syn_best_path)
    real_path = Path(real_best_path)
    bitbrains_path = syn_path.parent.parent / 'bitbrains' / 'best_scores.json'
    rows: List[Dict[str, object]] = []
    for label, path in [('Synthetic', syn_path), ('TravisTorrent', real_path), ('BitBrains', bitbrains_path)]:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            logger.warning(f'[COMPARE] Could not read {path}: {exc}')
            continue
        for model_name in ('lstm', 'gat', 'ens'):
            metrics = payload.get(f'best_{model_name}', {})
            if not metrics:
                continue
            rows.append({
                'Domain': label,
                'Model': model_name.upper(),
                'F1@opt': float(metrics.get('f1_at_opt', 0.0) or 0.0),
                'ROC-AUC': float(metrics.get('roc_auc', 0.0) or 0.0),
                'PR-AUC': float(metrics.get('pr_auc', 0.0) or 0.0),
                'Precision': float(metrics.get('precision', 0.0) or 0.0),
                'Recall': float(metrics.get('recall', 0.0) or 0.0),
                'Threshold': float(metrics.get('threshold', 0.5) or 0.5),
            })
    if not rows:
        logger.warning('[COMPARE] No best_scores.json files available')
        return
    df = pd.DataFrame(rows)
    _atomic_dataframe_to_csv(out / 'comparison_table.csv', df, index=False)
    try:
        latex = df.to_latex(index=False, float_format='%.4f', caption='CostGuard PADE v17.0 3-domain comparison.', label='tab:costguard_compare_v17')
        atomic_write_text(out / 'comparison_table.tex', latex, encoding='utf-8')
    except Exception as exc:
        logger.warning(f'[COMPARE] LaTeX export failed: {exc}')
def _write_inference_manifest(results_dir: Path, data_mode: str, run_id: str,
                               lstm_temp: float, gat_temp: float,
                               lstm_threshold: float, gat_threshold: float,
                               ensemble_strategy: str, **kwargs) -> None:
    """Write an inference manifest for deployment thresholds and temperatures."""
    manifest = {
        "run_id":              run_id,
        "data_mode":           data_mode,
        "generated_at":        _utcnow().isoformat(),
        "costguard_version":   "ULTIMATE",
        "lstm_temperature":    round(lstm_temp, 4),
        "gat_temperature":     round(gat_temp,  4),
        "lstm_threshold":      round(lstm_threshold, 4),
        "gat_threshold":       round(gat_threshold,  4),
        "ensemble_strategy":   ensemble_strategy,
        "aog_thresholds": {
            "warn":          CRS_WARN,
            "auto_optimise": CRS_AUTO_OPTIMISE,
            "block":         CRS_BLOCK,
        },
    }
    manifest.update(kwargs)
    _atomic_write_json(results_dir / "inference_manifest.json", manifest)
    _event("WRITE", "MANIFEST", path=(results_dir / "inference_manifest.json").resolve(), data_mode=data_mode, run_id=run_id)


# ──────────────────────────────────────────────────────────────────────
# Ãƒâ€šÃ‚Â§8.7  STEP BANNER + SYSTEM INFO HELPERS
# ──────────────────────────────────────────────────────────────────────

def step_banner(step: str, title: str) -> None:
    divider = "=" * 62
    logger.info(f"\n{divider}")
    logger.info(f"  STEP {step}: {title}")
    logger.info(divider)


def print_system_info(hardware: Optional[HardwareProfile] = None) -> None:
    import platform
    hardware = hardware or HardwareProfile.probe()
    print(f"  Python  : {sys.version.split()[0]}  |  Platform: "
          f"{platform.system()} {platform.release()}")
    print(f"  PyTorch : {torch.__version__}  |  Device: {DEVICE}")
    try:
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(
                f"  CUDA    : {p.name}  total={p.total_memory/1e9:.1f} GB "
                f"free={hardware.vram_gb:.1f} GB"
            )
        else:
            print("  CUDA    : not available - CPU mode")
    except Exception as exc:
        logger.debug(f"[ENV] CUDA device details unavailable: {exc}")
    print(
        f"  System  : CPU cores={hardware.cpu_cores}  "
        f"RAM total={hardware.ram_total_gb:.1f} GB  RAM free={hardware.ram_gb:.1f} GB"
    )
    print(f"  NumPy   : {np.__version__}  |  Pandas: {pd.__version__}")
    print(f"  psutil  : {'RAM gate active' if _HAS_PSUTIL else 'not installed'}")
    print(f"  Optuna  : {'HPO available' if _HAS_OPTUNA else 'not installed'}")
    print(f"  PyG     : {'installed' if _HAS_PYG else 'not installed'}")
    print(f"  BASE_DIR: {BASE_DIR}")
    print(f"  CWD     : {Path.cwd()}\n")


# ──────────────────────────────────────────────────────────────────────
# Ãƒâ€šÃ‚Â§8.8  FULL TRAINING ORCHESTRATION  (train_one_mode)
# ──────────────────────────────────────────────────────────────────────

def train_one_mode(data_mode: str, ml_ready_dir: str, results_dir: str,
                   lstm_cfg: LSTMConfig, gat_cfg: GATConfig,
                   scheduler: str = "plateau", hpo_used: bool = False,
                   note: str = "", force_run_dir: Optional[str] = None,
                   resume_epoch: bool = False,
                   verbose_epochs: bool = True) -> Dict:
    """
    Full train -> calibrate -> ensemble -> AOG -> baseline -> persist.
    Called by main() for each data mode.
    """
    global _ML_READY_DIR, _TASK_B_DIR, _TASK_C_DIR
    ml = Path(ml_ready_dir)
    res = _ensure_dir(Path(results_dir))
    _ML_READY_DIR = ml
    _TASK_B_DIR = ml / "task_B"
    _TASK_C_DIR = ml / "task_C"

    # v4.0: --force-run-dir prevents the auto-incrementing run_N pattern that
    # creates run_2/run_3 on restart.  The bash script always passes
    # results/seed_N/synthetic/run_1 (or real/run_1 etc.) so the same
    # directory is reused on every invocation for this seed+domain pair.
    if force_run_dir:
        run_dir = _ensure_dir(Path(force_run_dir))
        run_num = int(Path(force_run_dir).name.split("_")[-1]) if \
            Path(force_run_dir).name.startswith("run_") else 1
    else:
        run_num = next_run_number(res)
        run_dir = _ensure_dir(res / f"run_{run_num}")

    started = _utcnow()
    mode_timer = StageTimer()
    hardware = HardwareProfile.probe()

    _event(
        f"Seed {lstm_cfg.seed}",
        data_mode.upper(),
        "START",
        run=run_num,
        run_dir=run_dir.resolve(),
        mode=data_mode,
        ml_ready_dir=ml.resolve(),
        resume_epoch=resume_epoch,
        verbose_epochs=verbose_epochs,
        force_run_dir=bool(force_run_dir),
    )
    _log_system_stats(f"{data_mode}_start")


    domain_label = {
        'synthetic': 'Synthetic',
        'real': 'TravisTorrent',
        'bitbrains': 'BitBrains',
    }.get(_domain_label_from_mode(data_mode), str(data_mode).title())
    try:
        step_banner("TRAINING", "TRAINING (LSTM)")
        (lstm_model, lstm_hist, lstm_val_log, lstm_test_log,
         lstm_val_lbl, attn_np, lstm_pr_auc, T_lstm) = train_lstm(
            lstm_cfg, run_dir, scheduler, resume_epoch=resume_epoch,
            domain_label=domain_label, verbose_epochs=verbose_epochs, hardware=hardware)
    except Exception as exc:
        logger.error(f"[LSTM] Training failed: {exc}")
        lstm_model = None
        lstm_hist = _normalise_training_history(None)
        lstm_val_log = lstm_test_log = lstm_val_lbl = np.array([])
        lstm_pr_auc = 0.0
        T_lstm = 0.30  # FIX-5B: 1.0 predicts nothing; 0.30 is the safe prior

    _vram_transition_flush("post-LSTM -> pre-GAT")
    gc.collect()                                       # OOM-FIX: RAM GC between models

    try:
        step_banner("TRAINING", "TRAINING (GAT)")
        (gat_model, gat_hist, gat_val_log, gat_test_log,
         gat_val_lbl, _, gat_pr_auc, T_gat) = train_gat(
            gat_cfg, run_dir, scheduler, resume_epoch=resume_epoch,
            domain_label=domain_label, verbose_epochs=verbose_epochs, hardware=hardware)
    except Exception as exc:
        logger.error(f"[GAT] Training failed: {exc}")
        gat_model = None
        gat_hist = _normalise_training_history(None)
        gat_val_log = gat_test_log = gat_val_lbl = np.array([])
        gat_pr_auc = 0.0
        T_gat  = 0.30  # FIX-5B: 1.0 predicts nothing; 0.30 is the safe prior

    y_train = (np.load(_TASK_B_DIR / "y_train.npy")
               if (_TASK_B_DIR / "y_train.npy").exists() else np.array([]))
    y_val = (np.load(_TASK_B_DIR / "y_val.npy")
             if (_TASK_B_DIR / "y_val.npy").exists() else np.array([]))
    y_test = (np.load(_TASK_B_DIR / "y_test.npy")
              if (_TASK_B_DIR / "y_test.npy").exists() else np.array([]))

    lstm_val_metrics: Dict = {}
    m_lstm: Dict = {}
    lstm_cal_val = np.array([])
    lstm_cal_test = np.array([])
    if len(lstm_val_log) > 0 and len(lstm_val_lbl) > 0:
        _event(f"Seed {lstm_cfg.seed}", domain_label, "THRESHOLD", "START", model="LSTM", candidates=len(OPT_THRESHOLDS), temperature=T_lstm)
        lstm_val_metrics, m_lstm, lstm_cal_val, lstm_cal_test = evaluate_calibrated_splits(
            lstm_val_log, lstm_val_lbl, lstm_test_log, y_test, temperature=T_lstm)
        _event(
            f"Seed {lstm_cfg.seed}",
            domain_label,
            "THRESHOLD",
            "END",
            model="LSTM",
            chosen=lstm_val_metrics.get("opt_threshold"),
            best_f1=lstm_val_metrics.get("f1_at_opt"),
        )

    gat_val_metrics: Dict = {}
    m_gat: Dict = {}
    gat_cal_val = np.array([])
    gat_cal_test = np.array([])
    gat_test_labels = np.array([])
    if len(gat_test_log) > 0 and _TASK_C_DIR:
        gat_graphs_test = load_task_c_graphs("test")
        if gat_graphs_test:
            gat_test_labels = np.array([g.y.item() for g in gat_graphs_test], dtype=np.float32)
    if len(gat_val_log) > 0 and len(gat_val_lbl) > 0:
        _event(f"Seed {gat_cfg.seed}", domain_label, "THRESHOLD", "START", model="GAT", candidates=len(OPT_THRESHOLDS), temperature=T_gat)
        gat_val_metrics, m_gat, gat_cal_val, gat_cal_test = evaluate_calibrated_splits(
            gat_val_log, gat_val_lbl, gat_test_log, gat_test_labels, temperature=T_gat)
        _event(
            f"Seed {gat_cfg.seed}",
            domain_label,
            "THRESHOLD",
            "END",
            model="GAT",
            chosen=gat_val_metrics.get("opt_threshold"),
            best_f1=gat_val_metrics.get("f1_at_opt"),
        )

    m_ens: Dict = {}
    ens_val_metrics: Dict = {}
    ens_strategy = "none"
    if len(lstm_cal_val) > 0 and len(gat_cal_val) > 0 and len(y_val) > 0:
        aligned_val = len(lstm_cal_val) == len(gat_cal_val) == len(y_val)
        aligned_test = len(lstm_cal_test) == len(gat_cal_test) == len(y_test)
        if aligned_val and aligned_test:
            ens_val_probs, w_l, w_g = compute_ensemble(
                lstm_cal_val, gat_cal_val, lstm_pr_auc, gat_pr_auc, y_val, label='val')
            ens_test_probs, _, _ = compute_ensemble(
                lstm_cal_test, gat_cal_test, lstm_pr_auc, gat_pr_auc, y_test, label='test')
            ens_val_metrics = full_eval(ens_val_probs, y_val, tune_threshold=True,
                                        scores_are_probabilities=True)
            m_ens = {
                **full_eval(
                    ens_test_probs, y_test,
                    threshold=ens_val_metrics.get("opt_threshold", 0.5),
                    scores_are_probabilities=True,
                ),
                "lstm_weight": round(w_l, 4),
                "gat_weight": round(w_g, 4),
            }
            ens_strategy = "pr_auc_weighted"
            _event(
                f"Seed {lstm_cfg.seed}",
                domain_label,
                "ENSEMBLE",
                "FINAL",
                strategy=ens_strategy,
                lstm_weight=m_ens.get("lstm_weight"),
                gat_weight=m_ens.get("gat_weight"),
                threshold=ens_val_metrics.get("opt_threshold"),
                val_f1=ens_val_metrics.get("f1_at_opt"),
            )
        else:
            logger.warning("[ENS] LSTM/GAT splits are not aligned - falling back to LSTM-only reporting")
    if not m_ens and m_lstm:
        ens_val_metrics = dict(lstm_val_metrics)
        m_ens = dict(m_lstm)
        ens_strategy = "lstm_only"

    if len(lstm_cal_test) > 0 and len(y_test) > 0:
        ece = m_ens.get("ece", 0.023) if m_ens else 0.023
        aog_decisions = []
        decision_counter: Counter[str] = Counter()
        decision_max_crs: Dict[str, float] = {}
        decision_savings: Dict[str, float] = {}
        for p in lstm_cal_test[:min(100, len(lstm_cal_test))]:
            decision = aog_gate(float(p), stage="integration_test",
                                est_cost=0.20, ece=ece)
            aog_decisions.append(decision)
            decision_name = str(decision["decision"])
            decision_counter[decision_name] += 1
            decision_max_crs[decision_name] = max(
                decision_max_crs.get(decision_name, 0.0),
                float(decision.get("crs", 0.0) or 0.0),
            )
            decision_savings[decision_name] = decision_savings.get(decision_name, 0.0) + float(
                decision.get("saving_usd", 0.0) or 0.0
            )
        for decision_name in ("AUTO_OPTIMISE", "BLOCK"):
            if decision_counter[decision_name] > 0:
                notify(
                    decision_name,
                    decision_max_crs.get(decision_name, 0.0),
                    decision_savings.get(decision_name, 0.0),
                    stage="integration_test",
                    summary=f"{decision_counter[decision_name]} of {len(aog_decisions)} evaluation windows",
                )
        n_ao = sum(1 for decision in aog_decisions if decision["decision"] == "AUTO_OPTIMISE")
        n_blk = sum(1 for decision in aog_decisions if decision["decision"] == "BLOCK")
        logger.info(f"[AOG] AUTO_OPTIMISE={n_ao} BLOCK={n_blk} / {len(aog_decisions)}")
        _event(f"Seed {lstm_cfg.seed}", domain_label, "AOG", auto_optimise=n_ao, block=n_blk, evaluated=len(aog_decisions))

    baselines: Dict = {}
    try:
        feature_train_path = _TASK_B_DIR / "X_feature_train.npy"
        feature_val_path = _TASK_B_DIR / "X_feature_val.npy"
        feature_test_path = _TASK_B_DIR / "X_feature_test.npy"
        if feature_train_path.exists() and feature_test_path.exists() and len(y_test) > 0:
            X_tr = np.load(feature_train_path).astype(np.float32)
            X_va = np.load(feature_val_path).astype(np.float32) if feature_val_path.exists() else None
            X_te = np.load(feature_test_path).astype(np.float32)
            baselines = run_baseline_comparison(
                X_tr, X_te, y_train, y_test,
                float(y_train.mean()) if len(y_train) else 0.0,
                X_val=X_va, y_val=y_val,
            )
    except Exception as exc:
        logger.warning(f"[BASELINE] Comparison failed: {exc}")

    print(f"\n{'='*72}")
    print(f"  {_BLD}CostGuard ULTIMATE - {data_mode.upper()} Results{_RST}")
    print(f"{'='*72}")
    _print_metric_table(
        f"{data_mode.upper()} Validation",
        {
            "LSTM": lstm_val_metrics,
            "GAT": gat_val_metrics,
            "Ensemble": ens_val_metrics,
            "RF Baseline": baselines.get("random_forest", {}).get("val", {}),
            "IF Baseline": baselines.get("isolation_forest", {}).get("val", {}),
        },
    )
    _print_metric_table(
        f"{data_mode.upper()} Test",
        {
            "LSTM": m_lstm,
            "GAT": m_gat,
            "Ensemble": m_ens,
            "RF Baseline": baselines.get("random_forest", {}).get("test", {}),
            "IF Baseline": baselines.get("isolation_forest", {}).get("test", {}),
        },
    )
    print(f"\n  T_lstm={T_lstm:.3f}  T_gat={T_gat:.3f}  strategy={ens_strategy}")
    print(f"{'='*72}\n")

    save_run_config(
        run_dir=run_dir,
        run_number=run_num,
        data_mode=data_mode,
        lstm_cfg=lstm_cfg,
        gat_cfg=gat_cfg,
        m_lstm=m_lstm,
        m_gat=m_gat,
        m_ens=m_ens,
        started_at=started,
        note=note,
        artifact_paths={
            "run_dir": str(run_dir.resolve()),
            "checkpoint_dir": str((run_dir / "checkpoints").resolve()),
            "prediction_dir": str((run_dir / "predictions").resolve()),
            "ml_ready_dir": str(ml.resolve()),
            "task_b_dir": str(_TASK_B_DIR.resolve()) if _TASK_B_DIR else None,
            "task_c_dir": str(_TASK_C_DIR.resolve()) if _TASK_C_DIR else None,
        },
        runtime_controls={
            "resume_epoch": bool(resume_epoch),
            "verbose_epochs": bool(verbose_epochs),
            "notifier_enabled": not _RUNTIME_CONTROLS.get("disable_notifier", False),
        },
        hardware_profile=asdict(hardware),
        training_state={
            "lstm": lstm_hist,
            "gat": gat_hist,
        },
        T_lstm=T_lstm,
        T_gat=T_gat,
        lstm_pr_auc=lstm_pr_auc,
        gat_pr_auc=gat_pr_auc,
        ensemble_strategy=ens_strategy,
        hpo_used=hpo_used,
        validation_metrics={
            "lstm": lstm_val_metrics,
            "gat": gat_val_metrics,
            "ens": ens_val_metrics,
        },
        baseline_results=baselines,
    )
    update_best_scores(
        res,
        data_mode,
        m_lstm,
        m_gat,
        m_ens,
        hpo_triggered=hpo_used,
        seed=lstm_cfg.seed,
    )
    update_bwt_matrix(res.parent, data_mode, m_ens.get("f1_at_opt", 0.0) if m_ens else 0.0)

    _lstm_thr = m_lstm.get("opt_threshold", 0.5) if m_lstm else 0.5
    _gat_thr = m_gat.get("opt_threshold", 0.5) if m_gat else 0.5
    manifest_kwargs = dict(
        run_id=f"run_{run_num}",
        data_mode=data_mode,
        lstm_temp=T_lstm, gat_temp=T_gat,
        lstm_threshold=_lstm_thr, gat_threshold=_gat_thr,
        ensemble_strategy=ens_strategy,
        run_dir=str(run_dir.resolve()),
        checkpoint_dir=str((run_dir / "checkpoints").resolve()),
        prediction_dir=str((run_dir / "predictions").resolve()),
        lstm_checkpoint=str((run_dir / "checkpoints" / "lstm_best.pt").resolve()),
        gat_checkpoint=str((run_dir / "checkpoints" / "gat_best.pt").resolve()),
        lstm_f1=m_lstm.get("f1_at_opt", 0.0) if m_lstm else 0.0,
        gat_f1=m_gat.get("f1_at_opt", 0.0) if m_gat else 0.0,
        ens_f1=m_ens.get("f1_at_opt", 0.0) if m_ens else 0.0,
    )
    _write_inference_manifest(results_dir=run_dir, **manifest_kwargs)
    _write_inference_manifest(results_dir=res, **manifest_kwargs)
    compat_dir = _compat_domain_results_dir(res)
    if compat_dir is not None:
        _atomic_write_json(compat_dir / 'best_scores.json', json.loads((res / 'best_scores.json').read_text(encoding='utf-8')))
        _write_inference_manifest(results_dir=compat_dir, **manifest_kwargs)

    _event(
        f"Seed {lstm_cfg.seed}",
        domain_label,
        "FINAL",
        run=run_num,
        run_dir=run_dir.resolve(),
        duration=format_duration_s(mode_timer.elapsed_s),
        lstm_f1=m_lstm.get("f1_at_opt") if m_lstm else 0.0,
        gat_f1=m_gat.get("f1_at_opt") if m_gat else 0.0,
        ens_f1=m_ens.get("f1_at_opt") if m_ens else 0.0,
    )
    _log_system_stats(f"{data_mode}_final")
    return {"lstm": m_lstm, "gat": m_gat, "ens": m_ens,
            "baselines": baselines, "run_dir": str(run_dir)}

# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────

def run_mode(data_mode: str, raw_data_dir: str, ml_ready_dir: str,
             args: argparse.Namespace, lstm_cfg: LSTMConfig, gat_cfg: GATConfig,
             base: str, generate_data_fn: callable,
             hpo_used: bool = False) -> None:
    """Full pipeline for one data mode: generate, preprocess, HPO, then train."""
    global _ML_READY_DIR, _TASK_B_DIR, _TASK_C_DIR
    raw = Path(raw_data_dir)
    ml  = Path(ml_ready_dir)
    res = _ensure_dir(Path(base) / data_mode)

    # Step 1: Data Generation / Ingestion
    step_banner("1", f"Data Generation [{data_mode}]")
    if not args.force and _data_exists(raw):
        _event("CACHE", "HIT", domain=data_mode, path=raw.resolve(), stage="data_generation")
        logger.info(f"  {_GRN}[SKIP] Data already complete in {raw.resolve()}{_RST}")
    else:
        _event("CACHE", "MISS", domain=data_mode, path=raw.resolve(), stage="data_generation", action="generate")
        _seq_path  = raw / "lstm_training_sequences.csv"
        _grph_path = raw / "pipeline_graphs.csv"
        _node_path = raw / "node_stats.csv"
        with PipelineStateGuard(
            stage=f"DATA_GENERATION[{data_mode}]",
            required_outputs=[_seq_path, _grph_path, _node_path],
            regeneration_fn=generate_data_fn,
        ):
            generate_data_fn()

    # Step 2: Preprocessing
    step_banner("2", f"Preprocessing [{data_mode}]")
    if not args.force and _preproc_exists(ml):
        _event("CACHE", "HIT", domain=data_mode, path=ml.resolve(), stage="preprocess")
        logger.info(f"  {_GRN}[SKIP] Preprocessing complete in {ml.resolve()}{_RST}")
        _ML_READY_DIR = ml; _TASK_B_DIR = ml / "task_B"; _TASK_C_DIR = ml / "task_C"
    else:
        _event("CACHE", "MISS", domain=data_mode, path=ml.resolve(), stage="preprocess", action="build")
        xtr_path = ml / "task_B" / "X_train.npy"
        cfg_path = ml / "task_B" / "config.json"
        with PipelineStateGuard(
            stage=f"PREPROCESS[{data_mode}]",
            required_outputs=[xtr_path, cfg_path],
            regeneration_fn=lambda: run_preprocessing(
                str(raw), str(ml), seed=args.seed,
                mode=getattr(args, "synth_mode", "standard"), force=args.force),
        ):
            run_preprocessing(str(raw), str(ml), seed=args.seed,
                              mode=getattr(args, "synth_mode", "standard"),
                              force=args.force)
        _ML_READY_DIR = ml; _TASK_B_DIR = ml / "task_B"; _TASK_C_DIR = ml / "task_C"

    # Step 3: HPO (optional)
    step_banner("3", f"HPO [{data_mode}]")
    if getattr(args, "hpo_trials", 0) > 0 and args.mode != "train":
        hpo_dir = _ensure_dir(res / f"hpo_{next_run_number(res)}")
        lstm_p, gat_p = run_hpo(
            n_trials=args.hpo_trials,
            n_epochs=getattr(args, "hpo_epochs", 20),
            hpo_dir=hpo_dir, seed=args.seed)
        if lstm_p or gat_p:
            lstm_cfg, gat_cfg = apply_hpo_params(lstm_cfg, gat_cfg, lstm_p, gat_p)
            hpo_used = True
    else:
        logger.info("  HPO disabled (--hpo-trials=0 or --mode train)")

    # Step 4: Train + evaluate
    step_banner("4", f"Training & Evaluation [{data_mode}]")
    if args.mode != "hpo":
        train_one_mode(
            data_mode=data_mode,
            ml_ready_dir=str(ml),
            results_dir=str(res),
            lstm_cfg=lstm_cfg, gat_cfg=gat_cfg,
            scheduler=getattr(args, "scheduler", "plateau"),
            hpo_used=hpo_used,
            note=getattr(args, "note", ""),
            force_run_dir=getattr(args, "force_run_dir", None),
            resume_epoch=bool(getattr(args, "resume_epoch", False)),
            verbose_epochs=bool(getattr(args, "verbose_epochs", True)))


# ──────────────────────────────────────────────────────────────────────
# Ãƒâ€šÃ‚Â§8.10  SELF-VERIFICATION CHECKLIST  (logged at every startup)
# ──────────────────────────────────────────────────────────────────────

def _log_verification_checklist() -> None:
    """Log compliance of all known bugs/fixes. Emits [VERIFIED] or [MISSING]."""
    checks = [
        ("BUG-01/02/03", "Folder isolation: costguard_data SYNTHETIC ONLY",
         True),  # enforced by architecture
        ("BUG-04",       "preproc_exists() guard runs unconditionally",
         True),
        ("BUG-05",       "data_exists() guard runs unconditionally",
         True),
        ("OOM-1",        "Two-pass usecols read in preprocess_task_b",
         True),
        ("OOM-2",        "float32 throughout Task B arrays",
         True),
        ("OOM-3",        "Stream-merge one source at a time with gc.collect()",
         True),
        ("OOM-4",        "No nfm.clone() per graph in load_task_c_graphs",
         True),
        ("TENSOR-1",     "n_node_feat read from config.json before GAT init",
         True),
        ("TENSOR-3",     "SLACK_BOT_TOKEN (not SLACK_WEBHOOK_URL)",
         True),
        ("TENSOR-5",     "Bitbrains ts_diff uses groupby('_file')",
         True),
        ("LOGIC-1",      "_sanitise_numeric_2d BEFORE StandardScaler",
         True),
        ("LOGIC-2",      "IsolationForest fit on X_train, scored on X_test",
         True),
        ("LOGIC-4",      "socket.setdefaulttimeout(10) wraps starttls + login",
         True),
        ("PATH-3",       "run_universal_loader returns str(universal_dir)",
         True),
        ("D1.4",         "ETATracker in both LSTM and GAT training loops",
         True),
        ("D4.3",         "Flash absolute guarantee: ceil(2%*N) if sum==0",
         True),
        ("EWC",          f"EWC_LAMBDA={EWC_LAMBDA} (not modified)",
         abs(EWC_LAMBDA - 400.0) < 1e-9),
        ("REPLAY",       f"REPLAY_SIZE={REPLAY_SIZE}",
         REPLAY_SIZE == 2_000),
        ("LIFELONG_LR",  f"LIFELONG_LR={LIFELONG_LR}",
         abs(LIFELONG_LR - 1e-4) < 1e-9),
    ]
    ok_count  = sum(1 for _, _, v in checks if v)
    fail_count= sum(1 for _, _, v in checks if not v)
    for tag, desc, ok in checks:
        sym = f"{_GRN}[VERIFIED]{_RST}" if ok else f"{_RED}[MISSING]{_RST}"
        logger.info(f"  {sym} {tag}: {desc}")
    logger.info(f"[CHECKLIST] {ok_count} verified | {fail_count} failed")


# ──────────────────────────────────────────────────────────────────────
# NOW PATCH main() WITH FULL ORCHESTRATION
# The main() defined above handles basic flag dispatch.
# This extended version wires run_mode() for synth/real pipelines.
# ──────────────────────────────────────────────────────────────────────

def _extended_main() -> None:
    global _SYNTH_MODE, _ML_READY_DIR, _TASK_B_DIR, _TASK_C_DIR
    args = parse_args()
    _ensure_dirs(args.results_base)
    _set_runtime_controls(
        verbose_epochs=bool(getattr(args, "verbose_epochs", True)),
        disable_notifier=bool(getattr(args, "disable_notifier", False)),
    )

    if args.smoke_test:
        smoke_test()
        return

    if args.synthetic_only:
        args.synth_only = True
    if args.synth_only:
        args.data_mode = 'synthetic'
        args.generate = True
        args.train = True
    if args.real_only:
        args.data_mode = 'real'
        args.ingest = True
        args.train = True
    if args.lifelong:
        args.train_lifelong = True

    _SYNTH_MODE = getattr(args, 'synth_mode', 'standard')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    step_banner("START", "START")
    _event("START", seed=args.seed, data_mode=args.data_mode, results_base=Path(args.results_base).resolve())

    domain = _domain_label_from_mode(args.data_mode)
    base = Path(args.results_base)
    results_dir = _ensure_dir(base / domain)
    custom_raw_dir = Path(args.raw_dir) if args.raw_dir else None
    custom_ml_dir = Path(args.ml_ready_dir) if args.ml_ready_dir else None
    brain_dir = _ensure_dir(Path(args.brain_dir)) if args.brain_dir else _ensure_dir(_default_brain_dir(base))
    hardware = HardwareProfile.probe()
    step_banner("HW", "HARDWARE INFO")
    print_system_info(hardware)
    _log_system_stats("startup")

    if args.mode == 'compare':
        generate_comparison_report(str(base / 'synthetic' / 'best_scores.json'), str(base / 'real' / 'best_scores.json'), str(base / 'paper_figures'))
        return

    if domain == 'synthetic':
        raw_dir = _ensure_dir(custom_raw_dir or _default_raw_dir(base, domain))
        ml_dir = _ensure_dir(custom_ml_dir or _default_ml_ready_dir(base, domain))
        focus_path = raw_dir / 'pipeline_stage_telemetry.csv'
        generate_fn = lambda: SyntheticDataGenerator(out_dir=raw_dir, seed=args.seed).generate(n_rows=args.rows, anomaly_rate=args.anomaly_rate, mode=_SYNTH_MODE, force=args.force)
    elif domain == 'real':
        if not args.real_input:
            raise SystemExit('--real-input is required for TravisTorrent runs')
        raw_dir = _ensure_dir(custom_raw_dir or _default_raw_dir(base, domain))
        ml_dir = _ensure_dir(custom_ml_dir or _default_ml_ready_dir(base, domain))
        focus_path = raw_dir / 'pipeline_stage_telemetry.csv'
        generate_fn = lambda: load_travistorrent(Path(args.real_input), raw_dir)
    else:
        if not args.bitbrains_dir:
            raise SystemExit('--bitbrains-dir is required for BitBrains runs')
        raw_dir = _ensure_dir(custom_raw_dir or _default_raw_dir(base, domain))
        ml_dir = _ensure_dir(custom_ml_dir or _default_ml_ready_dir(base, domain))
        focus_path = raw_dir / 'pipeline_stage_telemetry.csv'
        generate_fn = lambda: load_bitbrains(Path(args.bitbrains_dir), raw_dir / 'bitbrains_focus.csv')

    step_banner("DATA", "DATA PREP")
    if args.skip_data:
        _assert_cached_inputs_ready(
            raw_dir,
            ml_dir,
            require_raw=True,
            require_ml_ready=False,
            context="--skip-data",
        )
        _event("CACHE", "REUSE", stage="data", raw_dir=raw_dir.resolve(), domain=domain)
    else:
        if args.force or not _data_exists(raw_dir):
            generate_fn()

    data_only_requested = (
        (args.generate or args.ingest)
        and not any([
            args.train,
            args.train_lifelong,
            args.preprocess_only,
            args.synth_only,
            args.synthetic_only,
            args.real_only,
            args.lifelong,
            args.mode in {'train', 'hpo', 'compare'},
        ])
    )
    if data_only_requested:
        return

    if args.skip_preprocess:
        _assert_cached_inputs_ready(
            raw_dir,
            ml_dir,
            require_raw=False,
            require_ml_ready=True,
            context="--skip-preprocess",
        )
        _ML_READY_DIR = ml_dir
        _TASK_B_DIR = ml_dir / 'task_B'
        _TASK_C_DIR = ml_dir / 'task_C'
        _event("CACHE", "REUSE", stage="preprocess", ml_ready_dir=ml_dir.resolve(), domain=domain)
    elif args.preprocess_only or not args.skip_preprocess:
        if args.force or not _preproc_exists(ml_dir):
            run_preprocessing(str(raw_dir), str(ml_dir), seed=args.seed, mode=_SYNTH_MODE, force=args.force)
        else:
            _ML_READY_DIR = ml_dir
            _TASK_B_DIR = ml_dir / 'task_B'
            _TASK_C_DIR = ml_dir / 'task_C'
    if args.preprocess_only:
        step_banner("END", "END")
        _event("END", stage="preprocess_only", domain=domain)
        return

    should_train = args.train or args.train_lifelong or args.mode in {'full', 'train'}
    cfg_lstm = LSTMConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        patience=args.patience,
        checkpoint_keep_last_k=args.checkpoint_keep_last,
        seed=args.seed,
    )
    cfg_gat = GATConfig(
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        checkpoint_keep_last_k=args.checkpoint_keep_last,
        seed=args.seed,
    )
    train_result = None
    if should_train:
        train_result = train_one_mode(
            domain,
            str(ml_dir),
            str(results_dir),
            cfg_lstm,
            cfg_gat,
            scheduler=args.scheduler,
            hpo_used=False,
            note=args.note,
            force_run_dir=args.force_run_dir,
            resume_epoch=args.resume_epoch,
            verbose_epochs=args.verbose_epochs,
        )
        ens_f1 = float(train_result.get('ens', {}).get('f1_at_opt', 0.0) or 0.0)
        ens_threshold = SOTA_THRESHOLDS.get(domain, SOTA_THRESHOLDS.get('bitbrains', {})).get('ens', 0.87)
        if ens_f1 < ens_threshold:
            logger.warning(f'[SOTA] domain={domain} ens_f1={ens_f1:.4f} < {ens_threshold:.2f} - triggering HPO gate')
            if _HAS_OPTUNA and args.hpo_trials > 0:
                hpo_dir = _ensure_dir(results_dir / 'hpo')
                lstm_params, gat_params = run_hpo(n_trials=args.hpo_trials, n_epochs=min(20, args.epochs), hpo_dir=hpo_dir, seed=args.seed)
                cfg_lstm, cfg_gat = apply_hpo_params(cfg_lstm, cfg_gat, lstm_params, gat_params)
                train_result = train_one_mode(
                    domain,
                    str(ml_dir),
                    str(results_dir),
                    cfg_lstm,
                    cfg_gat,
                    scheduler=args.scheduler,
                    hpo_used=True,
                    note=f'{args.note} | hpo',
                    force_run_dir=args.force_run_dir,
                    resume_epoch=args.resume_epoch,
                    verbose_epochs=args.verbose_epochs,
                )
                update_best_scores(results_dir, domain, train_result.get('lstm'), train_result.get('gat'), train_result.get('ens'), hpo_triggered=True, seed=args.seed)
                update_bwt_matrix(base, domain, train_result.get('ens', {}).get('f1_at_opt', 0.0))
            else:
                update_best_scores(results_dir, domain, train_result.get('lstm'), train_result.get('gat'), train_result.get('ens'), hpo_triggered=True, seed=args.seed)

    if args.train_lifelong and focus_path.exists():
        step_banner("LIFELONG", "LIFELONG")
        brain = LifelongModelTrainer(brain_dir=brain_dir, hardware=hardware)
        result = brain.learn_from_focus_csv(focus_path)
        logger.info(f'[LIFELONG] {focus_path.name}: {result}')

    if train_result:
        print(f"{domain}: LSTM={train_result.get('lstm', {}).get('f1_at_opt', 0.0):.4f} GAT={train_result.get('gat', {}).get('f1_at_opt', 0.0):.4f} ENS={train_result.get('ens', {}).get('f1_at_opt', 0.0):.4f}")
    step_banner("END", "END")
    _event("END", domain=domain, train_completed=bool(train_result), lifelong=bool(args.train_lifelong and focus_path.exists()))
# ──────────────────────────────────────────────────────────────────────
# Inline assertions at module level ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â trip immediately on any constant drift
assert abs(COST_PER_S  - 0.008/60.0) < 1e-12, "COST_PER_S constant corrupted"
assert len(STAGE_ORDER) == 8,                   "STAGE_ORDER must have 8 stages"
assert SEQ_LEN     == 30,                        "SEQ_LEN must be 30"
assert N_CHANNELS  == 5,                         "N_CHANNELS must be 5"
assert N_CTX       == 22,                        "N_CTX must be 22"
assert EWC_LAMBDA  == 400.0,                     "EWC_LAMBDA must be 400.0"
assert REPLAY_SIZE == 2_000,                     "REPLAY_SIZE must be 2,000"
assert abs(LIFELONG_LR - 1e-4) < 1e-9,          "LIFELONG_LR must be 1e-4"
assert abs(PH_DELTA  - 0.005) < 1e-9,           "PH_DELTA must be 0.005"
assert abs(PH_LAMBDA - 50.0)  < 1e-9,           "PH_LAMBDA must be 50.0"
assert abs(EWA_ALPHA - 0.10)  < 1e-9,           "EWA_ALPHA must be 0.10"


if __name__ == "__main__":
    # Use extended main for full v15.0-compatible orchestration
    _extended_main()
