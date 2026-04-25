#!/usr/bin/env bash
set -euo pipefail
# run_ieee_trials.sh - v8.0 (OOM-HARDENED LAPTOP-SAFE ORCHESTRATOR)
# Adds: --data-only, --train-only, --laptop-safe, --prep-synth/real/bitbrains,
#       inter-seed GC sleep, Phase A / Phase B split workflow.
# All training output is tee'd into training_terminal_logs.md automatically.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ─── Canonical constants ────────────────────────────────────────────────────
AUTHORITATIVE_SEED=42
IEEE_DEFAULT_SEEDS=(42 52 62 72 82 92 102 112 122 132)
DEFAULT_DOMAIN_ORDER=(synthetic real bitbrains)
DEFAULT_TT_CSV="$ROOT_DIR/final-2017-01-25.csv"
DEFAULT_BB_DIR="$ROOT_DIR/fastStorage/2013-8"

# ─── Overridable hyperparameters ────────────────────────────────────────────
EPOCHS="${COSTGUARD_EPOCHS:-150}"
LR="${COSTGUARD_LR:-5e-4}"
PATIENCE="${COSTGUARD_PATIENCE:-10}"
SCHEDULER="${COSTGUARD_SCHEDULER:-cosine}"
EWC_LAMBDA="${COSTGUARD_EWC_LAMBDA:-400}"
REPLAY_SIZE="${COSTGUARD_REPLAY_SIZE:-2000}"
FOCAL_ALPHA="${COSTGUARD_FOCAL_ALPHA:-0.25}"
FOCAL_GAMMA="${COSTGUARD_FOCAL_GAMMA:-2.0}"
AUTHORITATIVE_HPO_TRIALS_DEFAULT="${COSTGUARD_HPO_TRIALS:-30}"
IEEE_HPO_TRIALS_DEFAULT="${COSTGUARD_IEEE_HPO_TRIALS:-0}"

# ─── Runtime state ──────────────────────────────────────────────────────────
DRY_RUN=0
FORCE_RERUN=0
RESUME_EPOCH=0
RESUME_ARGS=()
RUN_MODE="authoritative"
SAW_AUTHORITATIVE=0
SAW_IEEE_10=0
CUSTOM_SEEDS_RAW=""
HPO_TRIALS_OVERRIDE=""
HPO_TRIALS=""
VERBOSE_EPOCHS=1
NOTIFIER_ENABLED=0
RESULTS_ROOT="${COSTGUARD_RESULTS_ROOT:-$ROOT_DIR/results}"
TRIALS_ROOT="$RESULTS_ROOT/trials"
AGGREGATE_DIR="$RESULTS_ROOT/aggregate"
FIGURE_DIR="$RESULTS_ROOT/paper_figures"
SHARED_CACHE_ROOT="$RESULTS_ROOT/shared_cache"
LIVE_LOG="$ROOT_DIR/training_terminal_logs.md"
SCRIPT="CostGuard_PADE_FULL.py"
TT_CSV="${TT_CSV:-$DEFAULT_TT_CSV}"
BB_DIR="${BB_DIR:-$DEFAULT_BB_DIR}"

# ─── New v8.0 mode flags ─────────────────────────────────────────────────────
# DATA_ONLY=1 → Phase A only (generate+preprocess all 3 domains, no training)
# TRAIN_ONLY=1 → Phase B only (skip data gen, assume cache exists, train only)
# LAPTOP_SAFE=1 → override epochs=30, patience=5, batch-friendly defaults
# PREP_DOMAIN="" → only prep one specific domain (synth|real|bitbrains)
DATA_ONLY=0
TRAIN_ONLY=0
LAPTOP_SAFE=0
PREP_DOMAIN=""
# Sleep seconds between seeds to let system GC (0 = disabled)
INTER_SEED_SLEEP="${COSTGUARD_INTER_SEED_SLEEP:-30}"

LOG_BLOCK_CLOSED=0
MASTER_LOG=""
EXPERIMENT_MANIFEST=""
EXPERIMENT_MANIFEST_COMPAT=""
RUN_INVOCATION_ID=""
RUN_ARCHIVE_DIR=""
ARCHIVED_LIVE_LOG=""
ARCHIVED_MASTER_LOG=""
RUN_STARTED_AT_UTC=""
declare -A SEED_LOG_PREPARED=()
declare -A SEED_LOG_ARCHIVES=()

# ─── Usage ──────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: bash run_ieee_trials.sh [SCRIPT] [TT_CSV] [BB_DIR] [options]

PHASE A – Data Preparation Only:
  bash run_ieee_trials.sh --data-only
  bash run_ieee_trials.sh --prep-synth
  bash run_ieee_trials.sh --prep-real
  bash run_ieee_trials.sh --prep-bitbrains

PHASE B – Training Only (requires Phase A cache):
  bash run_ieee_trials.sh --train-only --seeds "42"
  bash run_ieee_trials.sh --train-only --ieee-10

Full Pipeline (default):
  bash run_ieee_trials.sh
  bash run_ieee_trials.sh --ieee-10
  bash run_ieee_trials.sh --authoritative
  bash run_ieee_trials.sh --seeds "42,52,62"

Laptop-Safe Preset (reduces epochs/memory):
  bash run_ieee_trials.sh --laptop-safe
  bash run_ieee_trials.sh --laptop-safe --ieee-10

Positional arguments:
  SCRIPT   Path to CostGuard_PADE_FULL.py (default: $SCRIPT)
  TT_CSV   Path to TravisTorrent CSV (default: $DEFAULT_TT_CSV)
  BB_DIR   Path to BitBrains directory (default: $DEFAULT_BB_DIR)

Options:
  --authoritative  Canonical single-seed seed-42 workflow
  --ieee-10        IEEE default 10-trial experiment
  --seeds CSV      Custom seed list e.g. "42,52,62"
  --data-only      Phase A: prep all 3 domains, do NOT train
  --prep-synth     Phase A: prep synthetic domain only
  --prep-real      Phase A: prep TravisTorrent domain only
  --prep-bitbrains Phase A: prep BitBrains domain only
  --train-only     Phase B: train only (assumes cache exists from --data-only)
  --laptop-safe    Override epochs=30, patience=5, hpo-trials=0 for low-RAM
  --results-root P Override results root directory
  --hpo-trials N   Override HPO trials
  --patience N     Override early-stopping patience
  --quiet-epochs   Suppress epoch-by-epoch training lines
  --enable-notifier Re-enable Slack/Gmail notifier
  --resume-epoch   Pass --resume-epoch to training engine
  --force-rerun    Clear stage markers and rerun
  --dry-run        Print commands without executing
  -h, --help       This help message

Logs are always appended to: $LIVE_LOG

Environment overrides:
  COSTGUARD_EPOCHS           Training epochs (default: 150)
  COSTGUARD_LR               Learning rate (default: 5e-4)
  COSTGUARD_PATIENCE         Early stopping patience (default: 10)
  COSTGUARD_INTER_SEED_SLEEP Seconds to sleep between seeds (default: 30)
  COSTGUARD_RESULTS_ROOT     Results root directory
  PYTHON_BIN                 Override Python interpreter path
EOF
}

# ─── Python resolver ────────────────────────────────────────────────────────
resolve_python() {
  local candidate
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PYTHON_BIN"; return 0
  fi
  for candidate in \
    "$ROOT_DIR/venv/Scripts/python.exe" \
    "$ROOT_DIR/venv/bin/python" \
    "$ROOT_DIR/.venv/Scripts/python.exe" \
    "$ROOT_DIR/.venv/bin/python"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"; return 0
    fi
  done
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"; return 0
    fi
  done
  return 1
}

# ─── Timestamp helpers ───────────────────────────────────────────────────────
timestamp_utc()      { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
archive_timestamp_id() { date -u +"%Y%m%dT%H%M%SZ"; }

duration_s() {
  local start_epoch="$1" end_epoch="$2"
  printf '%ss' "$(( end_epoch - start_epoch ))"
}

# ─── Log lifecycle ───────────────────────────────────────────────────────────
archive_file_if_present() {
  local src="$1" dest_dir="$2"
  if [[ -f "$src" || -L "$src" ]]; then
    mkdir -p "$dest_dir"
    local archived="$dest_dir/$(basename "$src")"
    mv -f "$src" "$archived"
    printf '%s\n' "$archived"; return 0
  fi
  printf '%s\n' ""
}

initialise_log_lifecycle() {
  RUN_INVOCATION_ID="$(archive_timestamp_id)"
  RUN_ARCHIVE_DIR="$RESULTS_ROOT/log_archive/$RUN_INVOCATION_ID"
  mkdir -p "$RUN_ARCHIVE_DIR"
  ARCHIVED_LIVE_LOG="$(archive_file_if_present "$LIVE_LOG" "$RUN_ARCHIVE_DIR")"
  ARCHIVED_MASTER_LOG="$(archive_file_if_present "${MASTER_LOG:-}" "$RUN_ARCHIVE_DIR")"
  : >> "$LIVE_LOG"
  [[ -n "${MASTER_LOG:-}" ]] && : > "$MASTER_LOG"
}

prepare_seed_log() {
  local seed_dir="$1" seed_log="$2"
  if [[ -n "${SEED_LOG_PREPARED[$seed_log]:-}" ]]; then return 0; fi
  local archive_dir="$seed_dir/log_archive/$RUN_INVOCATION_ID"
  local archived
  archived="$(archive_file_if_present "$seed_log" "$archive_dir")"
  if [[ -n "$archived" ]]; then SEED_LOG_ARCHIVES["$seed_log"]="$archived"; fi
  mkdir -p "$seed_dir"
  : > "$seed_log"
  SEED_LOG_PREPARED["$seed_log"]=1
}

close_log_block() {
  if [[ "$LOG_BLOCK_CLOSED" -eq 0 ]]; then
    printf '\n~~~\n\n' >> "$LIVE_LOG"
    LOG_BLOCK_CLOSED=1
  fi
}
trap close_log_block EXIT INT TERM

write_log_header() {
  local started_at="$1" seeds_csv="$2" mode_label="$3"
  {
    printf '\n'
    printf '## %s\n' "$mode_label"
    printf -- '- Started (UTC): %s\n' "$started_at"
    printf -- '- Runner: %s\n' "$ROOT_DIR/run_ieee_trials.sh"
    printf -- '- Training Script: %s\n' "$SCRIPT"
    printf -- '- TravisTorrent: %s\n' "$TT_CSV"
    printf -- '- BitBrains: %s\n' "$BB_DIR"
    printf -- '- Seeds: %s\n' "$seeds_csv"
    printf -- '- Invocation ID: %s\n' "$RUN_INVOCATION_ID"
    printf -- '- Archive Dir: %s\n' "$RUN_ARCHIVE_DIR"
    printf -- '- Mode: DATA_ONLY=%s TRAIN_ONLY=%s LAPTOP_SAFE=%s\n' \
        "$DATA_ONLY" "$TRAIN_ONLY" "$LAPTOP_SAFE"
    printf '\n~~~text\n'
  } >> "$LIVE_LOG"
}

seed_list_to_csv() { local IFS=,; printf '%s' "$*"; }

parse_seed_csv() {
  local raw="$1"
  local -a parsed=()
  local -A seen=()
  local item
  local IFS=','
  read -r -a items <<< "$raw"
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -n "$item" ]] || continue
    if [[ ! "$item" =~ ^[0-9]+$ ]]; then
      printf 'Invalid seed value: %s\n' "$item" >&2; exit 1
    fi
    if [[ -z "${seen[$item]:-}" ]]; then
      seen["$item"]=1; parsed+=("$item")
    fi
  done
  if [[ "${#parsed[@]}" -eq 0 ]]; then
    printf 'Seed list is empty.\n' >&2; exit 1
  fi
  printf '%s\n' "${parsed[@]}"
}

log_line() {
  local message="$1"
  printf '%s\n' "$message" | tee -a "${MASTER_LOG:-/dev/null}"
}

log_seed_line() {
  local seed_log="$1" message="$2"
  printf '%s\n' "$message" | tee -a "$seed_log" "${MASTER_LOG:-/dev/null}"
}

manual_logs_dir() { printf '%s\n' "$RESULTS_ROOT/manual_logs"; }

slugify_label() {
  local raw="$1"
  raw="${raw,,}"; raw="${raw// /_}"
  raw="${raw//[^a-z0-9._-]/_}"
  while [[ "$raw" == *__* ]]; do raw="${raw//__/_}"; done
  raw="${raw##_}"; raw="${raw%%_}"
  printf '%s\n' "${raw:-stage}"
}

quote_cmd() {
  local out="" arg
  for arg in "$@"; do printf -v out '%s%q ' "$out" "$arg"; done
  printf '%s' "$out"
}

# ─── System snapshot helper ──────────────────────────────────────────────────
capture_system_snapshot() {
  local out_path="$1"
  "$PY" - "$out_path" <<'PY'
from __future__ import annotations
import json, os, shutil, subprocess, sys, time
from pathlib import Path
snapshot = {"cpu_cores": None, "cpu_percent": None, "ram_total_mb": None,
            "ram_used_mb": None, "ram_free_mb": None, "gpu_name": None,
            "gpu_util_percent": None, "vram_total_mb": None,
            "vram_used_mb": None, "vram_free_mb": None}
try:
    import psutil
    vm = psutil.virtual_memory()
    snapshot["cpu_cores"]   = psutil.cpu_count(logical=True) or 1
    snapshot["cpu_percent"] = float(psutil.cpu_percent(interval=0.0))
    snapshot["ram_total_mb"]= int(vm.total / (1024*1024))
    snapshot["ram_used_mb"] = int(vm.used  / (1024*1024))
    snapshot["ram_free_mb"] = int(vm.available / (1024*1024))
except Exception: pass
try:
    import torch
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info(0)
        snapshot["vram_total_mb"] = int(total_b / (1024*1024))
        snapshot["vram_free_mb"]  = int(free_b  / (1024*1024))
        snapshot["vram_used_mb"]  = max(0, snapshot["vram_total_mb"] - snapshot["vram_free_mb"])
        try: snapshot["gpu_name"] = torch.cuda.get_device_name(0)
        except: snapshot["gpu_name"] = "cuda"
except Exception: pass
nvidia_smi = shutil.which("nvidia-smi")
if nvidia_smi:
    try:
        out = subprocess.check_output(
            [nvidia_smi, "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=2)
        u, used, total = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        snapshot["gpu_util_percent"] = float(u)
        if snapshot["vram_used_mb"] is None: snapshot["vram_used_mb"] = int(float(used))
        if snapshot["vram_total_mb"] is None: snapshot["vram_total_mb"] = int(float(total))
        if snapshot["vram_total_mb"] is not None and snapshot["vram_used_mb"] is not None:
            snapshot["vram_free_mb"] = max(0, snapshot["vram_total_mb"] - snapshot["vram_used_mb"])
    except: pass
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
with tmp.open("w", encoding="utf-8", newline="") as handle:
    json.dump(snapshot, handle, indent=2)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(str(tmp), str(path))
PY
}

# ─── Markdown log renderer ───────────────────────────────────────────────────
render_manual_markdown_log() {
  local md_path="$1" stage_title="$2" status="$3" started_at="$4"
  local completed_at="$5" command_text="$6" raw_log_path="$7"
  local start_snapshot_path="$8" end_snapshot_path="$9"
  "$PY" - "$md_path" "$stage_title" "$status" "$started_at" "$completed_at" \
        "$command_text" "$raw_log_path" "$start_snapshot_path" "$end_snapshot_path" <<'PY'
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
def load_json(p):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except: return {}
def fmt_pct(v): return "n/a" if v is None else f"{float(v):.0f}%"
def fmt_mb(v): return "n/a" if v is None else f"{int(v)} MB"
def fmt_gb(v): return "n/a" if v is None else f"{int(v)/1024:.1f} GB"
md_path, stage_title, status, started_at, completed_at, command_text, \
    raw_log_path, start_snap, end_snap = sys.argv[1:]
ss = load_json(start_snap); es = load_json(end_snap)
raw_log = Path(raw_log_path).read_text(encoding="utf-8") if Path(raw_log_path).exists() else ""
lines = [
    f"# {stage_title}", "",
    "## START", f"- Start time (UTC): {started_at}", f"- Status: {status}", "",
    "## HARDWARE INFO",
    f"- GPU: {ss.get('gpu_name') or 'cpu'}",
    f"- VRAM free: {fmt_mb(ss.get('vram_free_mb'))}",
    f"- RAM free: {fmt_gb(ss.get('ram_free_mb'))}", "",
    "## DATA PREP", f"- Stage: {stage_title}", "",
    "## TRAINING (LSTM)", f"- Stage: {stage_title}", "",
    "## TRAINING (GAT)", f"- Stage: {stage_title}", "",
    "## LIFELONG", f"- Stage: {stage_title}", "",
    "## AGGREGATION", f"- Stage: {stage_title}", "",
    "## COMMAND", "```text", command_text, "```", "",
    "## LOGS", "```text", raw_log.rstrip(), "```", "",
    "## SYSTEM STATS (end)",
    f"- VRAM used: {fmt_mb(es.get('vram_used_mb'))}",
    f"- VRAM free: {fmt_mb(es.get('vram_free_mb'))}",
    f"- RAM used: {fmt_gb(es.get('ram_used_mb'))}",
    f"- RAM free: {fmt_gb(es.get('ram_free_mb'))}", "",
    "## END", f"- End time (UTC): {completed_at}", f"- Final status: {status}", "",
]
path = Path(md_path)
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
with tmp.open("w", encoding="utf-8", newline="") as handle:
    handle.write("\n".join(lines))
    handle.flush()
    os.fsync(handle.fileno())
os.replace(str(tmp), str(path))
PY
}

# ─── Logged command runner ────────────────────────────────────────────────────
run_logged_command() {
  local stage_slug="$1" stage_title="$2" status_out_var="$3" seed_log="${4:-}"
  shift 4
  local manual_dir ts md_path raw_log_path start_snap end_snap started_at completed_at status=0
  manual_dir="$(manual_logs_dir)"; mkdir -p "$manual_dir"
  ts="$(archive_timestamp_id)"
  md_path="$manual_dir/${stage_slug}_${ts}.md"
  raw_log_path="$manual_dir/.${stage_slug}_${ts}.log"
  start_snap="$manual_dir/.${stage_slug}_${ts}_start.json"
  end_snap="$manual_dir/.${stage_slug}_${ts}_end.json"
  started_at="$(timestamp_utc)"
  capture_system_snapshot "$start_snap"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[DRY-RUN] %s\n' "$(quote_cmd "$@")" > "$raw_log_path"
    status="dry_run"
  else
    if [[ -n "$seed_log" ]]; then
      set +e
      "$@" 2>&1 | tee -a "$seed_log" "${MASTER_LOG:-/dev/null}" "$raw_log_path" "$LIVE_LOG"
      status=${PIPESTATUS[0]}
      set -e
    else
      set +e
      "$@" 2>&1 | tee -a "${MASTER_LOG:-/dev/null}" "$raw_log_path" "$LIVE_LOG"
      status=${PIPESTATUS[0]}
      set -e
    fi
  fi
  completed_at="$(timestamp_utc)"
  capture_system_snapshot "$end_snap"
  render_manual_markdown_log "$md_path" "$stage_title" "$status" \
      "$started_at" "$completed_at" "$(quote_cmd "$@")" \
      "$raw_log_path" "$start_snap" "$end_snap"
  printf -v "$status_out_var" '%s' "$status"
}

# ─── Guards ───────────────────────────────────────────────────────────────────
require_file() {
  local path="$1" label="$2"
  if [[ -z "$path" || ! -f "$path" ]]; then
    log_line "[ERROR] Missing ${label}: ${path:-<empty>}"; exit 1
  fi
}
require_dir() {
  local path="$1" label="$2"
  if [[ -z "$path" || ! -d "$path" ]]; then
    log_line "[ERROR] Missing ${label}: ${path:-<empty>}"; exit 1
  fi
}
require_python() {
  local path="$1"
  if [[ -z "$path" ]]; then
    log_line "[ERROR] No Python interpreter found. Set PYTHON_BIN or create venv."; exit 1
  fi
  if ! "$path" --version >/dev/null 2>&1; then
    log_line "[ERROR] Python interpreter not executable: $path"; exit 1
  fi
}

# ─── Path helpers ─────────────────────────────────────────────────────────────
analytics_results_root() {
  if [[ "$RUN_MODE" == "authoritative" ]]; then printf '%s\n' "$RESULTS_ROOT"
  else printf '%s\n' "$TRIALS_ROOT"; fi
}

seed_results_dir() {
  local seed="$1"
  if [[ "$RUN_MODE" == "authoritative" ]]; then printf '%s\n' "$RESULTS_ROOT/seed_${seed}"
  else printf '%s\n' "$TRIALS_ROOT/seed_${seed}"; fi
}

seed_workspace_dir() { local seed_dir="$1"; printf '%s\n' "$seed_dir/_workspace"; }

domain_raw_dir() {
  local seed_dir="$1" domain="$2"
  if [[ "$RUN_MODE" == "authoritative" ]]; then printf '%s\n' ""; return 0; fi
  case "$domain" in
    synthetic) printf '%s\n' "$(seed_workspace_dir "$seed_dir")/synthetic_raw" ;;
    real)      printf '%s\n' "$SHARED_CACHE_ROOT/real_data" ;;
    bitbrains) printf '%s\n' "$SHARED_CACHE_ROOT/bitbrains_data" ;;
    *)         printf '%s\n' "" ;;
  esac
}

domain_ml_ready_dir() {
  local seed_dir="$1" domain="$2"
  if [[ "$RUN_MODE" == "authoritative" ]]; then printf '%s\n' ""; return 0; fi
  case "$domain" in
    synthetic) printf '%s\n' "$(seed_workspace_dir "$seed_dir")/ml_ready_synthetic" ;;
    real)      printf '%s\n' "$(seed_workspace_dir "$seed_dir")/ml_ready_real" ;;
    bitbrains) printf '%s\n' "$(seed_workspace_dir "$seed_dir")/ml_ready_bitbrains" ;;
    *)         printf '%s\n' "" ;;
  esac
}

seed_log_path() { local seed_dir="$1"; printf '%s\n' "$seed_dir/seed_run.log"; }

# ─── Stage runner ─────────────────────────────────────────────────────────────
# Respects .stage_*_complete stamp files for safe resume.
run_stage() {
  local seed="$1" seed_log="$2" stage_label="$3" stamp_file="$4"
  shift 4
  if [[ -f "$stamp_file" ]]; then
    log_seed_line "$seed_log" "[$(timestamp_utc)] [SKIP] $stage_label already complete (stamp: $stamp_file)"
    return 0
  fi
  log_seed_line "$seed_log" "[$(timestamp_utc)] [START] $stage_label"
  local status_var; status_var="stage_status_$$"
  run_logged_command "$(slugify_label "$stage_label")" "$stage_label" \
      "$status_var" "$seed_log" "$@"
  local status; status="${!status_var}"
  if [[ "$status" != "dry_run" && "$status" -ne 0 ]]; then
    log_seed_line "$seed_log" "[$(timestamp_utc)] [FAIL] $stage_label exit=$status"
    return "$status"
  fi
  if [[ "$DRY_RUN" -eq 0 ]]; then
    touch "$stamp_file"
  fi
  log_seed_line "$seed_log" "[$(timestamp_utc)] [OK] $stage_label"
  return 0
}

reset_seed_state() {
  local seed_dir="$1"
  rm -f "$seed_dir/.stage_d0_complete" \
        "$seed_dir/.stage_l1_complete" \
        "$seed_dir/.stage_l2_complete" \
        "$seed_dir/trial_complete.json"
}

reconcile_stage_stamp() {
  local seed_log="$1" stamp="$2" label="$3" artifact1="${4:-}" artifact2="${5:-}"
  if [[ -f "$stamp" ]]; then return 0; fi
  local has_artifacts=1
  for art in "$artifact1" "$artifact2"; do
    [[ -z "$art" || -f "$art" ]] || { has_artifacts=0; break; }
  done
  if [[ "$has_artifacts" -eq 1 && -n "$artifact1" && -f "$artifact1" ]]; then
    log_seed_line "$seed_log" "[RECONCILE] $label artifacts found → creating stamp $stamp"
    touch "$stamp"
  fi
}

reconcile_trial_complete_marker() {
  local seed_log="$1" seed_dir="$2" seed="$3"
  local complete_json="$seed_dir/trial_complete.json"
  [[ -f "$complete_json" ]] && return 0
  if [[ -f "$seed_dir/.stage_d0_complete" && \
        -f "$seed_dir/.stage_l1_complete" && \
        -f "$seed_dir/.stage_l2_complete" ]]; then
    log_seed_line "$seed_log" "[RECONCILE] All stages complete → creating trial_complete.json"
    write_trial_complete_marker "$seed" "$seed_dir" "$(timestamp_utc)"
  fi
}

write_trial_complete_marker() {
  local seed="$1" seed_dir="$2" completed_at="$3"
  "$PY" - "$seed_dir" "$seed" "$completed_at" <<'PY'
import json, os, sys, time
from pathlib import Path
def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(tmp), str(path))
seed_dir, seed, completed_at = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
atomic_write_json(seed_dir / "trial_complete.json", {"seed": int(seed), "completed_at": completed_at, "status": "complete"})
PY
}

write_trial_manifest() {
  local seed="$1" seed_dir="$2" seed_log="$3" status="$4"
  local started_at="$5" completed_at="$6" seeds_csv="$7"
  "$PY" - "$seed_dir" "$seed" "$status" "$started_at" "$completed_at" \
         "$SCRIPT" "$TT_CSV" "$BB_DIR" "$RUN_MODE" "$RESULTS_ROOT" \
         "$seeds_csv" "$(seed_list_to_csv "${DEFAULT_DOMAIN_ORDER[@]}")" \
         "$EPOCHS" "$LR" "$PATIENCE" "$SCHEDULER" "$HPO_TRIALS" \
         "$EWC_LAMBDA" "$REPLAY_SIZE" "$FOCAL_ALPHA" "$FOCAL_GAMMA" \
         "$RESUME_EPOCH" "$VERBOSE_EPOCHS" "$NOTIFIER_ENABLED" \
         "$RUN_INVOCATION_ID" "$RUN_STARTED_AT_UTC" \
         "${SEED_LOG_ARCHIVES[$seed_log]:-}" \
         "$ARCHIVED_LIVE_LOG" "${ARCHIVED_MASTER_LOG:-}" <<'PY'
import json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
def iso_now(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def abs_path(raw):
    if not raw: return None
    return str(Path(raw).expanduser().resolve())
def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(tmp), str(path))
args = sys.argv[1:]
(seed_dir_raw, seed, status, started_at, completed_at,
 training_script, tt_csv, bb_dir, run_mode, results_root,
 seeds_csv, domain_order_csv,
 epochs, lr, patience, scheduler, hpo_trials,
 ewc_lambda, replay_size, focal_alpha, focal_gamma,
 resume_epoch, verbose_epochs, notifier_enabled,
 run_invocation_id, invocation_started_at_utc,
 archived_seed_log, archived_live_log, archived_master_log) = args
seed_dir = Path(seed_dir_raw)
manifest_path = seed_dir / "trial_manifest.json"
existing = {}
if manifest_path.exists():
    try: existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except: pass
manifest = {
    **existing,
    "seed": int(seed), "status": status,
    "started_at": started_at or existing.get("started_at") or iso_now(),
    "completed_at": completed_at or existing.get("completed_at") or "",
    "updated_at": iso_now(),
    "run_mode": run_mode, "results_root": abs_path(results_root),
    "seeds_csv": seeds_csv, "domain_order": domain_order_csv.split(","),
    "hyperparameters": {"epochs": epochs, "lr": lr, "patience": patience,
                        "scheduler": scheduler, "hpo_trials": hpo_trials,
                        "ewc_lambda": ewc_lambda, "replay_size": replay_size,
                        "focal_alpha": focal_alpha, "focal_gamma": focal_gamma},
    "runtime_controls": {"resume_epoch": resume_epoch, "verbose_epochs": verbose_epochs,
                         "notifier_enabled": notifier_enabled},
    "invocation_id": run_invocation_id, "run_invocation_id": run_invocation_id,
    "invocation_started_at_utc": invocation_started_at_utc,
    "log_paths": {"archived_seed_log": archived_seed_log,
                  "archived_live_log": archived_live_log,
                  "archived_master_log": archived_master_log},
}
atomic_write_json(manifest_path, manifest)
PY
}

write_experiment_manifest() {
  local status="$1" started_at="$2" completed_at="$3" seeds_csv="$4"
  "$PY" - "$EXPERIMENT_MANIFEST" "$EXPERIMENT_MANIFEST_COMPAT" "$status" "$started_at" "$completed_at" \
         "$seeds_csv" "$RUN_MODE" "$RESULTS_ROOT" "$SCRIPT" \
         "$TT_CSV" "$BB_DIR" "$EPOCHS" "$LR" "$PATIENCE" "$HPO_TRIALS" \
         "$RUN_INVOCATION_ID" "$RUN_STARTED_AT_UTC" "$LIVE_LOG" "${MASTER_LOG:-}" \
         "$(manual_logs_dir)" "$RUN_ARCHIVE_DIR" "$ARCHIVED_LIVE_LOG" "${ARCHIVED_MASTER_LOG:-}" <<'PY'
import json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
def iso_now(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def abs_path(raw):
    if not raw: return ""
    return str(Path(raw).expanduser().resolve())
def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(tmp), str(path))
args = sys.argv[1:]
(manifest_path, compat_manifest_path, status, started_at, completed_at, seeds_csv,
 run_mode, results_root, training_script, tt_csv, bb_dir,
 epochs, lr, patience, hpo_trials, run_invocation_id, invocation_started_at_utc,
 live_log, master_log, manual_logs_dir, run_archive_dir, archived_live_log,
 archived_master_log) = args
path = Path(manifest_path)
existing = {}
if path.exists():
    try: existing = json.loads(path.read_text(encoding="utf-8"))
    except: pass
manifest = {**existing,
    "status": status,
    "started_at": started_at or existing.get("started_at") or iso_now(),
    "completed_at": completed_at or existing.get("completed_at") or "",
    "updated_at": iso_now(),
    "run_mode": run_mode, "seeds_csv": seeds_csv,
    "results_root": abs_path(results_root),
    "training_script": abs_path(training_script),
    "data_paths": {"travistorrent_csv": abs_path(tt_csv), "bitbrains_dir": abs_path(bb_dir)},
    "hyperparameters": {"epochs": epochs, "lr": lr, "patience": patience, "hpo_trials": hpo_trials},
    "invocation_id": run_invocation_id,
    "run_invocation_id": run_invocation_id,
    "invocation_started_at_utc": invocation_started_at_utc,
    "current_log_paths": {
        "live_log": abs_path(live_log),
        "master_log": abs_path(master_log),
        "manual_logs_dir": abs_path(manual_logs_dir),
    },
    "archived_log_paths": {
        "archive_dir": abs_path(run_archive_dir),
        "live_log": abs_path(archived_live_log),
        "master_log": abs_path(archived_master_log),
    },
}
atomic_write_json(path, manifest)
if compat_manifest_path:
    atomic_write_json(Path(compat_manifest_path), manifest)
PY
}

# ─── Inter-seed memory sleep ──────────────────────────────────────────────────
inter_seed_gc_sleep() {
  local seed="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then return 0; fi
  if [[ "${INTER_SEED_SLEEP:-0}" -gt 0 ]]; then
    log_line "[$(timestamp_utc)] [GC-SLEEP] Sleeping ${INTER_SEED_SLEEP}s between seeds for system GC (seed ${seed} done)"
    sleep "$INTER_SEED_SLEEP"
  fi
}

# ─── Phase A: data-only prep functions ───────────────────────────────────────
run_data_only_domain() {
  local domain="$1"   # synthetic | real | bitbrains
  local seed="${2:-42}"
  local seed_dir data_raw_dir data_ml_dir
  seed_dir="$(seed_results_dir "$seed")"
  data_raw_dir="$(domain_raw_dir "$seed_dir" "$domain")"
  data_ml_dir="$(domain_ml_ready_dir "$seed_dir" "$domain")"
  local -a cache_args=()
  if [[ -n "$data_raw_dir" ]]; then
    cache_args=(--raw-dir "$data_raw_dir" --ml-ready-dir "$data_ml_dir")
    mkdir -p "$data_raw_dir" "$data_ml_dir"
  fi
  local -a domain_args=()
  case "$domain" in
    synthetic)
      domain_args=(--generate --preprocess-only --data-mode synthetic) ;;
    real)
      require_file "$TT_CSV" "TravisTorrent CSV (--prep-real)"
      domain_args=(--ingest --preprocess-only --data-mode real --real-input "$TT_CSV") ;;
    bitbrains)
      require_dir "$BB_DIR" "BitBrains directory (--prep-bitbrains)"
      domain_args=(--ingest --preprocess-only --data-mode bitbrains --bitbrains-dir "$BB_DIR") ;;
    *) log_line "[ERROR] Unknown domain for data-prep: $domain"; exit 1 ;;
  esac
  local -a verbosity_args=()
  [[ "$VERBOSE_EPOCHS" -eq 1 ]] && verbosity_args=(--verbose-epochs) || verbosity_args=(--quiet-epochs)
  local status_var="dp_status_$$"
  log_line "[$(timestamp_utc)] [DATA-ONLY] Preparing domain: $domain"
  run_logged_command "data_prep_${domain}" "Data-Prep: $domain" "$status_var" "" \
    "$PY" "$SCRIPT" \
      "${domain_args[@]}" \
      --seed "$seed" \
      --results-base "$seed_dir" \
      --hpo-trials 0 \
      "${cache_args[@]}" \
      "${verbosity_args[@]}" \
      --disable-notifier
  local status="${!status_var}"
  if [[ "$status" != "dry_run" && "$status" -ne 0 ]]; then
    log_line "[ERROR] Data prep failed for domain=$domain exit=$status"
    return "$status"
  fi
  log_line "[$(timestamp_utc)] [DATA-ONLY] Domain $domain prepared OK"
}

# ─── Phase B: train-only function ────────────────────────────────────────────
# Identical to run_seed_trial but with --skip-data --skip-preprocess flags
run_seed_trial_train_only() {
  local seed="$1" seeds_csv="$2"
  local seed_dir seed_log seed_workspace note status=0
  local -a synthetic_cache_args=() real_cache_args=() bitbrains_cache_args=()
  local -a verbosity_args=() notifier_args=()
  local seed_started_epoch seed_completed_epoch trial_started_at completed_at

  seed_dir="$(seed_results_dir "$seed")"
  seed_log="$(seed_log_path "$seed_dir")"
  mkdir -p "$seed_dir"
  prepare_seed_log "$seed_dir" "$seed_log"
  seed_workspace="$(seed_workspace_dir "$seed_dir")"

  note="mode=train_only;seed=${seed};domain_order=D0->L1->L2"
  trial_started_at="$(timestamp_utc)"
  seed_started_epoch="$(date +%s)"
  log_seed_line "$seed_log" "[$trial_started_at] [SEED][TRAIN-ONLY][START] seed=${seed}"

  [[ "$VERBOSE_EPOCHS" -eq 1 ]] && verbosity_args=(--verbose-epochs) || verbosity_args=(--quiet-epochs)
  [[ "$NOTIFIER_ENABLED" -eq 0 ]] && notifier_args=(--disable-notifier)

  if [[ "$RUN_MODE" != "authoritative" ]]; then
    mkdir -p "$seed_workspace" "$SHARED_CACHE_ROOT"
    local syn_raw="$(domain_raw_dir "$seed_dir" "synthetic")"
    local syn_ml="$(domain_ml_ready_dir "$seed_dir" "synthetic")"
    local real_raw="$(domain_raw_dir "$seed_dir" "real")"
    local real_ml="$(domain_ml_ready_dir "$seed_dir" "real")"
    local bb_raw="$(domain_raw_dir "$seed_dir" "bitbrains")"
    local bb_ml="$(domain_ml_ready_dir "$seed_dir" "bitbrains")"

    # Train-only fallback: reuse authoritative caches when multi-seed cache paths
    # are missing (common after explicit --prep-* runs without --seeds).
    local auth_seed_dir="$RESULTS_ROOT/seed_${seed}"
    local auth_ws="$auth_seed_dir/_workspace"
    if [[ -n "$syn_raw" && ( ! -d "$syn_raw" || ! -f "$syn_raw/pipeline_stage_telemetry.csv" ) && -f "$auth_ws/synthetic_raw/pipeline_stage_telemetry.csv" ]]; then
      syn_raw="$auth_ws/synthetic_raw"
      syn_ml="$auth_ws/ml_ready_synthetic"
      log_seed_line "$seed_log" "[CACHE-FALLBACK] synthetic -> $syn_raw"
    fi
    if [[ -n "$real_raw" && ( ! -d "$real_raw" || ! -f "$real_raw/pipeline_stage_telemetry.csv" ) && -f "$auth_ws/real_data/pipeline_stage_telemetry.csv" ]]; then
      real_raw="$auth_ws/real_data"
      real_ml="$auth_ws/ml_ready_real"
      log_seed_line "$seed_log" "[CACHE-FALLBACK] real -> $real_raw"
    fi
    if [[ -n "$bb_raw" && ( ! -d "$bb_raw" || ! -f "$bb_raw/pipeline_stage_telemetry.csv" ) && -f "$auth_ws/bitbrains_data/pipeline_stage_telemetry.csv" ]]; then
      bb_raw="$auth_ws/bitbrains_data"
      bb_ml="$auth_ws/ml_ready_bitbrains"
      log_seed_line "$seed_log" "[CACHE-FALLBACK] bitbrains -> $bb_raw"
    fi

    [[ -n "$syn_raw" ]] && synthetic_cache_args=(--raw-dir "$syn_raw" --ml-ready-dir "$syn_ml")
    [[ -n "$real_raw" ]] && real_cache_args=(--raw-dir "$real_raw" --ml-ready-dir "$real_ml")
    [[ -n "$bb_raw" ]] && bitbrains_cache_args=(--raw-dir "$bb_raw" --ml-ready-dir "$bb_ml")
  fi

  [[ "$FORCE_RERUN" -eq 1 ]] && reset_seed_state "$seed_dir"

  reconcile_stage_stamp "$seed_log" "$seed_dir/.stage_d0_complete" "D0-Synthetic" \
      "$seed_dir/synthetic/best_scores.json" "$seed_dir/synthetic/run_1/run_config.json"
  reconcile_stage_stamp "$seed_log" "$seed_dir/.stage_l1_complete" "L1-TravisTorrent" \
      "$seed_dir/real/best_scores.json" "$seed_dir/real/run_1/run_config.json"
  reconcile_stage_stamp "$seed_log" "$seed_dir/.stage_l2_complete" "L2-BitBrains" \
      "$seed_dir/bitbrains/best_scores.json" "$seed_dir/bitbrains/run_1/run_config.json"

  if [[ -f "$seed_dir/trial_complete.json" ]]; then
    log_seed_line "$seed_log" "[SKIP] Seed ${seed} already complete"
    return 0
  fi

  # D0 Synthetic – training only (preprocess cache must exist)
  local -a _batch_args=()
  [[ -n "$BATCH_OVERRIDE" ]] && _batch_args=(--batch "$BATCH_OVERRIDE")
  run_stage "$seed" "$seed_log" "D0-Synthetic-TrainOnly" "$seed_dir/.stage_d0_complete" \
    "$PY" "$SCRIPT" \
      --train \
      --skip-data \
      --skip-preprocess \
      --data-mode synthetic \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --hpo-trials "$HPO_TRIALS" \
      --lr "$LR" --patience "$PATIENCE" --scheduler "$SCHEDULER" \
      --force-run-dir "$seed_dir/synthetic/run_1" \
      --results-base "$seed_dir" \
      --note "$note" \
      "${verbosity_args[@]}" "${notifier_args[@]}" \
      "${synthetic_cache_args[@]}" "${RESUME_ARGS[@]}" "${_batch_args[@]}" || status=$?
  if [[ "$status" -ne 0 ]]; then
    write_trial_manifest "$seed" "$seed_dir" "$seed_log" "failed" "$trial_started_at" "$(timestamp_utc)" "$seeds_csv"
    return "$status"
  fi

  # L1 TravisTorrent – training only
  run_stage "$seed" "$seed_log" "L1-TravisTorrent-TrainOnly" "$seed_dir/.stage_l1_complete" \
    "$PY" "$SCRIPT" \
      --train-lifelong \
      --skip-data \
      --skip-preprocess \
      --real-input "$TT_CSV" \
      --data-mode real \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --hpo-trials "$HPO_TRIALS" \
      --lr "$LR" --patience "$PATIENCE" --scheduler "$SCHEDULER" \
      --force-run-dir "$seed_dir/real/run_1" \
      --results-base "$seed_dir" \
      --note "$note" \
      "${verbosity_args[@]}" "${notifier_args[@]}" \
      "${real_cache_args[@]}" "${RESUME_ARGS[@]}" "${_batch_args[@]}" || status=$?
  if [[ "$status" -ne 0 ]]; then
    write_trial_manifest "$seed" "$seed_dir" "$seed_log" "failed" "$trial_started_at" "$(timestamp_utc)" "$seeds_csv"
    return "$status"
  fi

  # L2 BitBrains – training only
  run_stage "$seed" "$seed_log" "L2-BitBrains-TrainOnly" "$seed_dir/.stage_l2_complete" \
    "$PY" "$SCRIPT" \
      --train-lifelong \
      --skip-data \
      --skip-preprocess \
      --bitbrains-dir "$BB_DIR" \
      --data-mode bitbrains \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --hpo-trials "$HPO_TRIALS" \
      --lr "$LR" --patience "$PATIENCE" --scheduler "$SCHEDULER" \
      --force-run-dir "$seed_dir/bitbrains/run_1" \
      --results-base "$seed_dir" \
      --note "$note" \
      "${verbosity_args[@]}" "${notifier_args[@]}" \
      "${bitbrains_cache_args[@]}" "${RESUME_ARGS[@]}" "${_batch_args[@]}" || status=$?
  if [[ "$status" -ne 0 ]]; then
    write_trial_manifest "$seed" "$seed_dir" "$seed_log" "failed" "$trial_started_at" "$(timestamp_utc)" "$seeds_csv"
    return "$status"
  fi

  completed_at="$(timestamp_utc)"
  seed_completed_epoch="$(date +%s)"
  write_trial_complete_marker "$seed" "$seed_dir" "$completed_at"
  write_trial_manifest "$seed" "$seed_dir" "$seed_log" "complete" "$trial_started_at" "$completed_at" "$seeds_csv"
  log_seed_line "$seed_log" "[$(timestamp_utc)] [SEED][TRAIN-ONLY][END] seed=${seed} duration=$(duration_s "$seed_started_epoch" "$seed_completed_epoch")"
}

# ─── Full pipeline seed trial (original run_seed_trial) ──────────────────────
run_seed_trial() {
  local seed="$1" seeds_csv="$2"
  local seed_dir seed_log seed_workspace note status=0
  local synthetic_raw_dir synthetic_ml_dir real_raw_dir real_ml_dir bitbrains_raw_dir bitbrains_ml_dir
  local -a synthetic_cache_args=() real_cache_args=() bitbrains_cache_args=()
  local -a verbosity_args=() notifier_args=()
  local seed_started_epoch seed_completed_epoch trial_started_at completed_at

  seed_dir="$(seed_results_dir "$seed")"
  seed_log="$(seed_log_path "$seed_dir")"
  mkdir -p "$seed_dir"
  prepare_seed_log "$seed_dir" "$seed_log"
  seed_workspace="$(seed_workspace_dir "$seed_dir")"

  note="mode=${RUN_MODE};seed=${seed};domain_order=D0->L1->L2"
  trial_started_at="$(timestamp_utc)"
  seed_started_epoch="$(date +%s)"
  log_seed_line "$seed_log" "[$trial_started_at] [SEED][START] seed=${seed} run_invocation_id=${RUN_INVOCATION_ID}"

  [[ "$VERBOSE_EPOCHS" -eq 1 ]] && verbosity_args=(--verbose-epochs) || verbosity_args=(--quiet-epochs)
  [[ "$NOTIFIER_ENABLED" -eq 0 ]] && notifier_args=(--disable-notifier)

  synthetic_raw_dir="$(domain_raw_dir "$seed_dir" "synthetic")"
  synthetic_ml_dir="$(domain_ml_ready_dir "$seed_dir" "synthetic")"
  real_raw_dir="$(domain_raw_dir "$seed_dir" "real")"
  real_ml_dir="$(domain_ml_ready_dir "$seed_dir" "real")"
  bitbrains_raw_dir="$(domain_raw_dir "$seed_dir" "bitbrains")"
  bitbrains_ml_dir="$(domain_ml_ready_dir "$seed_dir" "bitbrains")"

  if [[ "$RUN_MODE" != "authoritative" ]]; then
    mkdir -p "$seed_workspace" "$SHARED_CACHE_ROOT"
    [[ -n "$synthetic_raw_dir" ]] && synthetic_cache_args=(--raw-dir "$synthetic_raw_dir" --ml-ready-dir "$synthetic_ml_dir")
    [[ -n "$real_raw_dir" ]]      && real_cache_args=(--raw-dir "$real_raw_dir" --ml-ready-dir "$real_ml_dir")
    [[ -n "$bitbrains_raw_dir" ]] && bitbrains_cache_args=(--raw-dir "$bitbrains_raw_dir" --ml-ready-dir "$bitbrains_ml_dir")
  fi

  [[ "$FORCE_RERUN" -eq 1 ]] && reset_seed_state "$seed_dir"

  reconcile_stage_stamp "$seed_log" "$seed_dir/.stage_d0_complete" "D0-Synthetic" \
      "$seed_dir/synthetic/best_scores.json" "$seed_dir/synthetic/run_1/run_config.json"
  reconcile_stage_stamp "$seed_log" "$seed_dir/.stage_l1_complete" "L1-TravisTorrent" \
      "$seed_dir/real/best_scores.json" "$seed_dir/real/run_1/run_config.json"
  reconcile_stage_stamp "$seed_log" "$seed_dir/.stage_l2_complete" "L2-BitBrains" \
      "$seed_dir/bitbrains/best_scores.json" "$seed_dir/bitbrains/run_1/run_config.json"
  reconcile_trial_complete_marker "$seed_log" "$seed_dir" "$seed"

  if [[ -f "$seed_dir/trial_complete.json" ]]; then
    log_seed_line "$seed_log" "[SKIP] Seed ${seed} already complete"
    write_trial_manifest "$seed" "$seed_dir" "$seed_log" "complete" "" "" "$seeds_csv"
    return 0
  fi

  write_trial_manifest "$seed" "$seed_dir" "$seed_log" "running" "$trial_started_at" "" "$seeds_csv"

  # D0 Synthetic
  local -a _batch_args=()
  [[ -n "$BATCH_OVERRIDE" ]] && _batch_args=(--batch "$BATCH_OVERRIDE")
  run_stage "$seed" "$seed_log" "D0-Synthetic" "$seed_dir/.stage_d0_complete" \
    "$PY" "$SCRIPT" \
      --synth-only \
      --seed "$seed" --epochs "$EPOCHS" --hpo-trials "$HPO_TRIALS" \
      --lr "$LR" --patience "$PATIENCE" --scheduler "$SCHEDULER" \
      --force-run-dir "$seed_dir/synthetic/run_1" \
      --results-base "$seed_dir" --note "$note" \
      "${verbosity_args[@]}" "${notifier_args[@]}" \
      "${synthetic_cache_args[@]}" "${RESUME_ARGS[@]}" "${_batch_args[@]}" || status=$?
  if [[ "$status" -ne 0 ]]; then
    write_trial_manifest "$seed" "$seed_dir" "$seed_log" "failed" "$trial_started_at" "$(timestamp_utc)" "$seeds_csv"
    return "$status"
  fi
  write_trial_manifest "$seed" "$seed_dir" "$seed_log" "running" "$trial_started_at" "" "$seeds_csv"

  # L1 TravisTorrent
  run_stage "$seed" "$seed_log" "L1-TravisTorrent" "$seed_dir/.stage_l1_complete" \
    "$PY" "$SCRIPT" \
      --train-lifelong \
      --real-input "$TT_CSV" \
      --data-mode real \
      --seed "$seed" --epochs "$EPOCHS" --hpo-trials "$HPO_TRIALS" \
      --lr "$LR" --patience "$PATIENCE" --scheduler "$SCHEDULER" \
      --force-run-dir "$seed_dir/real/run_1" \
      --results-base "$seed_dir" --note "$note" \
      "${verbosity_args[@]}" "${notifier_args[@]}" \
      "${real_cache_args[@]}" "${RESUME_ARGS[@]}" "${_batch_args[@]}" || status=$?
  if [[ "$status" -ne 0 ]]; then
    write_trial_manifest "$seed" "$seed_dir" "$seed_log" "failed" "$trial_started_at" "$(timestamp_utc)" "$seeds_csv"
    return "$status"
  fi
  write_trial_manifest "$seed" "$seed_dir" "$seed_log" "running" "$trial_started_at" "" "$seeds_csv"

  # L2 BitBrains
  run_stage "$seed" "$seed_log" "L2-BitBrains" "$seed_dir/.stage_l2_complete" \
    "$PY" "$SCRIPT" \
      --train-lifelong \
      --bitbrains-dir "$BB_DIR" \
      --data-mode bitbrains \
      --seed "$seed" --epochs "$EPOCHS" --hpo-trials "$HPO_TRIALS" \
      --lr "$LR" --patience "$PATIENCE" --scheduler "$SCHEDULER" \
      --force-run-dir "$seed_dir/bitbrains/run_1" \
      --results-base "$seed_dir" --note "$note" \
      "${verbosity_args[@]}" "${notifier_args[@]}" \
      "${bitbrains_cache_args[@]}" "${RESUME_ARGS[@]}" "${_batch_args[@]}" || status=$?
  if [[ "$status" -ne 0 ]]; then
    write_trial_manifest "$seed" "$seed_dir" "$seed_log" "failed" "$trial_started_at" "$(timestamp_utc)" "$seeds_csv"
    return "$status"
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log_seed_line "$seed_log" "[DRY-RUN] Seed ${seed} trial plan complete"
    seed_completed_epoch="$(date +%s)"
    log_seed_line "$seed_log" "[SEED][END] seed=${seed} duration=$(duration_s "$seed_started_epoch" "$seed_completed_epoch") dry_run=1"
    return 0
  fi

  completed_at="$(timestamp_utc)"
  seed_completed_epoch="$(date +%s)"
  write_trial_complete_marker "$seed" "$seed_dir" "$completed_at"
  write_trial_manifest "$seed" "$seed_dir" "$seed_log" "complete" "$trial_started_at" "$completed_at" "$seeds_csv"
  log_seed_line "$seed_log" "[$(timestamp_utc)] [SEED][END] seed=${seed} duration=$(duration_s "$seed_started_epoch" "$seed_completed_epoch")"
}

# ─── Analytics ────────────────────────────────────────────────────────────────
run_analytics() {
  local analytics_root="$1"
  local status=0
  local analytics_started_epoch manual_dir ts md_path raw_log_path start_snap end_snap started_at completed_at command_text
  analytics_started_epoch="$(date +%s)"
  manual_dir="$(manual_logs_dir)"; mkdir -p "$manual_dir"
  ts="$(archive_timestamp_id)"
  md_path="$manual_dir/aggregate_${ts}.md"
  raw_log_path="$manual_dir/.aggregate_${ts}.log"
  start_snap="$manual_dir/.aggregate_${ts}_start.json"
  end_snap="$manual_dir/.aggregate_${ts}_end.json"
  started_at="$(timestamp_utc)"
  : > "$raw_log_path"
  capture_system_snapshot "$start_snap"
  command_text="$PY $ROOT_DIR/aggregate_results.py --results-root $analytics_root --aggregate-dir $AGGREGATE_DIR --csv --latex"$'\n'"$PY $ROOT_DIR/generate_paper_figures.py --results-dir $analytics_root --aggregate-json $AGGREGATE_DIR/ieee_aggregate_summary.json --out-dir $FIGURE_DIR --dpi 1000 --formats pdf eps"
  log_line "[$(timestamp_utc)] [AGGREGATE][START] analytics_root=${analytics_root}"

  # Only run analytics if the scripts exist
  if [[ ! -f "$ROOT_DIR/aggregate_results.py" ]]; then
    log_line "[WARN] aggregate_results.py not found - skipping analytics"
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    {
      printf '[DRY-RUN] %s %s --results-root %s --aggregate-dir %s --csv --latex\n' \
        "$PY" "$ROOT_DIR/aggregate_results.py" "$analytics_root" "$AGGREGATE_DIR"
      if [[ -f "$ROOT_DIR/generate_paper_figures.py" ]]; then
        printf '[DRY-RUN] %s %s --results-dir %s --aggregate-json %s --out-dir %s --dpi 1000 --formats pdf eps\n' \
          "$PY" "$ROOT_DIR/generate_paper_figures.py" "$analytics_root" \
          "$AGGREGATE_DIR/ieee_aggregate_summary.json" "$FIGURE_DIR"
      fi
    } | tee -a "${MASTER_LOG:-/dev/null}" "$raw_log_path" "$LIVE_LOG"
    completed_at="$(timestamp_utc)"
    capture_system_snapshot "$end_snap"
    render_manual_markdown_log "$md_path" "Aggregation" "dry_run" "$started_at" "$completed_at" "$command_text" "$raw_log_path" "$start_snap" "$end_snap"
    log_line "[$(timestamp_utc)] [AGGREGATE][END] duration=$(duration_s "$analytics_started_epoch" "$(date +%s)") dry_run=1"
    return 0
  fi

  set +e
  "$PY" "$ROOT_DIR/aggregate_results.py" \
    --results-root "$analytics_root" \
    --aggregate-dir "$AGGREGATE_DIR" \
    --csv --latex \
    2>&1 | tee -a "${MASTER_LOG:-/dev/null}" "$raw_log_path" "$LIVE_LOG"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "$status" -ne 0 ]]; then
    log_line "[ERROR] aggregate_results.py failed with exit $status"
    completed_at="$(timestamp_utc)"
    capture_system_snapshot "$end_snap"
    render_manual_markdown_log "$md_path" "Aggregation" "failed" "$started_at" "$completed_at" "$command_text" "$raw_log_path" "$start_snap" "$end_snap"
    return "$status"
  fi

  if [[ -f "$ROOT_DIR/generate_paper_figures.py" ]]; then
    set +e
    "$PY" "$ROOT_DIR/generate_paper_figures.py" \
      --results-dir "$analytics_root" \
      --aggregate-json "$AGGREGATE_DIR/ieee_aggregate_summary.json" \
      --out-dir "$FIGURE_DIR" \
      --dpi 1000 --formats pdf eps \
      2>&1 | tee -a "${MASTER_LOG:-/dev/null}" "$raw_log_path" "$LIVE_LOG"
    local fig_status=${PIPESTATUS[0]}
    set -e
    [[ "$fig_status" -ne 0 ]] && log_line "[WARN] generate_paper_figures.py failed with exit $fig_status"
  fi

  completed_at="$(timestamp_utc)"
  capture_system_snapshot "$end_snap"
  render_manual_markdown_log "$md_path" "Aggregation" "ok" "$started_at" "$completed_at" "$command_text" "$raw_log_path" "$start_snap" "$end_snap"
  log_line "[$(timestamp_utc)] [AGGREGATE][END] duration=$(duration_s "$analytics_started_epoch" "$(date +%s)")"
}

# ─── Argument parsing ─────────────────────────────────────────────────────────
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --authoritative)   RUN_MODE="authoritative"; SAW_AUTHORITATIVE=1; shift ;;
    --ieee-10)         RUN_MODE="ieee_10"; SAW_IEEE_10=1; shift ;;
    --seeds)
      CUSTOM_SEEDS_RAW="${2:-}"
      [[ -z "$CUSTOM_SEEDS_RAW" ]] && { printf 'Missing value for --seeds\n' >&2; exit 1; }
      shift 2 ;;
    # ── v8.0 new modes ──
    --data-only)       DATA_ONLY=1; shift ;;
    --train-only)      TRAIN_ONLY=1; shift ;;
    --laptop-safe)     LAPTOP_SAFE=1; shift ;;
    --prep-synth)      PREP_DOMAIN="synthetic"; DATA_ONLY=1; shift ;;
    --prep-real)       PREP_DOMAIN="real"; DATA_ONLY=1; shift ;;
    --prep-bitbrains)  PREP_DOMAIN="bitbrains"; DATA_ONLY=1; shift ;;
    --results-root)
      RESULTS_ROOT="${2:-}"
      [[ -z "$RESULTS_ROOT" ]] && { printf 'Missing value for --results-root\n' >&2; exit 1; }
      TRIALS_ROOT="$RESULTS_ROOT/trials"
      AGGREGATE_DIR="$RESULTS_ROOT/aggregate"
      FIGURE_DIR="$RESULTS_ROOT/paper_figures"
      SHARED_CACHE_ROOT="$RESULTS_ROOT/shared_cache"
      shift 2 ;;
    --hpo-trials)
      HPO_TRIALS_OVERRIDE="${2:-}"
      [[ -z "$HPO_TRIALS_OVERRIDE" || ! "$HPO_TRIALS_OVERRIDE" =~ ^[0-9]+$ ]] && \
        { printf 'Invalid --hpo-trials value\n' >&2; exit 1; }
      shift 2 ;;
    --patience)
      PATIENCE="${2:-}"
      [[ -z "$PATIENCE" || ! "$PATIENCE" =~ ^[0-9]+$ || "$PATIENCE" -lt 1 ]] && \
        { printf 'Invalid --patience value\n' >&2; exit 1; }
      shift 2 ;;
    --quiet-epochs)    VERBOSE_EPOCHS=0; shift ;;
    --enable-notifier) NOTIFIER_ENABLED=1; shift ;;
    --resume-epoch)    RESUME_EPOCH=1; shift ;;
    --force-rerun)     FORCE_RERUN=1; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do POSITIONAL+=("$1"); shift; done ;;
    -*)
      printf 'Unknown option: %s\n\n' "$1" >&2; usage; exit 1 ;;
    *)
      POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ "$SAW_AUTHORITATIVE" -eq 1 && "$SAW_IEEE_10" -eq 1 ]]; then
  printf 'Invalid options: --authoritative and --ieee-10 cannot be used together\n' >&2
  exit 1
fi

[[ "${#POSITIONAL[@]}" -gt 3 ]] && { printf 'Too many positional arguments\n' >&2; usage; exit 1; }
[[ "${#POSITIONAL[@]}" -ge 1 ]] && SCRIPT="${POSITIONAL[0]}"
[[ "${#POSITIONAL[@]}" -ge 2 ]] && TT_CSV="${POSITIONAL[1]}"
[[ "${#POSITIONAL[@]}" -ge 3 ]] && BB_DIR="${POSITIONAL[2]}"

# ─── Laptop-safe preset overrides ────────────────────────────────────────────
BATCH_OVERRIDE=""
if [[ "$LAPTOP_SAFE" -eq 1 ]]; then
  EPOCHS="${COSTGUARD_EPOCHS:-30}"    # reduced from 150
  PATIENCE="${COSTGUARD_PATIENCE:-5}" # reduced from 10
  HPO_TRIALS_OVERRIDE="${HPO_TRIALS_OVERRIDE:-0}"
  VERBOSE_EPOCHS=0
  BATCH_OVERRIDE="32"
  printf '[LAPTOP-SAFE] epochs=%s patience=%s hpo=0 quiet_epochs=1 batch=%s\n' "$EPOCHS" "$PATIENCE" "$BATCH_OVERRIDE"
fi

# ─── Seed resolution ─────────────────────────────────────────────────────────
if [[ -n "$CUSTOM_SEEDS_RAW" ]]; then
  RUN_MODE="multi_seed"
  mapfile -t SEEDS < <(parse_seed_csv "$CUSTOM_SEEDS_RAW")
elif [[ "$RUN_MODE" == "ieee_10" ]]; then
  SEEDS=("${IEEE_DEFAULT_SEEDS[@]}")
else
  RUN_MODE="authoritative"
  SEEDS=("$AUTHORITATIVE_SEED")
fi

# HPO trials
if [[ -n "$HPO_TRIALS_OVERRIDE" ]]; then
  HPO_TRIALS="$HPO_TRIALS_OVERRIDE"
elif [[ "$RUN_MODE" == "authoritative" ]]; then
  HPO_TRIALS="$AUTHORITATIVE_HPO_TRIALS_DEFAULT"
else
  HPO_TRIALS="$IEEE_HPO_TRIALS_DEFAULT"
fi

PY="$(resolve_python || true)"
[[ "$RESUME_EPOCH" -eq 1 ]] && RESUME_ARGS=(--resume-epoch)

mkdir -p "$RESULTS_ROOT" "$AGGREGATE_DIR" "$FIGURE_DIR" "$SHARED_CACHE_ROOT"

if [[ "$RUN_MODE" == "authoritative" ]]; then
  MASTER_LOG="$RESULTS_ROOT/ieee_audit.log"
  EXPERIMENT_MANIFEST="$RESULTS_ROOT/authoritative_experiment_manifest.json"
  EXPERIMENT_MANIFEST_COMPAT="$AGGREGATE_DIR/authoritative_experiment_manifest.json"
  MODE_LABEL="Authoritative Trial - Seed ${AUTHORITATIVE_SEED}"
else
  MASTER_LOG="$RESULTS_ROOT/experiment_master.log"
  EXPERIMENT_MANIFEST="$RESULTS_ROOT/experiment_manifest.json"
  EXPERIMENT_MANIFEST_COMPAT="$AGGREGATE_DIR/ieee_experiment_manifest.json"
  MODE_LABEL="IEEE Multi-Trial Experiment"
fi

# Append data-only / train-only suffixes to mode label
[[ "$DATA_ONLY" -eq 1 ]]  && MODE_LABEL="$MODE_LABEL [DATA-ONLY]"
[[ "$TRAIN_ONLY" -eq 1 ]] && MODE_LABEL="$MODE_LABEL [TRAIN-ONLY]"
[[ "$LAPTOP_SAFE" -eq 1 ]] && MODE_LABEL="$MODE_LABEL [LAPTOP-SAFE]"

initialise_log_lifecycle

# ─── Main entry point ─────────────────────────────────────────────────────────
run_main() {
  local started_at completed_at seeds_csv analytics_root seed status=0

  started_at="$(timestamp_utc)"
  RUN_STARTED_AT_UTC="$started_at"
  seeds_csv="$(seed_list_to_csv "${SEEDS[@]}")"
  analytics_root="$(analytics_results_root)"

  write_log_header "$started_at" "$seeds_csv" "$MODE_LABEL"

  log_line "===================================================================="
  log_line "CostGuard v17.0 / run_ieee_trials.sh v8.0"
  log_line "Mode: $MODE_LABEL"
  log_line "Seeds=${seeds_csv}"
  log_line "Domain order: D0 (Synthetic) -> L1 (TravisTorrent) -> L2 (BitBrains)"
  log_line "Results root: $RESULTS_ROOT"
  log_line "Hyperparameters: epochs=${EPOCHS} lr=${LR} patience=${PATIENCE} hpo=${HPO_TRIALS}"
  log_line "Log file: $LIVE_LOG"
  log_line "===================================================================="

  require_python "$PY"
  require_file "$SCRIPT" "canonical engine"

  # ── Phase A: Data-only ──
  if [[ "$DATA_ONLY" -eq 1 ]]; then
    log_line "[$(timestamp_utc)] [PHASE-A] Starting data preparation only"
    local domains_to_prep=()
    if [[ -n "$PREP_DOMAIN" ]]; then
      domains_to_prep=("$PREP_DOMAIN")
    else
      domains_to_prep=("synthetic" "real" "bitbrains")
    fi
    # Use the first seed for data prep (shared cache)
    local prep_seed="${SEEDS[0]}"
    for domain in "${domains_to_prep[@]}"; do
      log_line "[$(timestamp_utc)] [PHASE-A] Prepping domain=${domain} seed=${prep_seed}"
      run_data_only_domain "$domain" "$prep_seed" || status=$?
      if [[ "$status" -ne 0 ]]; then
        log_line "[ERROR] Data prep failed for domain=${domain} exit=${status}"
        return "$status"
      fi
      inter_seed_gc_sleep "$domain"
    done
    log_line "[$(timestamp_utc)] [PHASE-A] All data preparation complete"
    log_line "Log: $LIVE_LOG"
    return 0
  fi

  # ── Phase B: Train-only ──
  if [[ "$TRAIN_ONLY" -eq 1 ]]; then
    log_line "[$(timestamp_utc)] [PHASE-B] Training only (assumes cache from Phase A)"
    write_experiment_manifest "running" "$started_at" "" "$seeds_csv"
    local first_seed=1
    for seed in "${SEEDS[@]}"; do
      [[ "$first_seed" -eq 0 ]] && inter_seed_gc_sleep "$seed"
      first_seed=0
      log_line "---- Seed ${seed} [TRAIN-ONLY] ----------------------------------------"
      run_seed_trial_train_only "$seed" "$seeds_csv" || status=$?
      if [[ "$status" -ne 0 ]]; then
        completed_at="$(timestamp_utc)"
        write_experiment_manifest "failed" "$started_at" "$completed_at" "$seeds_csv"
        log_line "[ERROR] Seed ${seed} TRAIN-ONLY failed. Stopping."
        return "$status"
      fi
    done
    run_analytics "$analytics_root" || status=$?
    completed_at="$(timestamp_utc)"
    write_experiment_manifest "complete" "$started_at" "$completed_at" "$seeds_csv"
    log_line "[$(timestamp_utc)] [PHASE-B] Training complete"
    log_line "Log: $LIVE_LOG"
    return "$status"
  fi

  # ── Full pipeline (default) ──
  if [[ "$DRY_RUN" -eq 0 ]]; then
    if [[ "$RUN_MODE" != "authoritative" ]] || [[ -n "${TT_CSV:-}" ]]; then
      # Only validate files in non-dry-run full mode
      require_file "$TT_CSV" "TravisTorrent CSV (use --data-only first if not available)"
      require_dir  "$BB_DIR" "BitBrains directory (use --data-only first if not available)"
    fi
  fi
  log_line "Python: $PY  |  Master log: ${MASTER_LOG:-none}"
  write_experiment_manifest "running" "$started_at" "" "$seeds_csv"

  local first_seed=1
  for seed in "${SEEDS[@]}"; do
    [[ "$first_seed" -eq 0 ]] && inter_seed_gc_sleep "$seed"
    first_seed=0
    log_line ""
    log_line "---- Seed ${seed} -------------------------------------------------------"
    run_seed_trial "$seed" "$seeds_csv" || status=$?
    write_experiment_manifest \
      "$([[ "$status" -eq 0 ]] && printf 'running' || printf 'failed')" \
      "$started_at" "" "$seeds_csv"
    if [[ "$status" -ne 0 ]]; then
      completed_at="$(timestamp_utc)"
      write_experiment_manifest "failed" "$started_at" "$completed_at" "$seeds_csv"
      log_line "[ERROR] Seed ${seed} failed. Stopping for safe resume."
      return "$status"
    fi
  done

  run_analytics "$analytics_root" || status=$?
  completed_at="$(timestamp_utc)"
  if [[ "$status" -ne 0 ]]; then
    write_experiment_manifest "failed" "$started_at" "$completed_at" "$seeds_csv"
    return "$status"
  fi

  write_experiment_manifest "complete" "$started_at" "$completed_at" "$seeds_csv"
  log_line "Results root: $RESULTS_ROOT"
  log_line "Aggregate directory: $AGGREGATE_DIR"
  log_line "Live transcript: $LIVE_LOG"
  log_line "Run invocation ID: $RUN_INVOCATION_ID"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%s\n' 'IEEE DRY-RUN COMPLETE - no training or artifacts generated.'
  else
    printf '%s\n' 'IEEE EXPERIMENT COMPLETE - all requested artifacts generated.'
  fi
}

set +e
run_main 2>&1 | tee -a "$LIVE_LOG"
status=${PIPESTATUS[0]}
set -e
exit "$status"
