# Quick Start

This guide runs the hardened CostGuard v17.0 workflows with one runner, one artifact pipeline, and one canonical `results/` store.

## Execution Model

- Official IEEE multi-seed flow: `42,52,62,72,82,92,102,112,122,132`
- Demo seed: `42`
- Domain order: `D0 -> L1 -> L2`
- Runner: `run_ieee_trials.sh`
- Live log sink: `training_terminal_logs.md`
- Publication outputs: JSON, CSV, LaTeX, PDF, EPS

## Prerequisites

- Windows 11 with PowerShell and Git Bash available
- Python 3.11+ or 3.12+
- A working virtual environment at `venv`
- TravisTorrent CSV at `final-2017-01-25.csv`
- BitBrains directory at `fastStorage/2013-8`

## Step 1: Activate the Environment

```powershell
cd C:\Users\Tejas-Kamble\OneDrive\Desktop\Costguard
.\venv\Scripts\Activate.ps1
```

## Step 1A: Start the Runtime Stack

For the hardened FastAPI and dashboard runtime, launch the project with the venv interpreter:

```powershell
.\venv\Scripts\python.exe costguard_start.py
```

This reads `.env`, validates required runtime settings, starts PostgreSQL on `DB_PORT`, boots the API on `API_BASE_URL`, and serves the dashboard on `DASHBOARD_BASE_URL`.

## Step 2: Verify the Canonical Engine

```powershell
.\venv\Scripts\python.exe CostGuard_PADE_FULL.py --smoke-test
```

Expected result:

```text
SMOKE TEST PASSED - 12/12 checks green
```

## Step 3: Run the Official IEEE 10-Trial Pipeline

Open Git Bash in the repository root and execute:

```bash
bash run_ieee_trials.sh --ieee-10
```

What happens:

1. The runner validates the inputs.
2. The 10 IEEE seeds run with safe resume markers.
3. The domains execute in order:
   - synthetic
   - TravisTorrent
   - BitBrains
4. Epoch-by-epoch training progress is printed live in the terminal.
5. Unified analytics writes aggregate summaries and publication figures automatically.

## Step 3B: Split Workflow For Cache Reuse

Use this path when you want to prepare data once and then rerun only training across many seeds.

```bash
PY=./venv/Scripts/python.exe
RESULTS_ROOT=results
CACHE_ROOT="$RESULTS_ROOT/shared_cache"
```

Prepare shared data once:

```bash
"$PY" CostGuard_PADE_FULL.py --generate --preprocess-only --data-mode synthetic --seed 42 --results-base "$RESULTS_ROOT" --raw-dir "$CACHE_ROOT/synthetic_raw" --ml-ready-dir "$CACHE_ROOT/ml_ready_synthetic"
"$PY" CostGuard_PADE_FULL.py --ingest --preprocess-only --data-mode real --real-input ./final-2017-01-25.csv --seed 42 --results-base "$RESULTS_ROOT" --raw-dir "$CACHE_ROOT/real_data" --ml-ready-dir "$CACHE_ROOT/ml_ready_real"
"$PY" CostGuard_PADE_FULL.py --ingest --preprocess-only --data-mode bitbrains --bitbrains-dir ./fastStorage/2013-8 --seed 42 --results-base "$RESULTS_ROOT" --raw-dir "$CACHE_ROOT/bitbrains_data" --ml-ready-dir "$CACHE_ROOT/ml_ready_bitbrains"
```

Then train a seed from the prepared cache:

```bash
SEED=42
SEED_ROOT="$RESULTS_ROOT/trials/seed_$SEED"

"$PY" CostGuard_PADE_FULL.py --train --data-mode synthetic --seed "$SEED" --hpo-trials 0 --skip-data --skip-preprocess --results-base "$SEED_ROOT" --raw-dir "$CACHE_ROOT/synthetic_raw" --ml-ready-dir "$CACHE_ROOT/ml_ready_synthetic" --force-run-dir "$SEED_ROOT/synthetic/run_1" --quiet-epochs
"$PY" CostGuard_PADE_FULL.py --train-lifelong --data-mode real --real-input ./final-2017-01-25.csv --seed "$SEED" --hpo-trials 0 --skip-data --skip-preprocess --results-base "$SEED_ROOT" --raw-dir "$CACHE_ROOT/real_data" --ml-ready-dir "$CACHE_ROOT/ml_ready_real" --force-run-dir "$SEED_ROOT/real/run_1" --quiet-epochs
"$PY" CostGuard_PADE_FULL.py --train-lifelong --data-mode bitbrains --bitbrains-dir ./fastStorage/2013-8 --seed "$SEED" --hpo-trials 0 --skip-data --skip-preprocess --results-base "$SEED_ROOT" --raw-dir "$CACHE_ROOT/bitbrains_data" --ml-ready-dir "$CACHE_ROOT/ml_ready_bitbrains" --force-run-dir "$SEED_ROOT/bitbrains/run_1" --quiet-epochs
```

## Step 3A: Run the Faster Single-Seed Demo Path

```bash
bash run_ieee_trials.sh
```

This keeps the same hardened logging and artifact behavior, but runs the single authoritative `Seed 42` profile for a faster demonstration.

## Step 4: Inspect the Outputs

Primary outputs:

- `results/trials/seed_<N>/`
- `results/trials/seed_<N>/_workspace/`
- `results/shared_cache/`
- `results/aggregate/ieee_aggregate_summary.json`
- `results/aggregate/ieee_aggregate_summary.csv`
- `results/aggregate/ieee_aggregate_summary.tex`
- `results/paper_figures/ieee_f1_comparison.pdf`
- `results/paper_figures/ieee_f1_comparison.eps`
- `results/paper_figures/ieee_roc_auc_heatmap.pdf`
- `results/paper_figures/ieee_bwt_summary.pdf`
- `training_terminal_logs.md`

## Manual Analytics Rebuild

If you need to rerun analytics separately:

```powershell
.\venv\Scripts\python.exe aggregate_results.py --results-root results --latex --csv
.\venv\Scripts\python.exe generate_paper_figures.py --results-dir results --aggregate-json results\aggregate\ieee_aggregate_summary.json --dpi 1000 --formats pdf eps
```

## Runner Defaults

The official IEEE runner now defaults to:

- `--patience 10`
- verbose epoch logging enabled
- notifier delivery disabled unless you opt in with `--enable-notifier`
- canonical artifacts under `results/`

## Policy Layer Notes

CostGuard includes structured OPA governance for:

- stage-aware cost ceilings
- protected branch enforcement
- PR restrictions for sensitive deployment stages
- optional core-team gating

If OPA is unavailable, the inline fallback remains active and aligned with the same decision rules.

## Training Boundary

The developer owns all actual training execution. Runtime hardening may load or validate trained checkpoints for inference, but it must not start training jobs, rewrite training loops, or alter model architecture without explicit instruction.

## Test Suite

```powershell
$tmp = Join-Path $env:TEMP ('costguard_pytest_' + [guid]::NewGuid().ToString('N'))
$cov = Join-Path $env:TEMP ('costguard_coverage_' + [guid]::NewGuid().ToString('N'))
$env:COVERAGE_FILE = $cov
.\venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=$tmp --cov=backend --cov-report=term
```

## Completion Signal

A successful authoritative run ends with:

```text
IEEE MASTERPIECE COMPLETE — ALL ARTIFACTS GENERATED.
```
