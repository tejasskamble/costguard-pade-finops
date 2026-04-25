# COMMANDS.md

This is the master runbook for the full CostGuard v17.0 authoritative workflow.

## 1. Environment Setup

### PowerShell

```powershell
cd C:\Users\Tejas-Kamble\OneDrive\Desktop\Costguard
.\venv\Scripts\Activate.ps1
.\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\venv\Scripts\python.exe -m pip install pytest==8.2.0 pytest-asyncio==0.23.7 pytest-cov==5.0.0 pyyaml==6.0.1
```

## 2. Smoke Validation

```powershell
.\venv\Scripts\python.exe CostGuard_PADE_FULL.py --smoke-test
```

## 3. Synthetic Data Generation Only

```powershell
.\venv\Scripts\python.exe CostGuard_PADE_FULL.py --generate --seed 42 --force
```

## 4. Shared Data Preparation Once

Run these from Git Bash to prepare shared cache artifacts exactly once per dataset. Each command stops after data loading and preprocessing because of `--preprocess-only`.

```bash
PY=./venv/Scripts/python.exe
RESULTS_ROOT=results
CACHE_ROOT="$RESULTS_ROOT/shared_cache"

"$PY" CostGuard_PADE_FULL.py \
  --generate \
  --preprocess-only \
  --data-mode synthetic \
  --seed 42 \
  --results-base "$RESULTS_ROOT" \
  --raw-dir "$CACHE_ROOT/synthetic_raw" \
  --ml-ready-dir "$CACHE_ROOT/ml_ready_synthetic"

"$PY" CostGuard_PADE_FULL.py \
  --ingest \
  --preprocess-only \
  --data-mode real \
  --real-input ./final-2017-01-25.csv \
  --seed 42 \
  --results-base "$RESULTS_ROOT" \
  --raw-dir "$CACHE_ROOT/real_data" \
  --ml-ready-dir "$CACHE_ROOT/ml_ready_real"

"$PY" CostGuard_PADE_FULL.py \
  --ingest \
  --preprocess-only \
  --data-mode bitbrains \
  --bitbrains-dir ./fastStorage/2013-8 \
  --seed 42 \
  --results-base "$RESULTS_ROOT" \
  --raw-dir "$CACHE_ROOT/bitbrains_data" \
  --ml-ready-dir "$CACHE_ROOT/ml_ready_bitbrains"
```

## 5. Training Only From Shared Cache

The commands below reuse the shared cache and write seed-specific outputs under `results/trials/seed_<N>/`. They train, threshold, persist checkpoints, run lifelong learning where applicable, and leave aggregation to the final analytics step.

```bash
PY=./venv/Scripts/python.exe
RESULTS_ROOT=results
CACHE_ROOT="$RESULTS_ROOT/shared_cache"
SEED=42
SEED_ROOT="$RESULTS_ROOT/trials/seed_$SEED"

"$PY" CostGuard_PADE_FULL.py \
  --train \
  --data-mode synthetic \
  --seed "$SEED" \
  --epochs 150 \
  --lr 5e-4 \
  --patience 10 \
  --scheduler cosine \
  --hpo-trials 0 \
  --skip-data \
  --skip-preprocess \
  --results-base "$SEED_ROOT" \
  --raw-dir "$CACHE_ROOT/synthetic_raw" \
  --ml-ready-dir "$CACHE_ROOT/ml_ready_synthetic" \
  --force-run-dir "$SEED_ROOT/synthetic/run_1" \
  --quiet-epochs

"$PY" CostGuard_PADE_FULL.py \
  --train-lifelong \
  --data-mode real \
  --real-input ./final-2017-01-25.csv \
  --seed "$SEED" \
  --epochs 150 \
  --lr 5e-4 \
  --patience 10 \
  --scheduler cosine \
  --hpo-trials 0 \
  --skip-data \
  --skip-preprocess \
  --results-base "$SEED_ROOT" \
  --raw-dir "$CACHE_ROOT/real_data" \
  --ml-ready-dir "$CACHE_ROOT/ml_ready_real" \
  --force-run-dir "$SEED_ROOT/real/run_1" \
  --quiet-epochs

"$PY" CostGuard_PADE_FULL.py \
  --train-lifelong \
  --data-mode bitbrains \
  --bitbrains-dir ./fastStorage/2013-8 \
  --seed "$SEED" \
  --epochs 150 \
  --lr 5e-4 \
  --patience 10 \
  --scheduler cosine \
  --hpo-trials 0 \
  --skip-data \
  --skip-preprocess \
  --results-base "$SEED_ROOT" \
  --raw-dir "$CACHE_ROOT/bitbrains_data" \
  --ml-ready-dir "$CACHE_ROOT/ml_ready_bitbrains" \
  --force-run-dir "$SEED_ROOT/bitbrains/run_1" \
  --quiet-epochs
```

## 6. IEEE-10 Training Loop From Shared Cache

```bash
PY=./venv/Scripts/python.exe
RESULTS_ROOT=results
CACHE_ROOT="$RESULTS_ROOT/shared_cache"
SEEDS=(42 52 62 72 82 92 102 112 122 132)

for SEED in "${SEEDS[@]}"; do
  SEED_ROOT="$RESULTS_ROOT/trials/seed_$SEED"

  "$PY" CostGuard_PADE_FULL.py \
    --train \
    --data-mode synthetic \
    --seed "$SEED" \
    --epochs 150 \
    --lr 5e-4 \
    --patience 10 \
    --scheduler cosine \
    --hpo-trials 0 \
    --skip-data \
    --skip-preprocess \
    --results-base "$SEED_ROOT" \
    --raw-dir "$CACHE_ROOT/synthetic_raw" \
    --ml-ready-dir "$CACHE_ROOT/ml_ready_synthetic" \
    --force-run-dir "$SEED_ROOT/synthetic/run_1" \
    --quiet-epochs

  "$PY" CostGuard_PADE_FULL.py \
    --train-lifelong \
    --data-mode real \
    --real-input ./final-2017-01-25.csv \
    --seed "$SEED" \
    --epochs 150 \
    --lr 5e-4 \
    --patience 10 \
    --scheduler cosine \
    --hpo-trials 0 \
    --skip-data \
    --skip-preprocess \
    --results-base "$SEED_ROOT" \
    --raw-dir "$CACHE_ROOT/real_data" \
    --ml-ready-dir "$CACHE_ROOT/ml_ready_real" \
    --force-run-dir "$SEED_ROOT/real/run_1" \
    --quiet-epochs

  "$PY" CostGuard_PADE_FULL.py \
    --train-lifelong \
    --data-mode bitbrains \
    --bitbrains-dir ./fastStorage/2013-8 \
    --seed "$SEED" \
    --epochs 150 \
    --lr 5e-4 \
    --patience 10 \
    --scheduler cosine \
    --hpo-trials 0 \
    --skip-data \
    --skip-preprocess \
    --results-base "$SEED_ROOT" \
    --raw-dir "$CACHE_ROOT/bitbrains_data" \
    --ml-ready-dir "$CACHE_ROOT/ml_ready_bitbrains" \
    --force-run-dir "$SEED_ROOT/bitbrains/run_1" \
    --quiet-epochs
done

"$PY" aggregate_results.py \
  --results-root "$RESULTS_ROOT/trials" \
  --aggregate-dir "$RESULTS_ROOT/aggregate" \
  --latex \
  --csv

"$PY" generate_paper_figures.py \
  --results-dir "$RESULTS_ROOT/trials" \
  --aggregate-json "$RESULTS_ROOT/aggregate/ieee_aggregate_summary.json" \
  --out-dir "$RESULTS_ROOT/paper_figures" \
  --dpi 1000 \
  --formats pdf eps
```

## 7. Single Authoritative Trial

Run from Git Bash so the script can mirror stdout and stderr live into `training_terminal_logs.md`.

```bash
bash run_ieee_trials.sh CostGuard_PADE_FULL.py final-2017-01-25.csv fastStorage/2013-8
```

### What the Runner Does

- validates the TravisTorrent and BitBrains inputs
- fixes the seed to `42`
- executes:
  1. synthetic
  2. TravisTorrent
  3. BitBrains
- keeps structured OPA governance active for policy evaluation, with inline parity fallback if OPA is unavailable
- appends terminal output to `training_terminal_logs.md`
- writes `results/ieee_audit.log`
- triggers aggregate reporting and figure generation automatically

## 8. Unified Analytics Only

```powershell
.\venv\Scripts\python.exe aggregate_results.py --results-root results --latex --csv
```

This writes:

- `results/aggregate/ieee_aggregate_summary.json`
- `results/aggregate/ieee_aggregate_summary.csv`
- `results/aggregate/ieee_aggregate_summary.tex`

## 9. Unified Figure Generation Only

```powershell
.\venv\Scripts\python.exe generate_paper_figures.py --results-dir results --aggregate-json results\aggregate\ieee_aggregate_summary.json --dpi 1000 --formats pdf eps
```

This writes high-resolution vector artifacts such as:

- `results/paper_figures/ieee_f1_comparison.pdf`
- `results/paper_figures/ieee_f1_comparison.eps`
- `results/paper_figures/ieee_roc_auc_heatmap.pdf`
- `results/paper_figures/ieee_bwt_summary.pdf`

## 10. BWT Verification

```powershell
.\venv\Scripts\python.exe -c "import json, pathlib; s=json.loads(pathlib.Path('results/aggregate/ieee_aggregate_summary.json').read_text(encoding='utf-8')); print('BWT mean:', s['bwt']['mean']); print('BWT std :', s['bwt']['std'])"
```

## 11. Full Backend Test Suite

Use Windows temp locations for the pytest base temp and coverage file to avoid OneDrive locking issues.

```powershell
$tmp = Join-Path $env:TEMP ('costguard_pytest_' + [guid]::NewGuid().ToString('N'))
$cov = Join-Path $env:TEMP ('costguard_coverage_' + [guid]::NewGuid().ToString('N'))
$env:COVERAGE_FILE = $cov
.\venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=$tmp --cov=backend --cov-report=term
```

## 12. Bash Syntax Validation

```bash
bash -n run_ieee_trials.sh
```

## 13. Completion Message

A successful full run ends with:

```text
IEEE MASTERPIECE COMPLETE — ALL ARTIFACTS GENERATED.
```
