# CostGuard v17.0

CostGuard is a publication-grade FinOps and ML observability platform built around one canonical IEEE execution path:

- `D0 = Synthetic`
- `L1 = TravisTorrent`
- `L2 = BitBrains`

The hardened repository supports both the canonical single-seed demo profile on `Seed 42` and the official resumable IEEE multi-trial workflow driven by `bash run_ieee_trials.sh --ieee-10`. Both flows preserve the same lifelong order, analytics contract, and publication outputs under a single canonical `results/` artifact store.

## What This Repository Delivers

- A canonical research engine in `CostGuard_PADE_FULL.py`
- A production runtime and dashboard launched with `.\venv\Scripts\python.exe costguard_start.py`
- A structured governance layer backed by Open Policy Agent (OPA) with an inline parity fallback
- A unified analytics engine in `costguard_analytics.py`
- A one-command authoritative trial runner in `run_ieee_trials.sh`
- IEEE-ready artifacts:
  - aggregate JSON
  - CSV exports
  - LaTeX tables
  - 1000 DPI vector figures in `.pdf` and `.eps`

## Canonical Architecture

### Lifelong Domain Order

The execution order is immutable:

1. `D0 -> Synthetic`
2. `L1 -> TravisTorrent`
3. `L2 -> BitBrains`

Legacy fourth-domain logic is retired from the active repository surface. Only the current 3-domain publication workflow is in scope.

### Core Models

- `BahdanauBiLSTM`:
  - 3 residual BiLSTM blocks
  - LayerNorm and dropout between blocks
  - Bahdanau attention over the final sequence states
- `GATv2Pipeline`:
  - 3 graph-attention layers
  - dataset-aware node feature dimensions
  - global mean and max pooling
- Ensemble:
  - validation-optimised thresholding
  - test evaluation with the locked validation threshold

### Governance

CostGuard uses a structured policy payload for governance decisions:

- `metrics`:
  - `crs`
  - `billed_cost`
  - `duration_seconds`
  - `latency_p95`
- `context`:
  - `run_id`
  - `stage_name`
  - `executor_type`
  - `branch`
  - `domain`
  - `gh_is_pr`
  - `gh_by_core_team_member`
- `policy_config`:
  - thresholds
  - protected branches
  - stage-aware cost ceilings
  - deployment and core-team rules

OPA is the preferred decision engine when the service is available. The inline fallback is required to stay semantically aligned with the Rego policy so local and offline execution remain deterministic.

### Unified Analytics Engine

`costguard_analytics.py` is the single analytics authority for:

- T=3 BWT computation
- aggregate statistics across `synthetic`, `real`, and `bitbrains`
- IEEE LaTeX table generation
- 1000 DPI vector figure generation

The canonical BWT formula is:

```text
BWT = 0.5 * ((R(D0|theta_L2) - R(D0|theta_D0)) + (R(L1|theta_L2) - R(L1|theta_L1)))
```

## Authoritative Execution Path

### 1. Smoke Test

```powershell
.\venv\Scripts\python.exe CostGuard_PADE_FULL.py --smoke-test
```

### 2. Official IEEE Workflow

From Git Bash:

```bash
bash run_ieee_trials.sh --ieee-10
```

This is the official one-command IEEE experiment path. It runs the 10-seed resumable trial suite, prints stage-by-stage and epoch-by-epoch progress to the terminal, disables notifier spam by default, writes manifests along the way, and triggers unified analytics plus figure generation automatically at the end.

### 3. Single Authoritative Trial

From Git Bash:

```bash
bash run_ieee_trials.sh
```

This runs the single-seed `Seed 42` profile for a faster demo-oriented pass while keeping the same artifact conventions and compatibility outputs.

### 4. Artifact Layout

After a hardened run, expect a clean project-level layout:

- `results/trials/seed_<N>/`
- `results/trials/seed_<N>/_workspace/`
- `results/shared_cache/`
- `results/aggregate/ieee_aggregate_summary.json`
- `results/aggregate/ieee_aggregate_summary.csv`
- `results/aggregate/ieee_aggregate_summary.tex`
- `results/paper_figures/*.pdf`
- `results/paper_figures/*.eps`
- `results/*/inference_manifest.json` compatibility outputs for demo/runtime consumers
- `training_terminal_logs.md`

Per-seed runs now keep raw prepared data, ML-ready tensors, checkpoints, predictions, run configs, manifests, and lifelong state under one consistent root so the full experiment is easier to audit and demonstrate live.

## Repository Guide

- `COMMANDS.md`: master runbook for setup, execution, analytics, and figures
- `QUICKSTART.md`: shortest path from clone to authoritative run
- `IEEE_EXECUTION_PROTOCOL.md`: reproducibility and artifact contract
- `BEST_COMMANDS_FOR_YOUR_MACHINE.md`: Windows and RTX 3050 tuned commands
- `AGILE_SDLC.md`: engineering quality gates and documentation policy

## Runtime Startup

For the production-style runtime and dashboard, start CostGuard with the project venv interpreter:

```powershell
.\venv\Scripts\python.exe costguard_start.py
```

The launcher validates the active venv, reads the root `.env`, starts PostgreSQL with Docker Compose when available, boots FastAPI on `API_BASE_URL`, and opens the dashboard at `DASHBOARD_BASE_URL`.

## Training Boundary

Runtime hardening and inference integration live in this repository, but model training execution remains a deliberate manual workflow. Do not trigger training from automation unless the developer explicitly requests it. The active runtime only consumes trained checkpoints through the inference-side loader and resolver.

## Verification Gates

```powershell
.\venv\Scripts\python.exe CostGuard_PADE_FULL.py --smoke-test
.\venv\Scripts\python.exe -m pytest tests -q
```

```bash
bash -n run_ieee_trials.sh
```

## Status

The active repository surface is now aligned to the v17.0 3-domain IEEE profile, the OPA governance model, and the unified analytics workflow.
