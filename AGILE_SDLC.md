# Agile SDLC and Quality Gates

## Active Engineering Scope

This repository now operates as a single-source publication system anchored on:

- one canonical ML engine
- one canonical governance layer
- one canonical analytics engine
- one authoritative `Seed 42` trial profile

The active publication execution order is:

1. `D0 = Synthetic`
2. `L1 = TravisTorrent`
3. `L2 = BitBrains`

## SDLC Principles

### 1. Single Source of Truth

- `CostGuard_PADE_FULL.py` is the authoritative training and evaluation engine.
- Backend PADE adapters must wrap, not fork, the canonical engine.
- `costguard_analytics.py` is the sole analytics authority for BWT, LaTeX, CSV, JSON, and figures.

### 2. Governance by Policy

All runtime decisions that can affect cost, risk, or access must flow through the OPA-aligned policy contract.

Policy coverage includes:

- CRS thresholding
- stage-aware budget limits
- protected branch controls
- deployment restrictions for PRs
- optional core-team controls for sensitive stages

### 3. Documentation as Release Surface

Operational documentation is part of the release. Any change to the runner, governance layer, or analytics outputs must update:

- `README.md`
- `QUICKSTART.md`
- `IEEE_EXECUTION_PROTOCOL.md`
- `COMMANDS.md`

## Quality Gates

### Code Gate

- no placeholder code
- no silent exception swallowing in active paths
- UTF-8 without BOM across repo-owned text files
- no stale fourth-domain references in active source and docs

### Test Gate

- `CostGuard_PADE_FULL.py --smoke-test` must pass `12/12`
- backend `pytest` suite must pass before release
- policy parity coverage must remain green
- analytics regression coverage must remain green

### Orchestration Gate

The authoritative runner must:

- execute only `Seed 42`
- preserve the `D0 -> L1 -> L2` order
- append terminal output to `training_terminal_logs.md` live
- trigger aggregate and figure generation automatically
- finish with the IEEE completion message

## CI/CD Expectations

The active CI path should validate:

- Python syntax and import integrity
- backend pytest suite
- shell syntax for `run_ieee_trials.sh`
- smoke test for the canonical engine
- analytics wrapper integrity

## Publication Artifact Policy

Release candidates are only valid when all of the following exist:

- `results/seed_42/trial_complete.json`
- `results/aggregate/ieee_aggregate_summary.json`
- `results/aggregate/ieee_aggregate_summary.tex`
- vector figures in `results/paper_figures/`
- live execution transcript in `training_terminal_logs.md`

## Documentation Hygiene Policy

Repo-owned markdown must remain aligned with the active architecture.

Prohibited in active docs:

- retired legacy dataset operating guides
- fourth-domain execution language
- obsolete multi-domain workflow instructions that conflict with the active 3-domain profile

Historical material may be retained only as archived notes and must point readers back to the current v17.0 documentation set.
