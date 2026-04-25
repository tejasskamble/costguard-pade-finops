# Best Commands for Your Machine

This runbook is tuned for the current CostGuard workstation profile and the single authoritative v17.0 publication run.

## Machine Profile

- OS: Windows 11
- Shells: PowerShell + Git Bash
- GPU target: RTX 3050 4 GB
- Repository root: `C:\Users\Tejas-Kamble\OneDrive\Desktop\Costguard`

## Standard Session Start

```powershell
cd C:\Users\Tejas-Kamble\OneDrive\Desktop\Costguard
.\venv\Scripts\Activate.ps1
```

## Fast Sanity Checks

```powershell
.\venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
.\venv\Scripts\python.exe CostGuard_PADE_FULL.py --smoke-test
```

## Full Backend Test Pass

```powershell
$tmp = Join-Path $env:TEMP ('costguard_pytest_' + [guid]::NewGuid().ToString('N'))
$cov = Join-Path $env:TEMP ('costguard_coverage_' + [guid]::NewGuid().ToString('N'))
$env:COVERAGE_FILE = $cov
.\venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=$tmp --cov=backend --cov-report=term
```

## Authoritative Trial

Run from Git Bash:

```bash
bash run_ieee_trials.sh CostGuard_PADE_FULL.py final-2017-01-25.csv fastStorage/2013-8
```

What it does:

- locks execution to `Seed 42`
- executes `D0 -> L1 -> L2`
- keeps OPA governance checks active, with the inline parity fallback still available for offline execution
- appends terminal output live to `training_terminal_logs.md`
- builds aggregate JSON, CSV, LaTeX, PDF, and EPS outputs automatically

## Manual Data Generation Only

```powershell
.\venv\Scripts\python.exe CostGuard_PADE_FULL.py --generate --seed 42 --force
```

## Manual Unified Analytics Rebuild

```powershell
.\venv\Scripts\python.exe aggregate_results.py --results-root results --latex --csv
.\venv\Scripts\python.exe generate_paper_figures.py --results-dir results --aggregate-json results\aggregate\ieee_aggregate_summary.json --dpi 1000 --formats pdf eps
```

## Shell Validation

```bash
bash -n run_ieee_trials.sh
```

## Artifact Checks

```powershell
Get-ChildItem results\aggregate
Get-ChildItem results\paper_figures
Get-Content training_terminal_logs.md -Tail 40
```

## Expected Completion Signal

```text
IEEE MASTERPIECE COMPLETE — ALL ARTIFACTS GENERATED.
```

## Notes

- Use Git Bash or another Bash-compatible shell for the orchestrator; the runner mirrors stdout and stderr into `training_terminal_logs.md` through a `tee -a` pipeline.
- Keep TravisTorrent and BitBrains inputs in place before launching the trial.
- The repository no longer uses any fourth-domain execution path in the active workflow.
