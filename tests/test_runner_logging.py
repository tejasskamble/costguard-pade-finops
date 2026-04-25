from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_bash() -> str | None:
    candidates = [
        shutil.which("bash"),
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            candidate_path = Path(candidate)
            if candidate_path.name.lower() == "bash.exe" and "system32" in candidate_path.parts:
                continue
            return str(candidate_path)
    return None


@pytest.mark.skipif(_resolve_bash() is None, reason="bash is required for runner tests")
def test_runner_archives_old_logs_and_starts_fresh(tmp_path: Path):
    bash_path = _resolve_bash()
    assert bash_path is not None
    harness = tmp_path / "runner_harness"
    harness.mkdir(parents=True, exist_ok=True)

    script_src = REPO_ROOT / "run_ieee_trials.sh"
    script_dst = harness / "run_ieee_trials.sh"
    script_dst.write_text(script_src.read_text(encoding="utf-8"), encoding="utf-8")

    for stub_name in ("CostGuard_PADE_FULL.py", "aggregate_results.py", "generate_paper_figures.py"):
        (harness / stub_name).write_text("print('stub')\n", encoding="utf-8")

    results_root = harness / "results"
    seed_dir = results_root / "seed_42"
    seed_dir.mkdir(parents=True, exist_ok=True)

    live_log = harness / "training_terminal_logs.md"
    master_log = results_root / "ieee_audit.log"
    seed_log = seed_dir / "seed_run.log"
    live_log.write_text("OLD LIVE SENTINEL\n", encoding="utf-8")
    master_log.parent.mkdir(parents=True, exist_ok=True)
    master_log.write_text("OLD MASTER SENTINEL\n", encoding="utf-8")
    seed_log.write_text("OLD SEED SENTINEL\n", encoding="utf-8")

    completed = subprocess.run(
        [bash_path, "run_ieee_trials.sh", "--authoritative", "--dry-run"],
        cwd=harness,
        capture_output=True,
        text=True,
        check=True,
    )

    archive_root = results_root / "log_archive"
    archive_dirs = [path for path in archive_root.iterdir() if path.is_dir()]
    assert archive_dirs, "expected a run archive directory"
    run_archive = archive_dirs[0]

    assert (run_archive / "training_terminal_logs.md").read_text(encoding="utf-8") == "OLD LIVE SENTINEL\n"
    assert (run_archive / "ieee_audit.log").read_text(encoding="utf-8") == "OLD MASTER SENTINEL\n"

    seed_archive_dirs = [path for path in (seed_dir / "log_archive").iterdir() if path.is_dir()]
    assert seed_archive_dirs, "expected a seed archive directory"
    assert (seed_archive_dirs[0] / "seed_run.log").read_text(encoding="utf-8") == "OLD SEED SENTINEL\n"

    current_live = live_log.read_text(encoding="utf-8")
    current_master = master_log.read_text(encoding="utf-8")
    current_seed = seed_log.read_text(encoding="utf-8")
    assert "OLD LIVE SENTINEL" not in current_live
    assert "OLD MASTER SENTINEL" not in current_master
    assert "OLD SEED SENTINEL" not in current_seed
    assert "Invocation ID" in current_live
    assert "Run invocation ID:" in current_master
    assert "[SEED][START]" in current_seed

    manual_logs_dir = results_root / "manual_logs"
    manual_logs = sorted(manual_logs_dir.glob("*.md"))
    assert len(manual_logs) >= 4
    seed_manual_log = next(path for path in manual_logs if "d0-synthetic" in path.name)
    aggregate_manual_log = next(path for path in manual_logs if path.name.startswith("aggregate_"))

    seed_manual_text = seed_manual_log.read_text(encoding="utf-8")
    assert "## START" in seed_manual_text
    assert "## HARDWARE INFO" in seed_manual_text
    assert "## DATA PREP" in seed_manual_text
    assert "## TRAINING (LSTM)" in seed_manual_text
    assert "## TRAINING (GAT)" in seed_manual_text
    assert "## LIFELONG" in seed_manual_text
    assert "## SYSTEM STATS" in seed_manual_text

    aggregate_manual_text = aggregate_manual_log.read_text(encoding="utf-8")
    assert "## AGGREGATION" in aggregate_manual_text
    assert "## LOGS" in aggregate_manual_text

    manifest = json.loads((results_root / "authoritative_experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_invocation_id"]
    assert manifest["current_log_paths"]["live_log"].endswith("training_terminal_logs.md")
    assert Path(manifest["current_log_paths"]["manual_logs_dir"]).name == "manual_logs"
    assert manifest["archived_log_paths"]["master_log"]
    assert "[AGGREGATE][START]" in completed.stdout
