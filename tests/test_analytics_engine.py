"""Analytics-engine regression tests for CostGuard v17.0."""
import json
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest

from costguard_analytics import compute_bwt_from_matrix, resolve_seed_root, write_aggregate_bundle


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_ROOT = REPO_ROOT / 'test_runtime_tmp'


@contextmanager
def _results_root_fixture():
    scratch_dir = SCRATCH_ROOT / f'analytics_{uuid4().hex}'
    results_root = scratch_dir / 'results'
    results_root.mkdir(parents=True, exist_ok=True)
    try:
        yield results_root
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def _best_scores(domain: str, seed: int, base: float) -> dict:
    return {
        'best_lstm': {
            'f1_at_opt': base,
            'roc_auc': base + 0.05,
            'pr_auc': base + 0.02,
            'precision': base - 0.01,
            'recall': base + 0.01,
            'threshold': 0.50,
        },
        'best_gat': {
            'f1_at_opt': base + 0.02,
            'roc_auc': base + 0.06,
            'pr_auc': base + 0.03,
            'precision': base,
            'recall': base + 0.02,
            'threshold': 0.52,
        },
        'best_ens': {
            'f1_at_opt': base + 0.04,
            'roc_auc': base + 0.08,
            'pr_auc': base + 0.05,
            'precision': base + 0.01,
            'recall': base + 0.03,
            'threshold': 0.54,
        },
        'hpo_triggered': False,
        'domain': domain,
        'seed': seed,
    }


def _seed_fixture(results_root: Path, seed: int, offset: float, complete: bool = True) -> None:
    seed_dir = results_root / f'seed_{seed}'
    for domain, base in [('synthetic', 0.80 + offset), ('real', 0.76 + offset), ('bitbrains', 0.74 + offset)]:
        domain_dir = seed_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        (domain_dir / 'best_scores.json').write_text(
            json.dumps(_best_scores(domain, seed, base), indent=2),
            encoding='utf-8',
        )
    (seed_dir / 'bwt_matrix.json').write_text(
        json.dumps(
            {
                'after_D0': {'D0': 0.90},
                'after_L1': {'D0': 0.89, 'L1': 0.88},
                'after_L2': {'D0': 0.88, 'L1': 0.87, 'L2': 0.86},
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    if complete:
        (seed_dir / 'trial_complete.json').write_text(
            json.dumps({'seed': seed, 'completed_at': '2026-01-01T00:00:00'}),
            encoding='utf-8',
        )


class TestAnalyticsCore:
    def test_compute_bwt_matches_t3_formula(self):
        value = compute_bwt_from_matrix(
            {
                'after_D0': {'D0': 0.90},
                'after_L1': {'D0': 0.89, 'L1': 0.88},
                'after_L2': {'D0': 0.87, 'L1': 0.86, 'L2': 0.85},
            }
        )
        assert value == pytest.approx(0.5 * ((0.87 - 0.90) + (0.86 - 0.88)))

    def test_write_aggregate_bundle_writes_json_csv_and_tex(self):
        with _results_root_fixture() as results_root:
            _seed_fixture(results_root, 0, 0.00)
            _seed_fixture(results_root, 7, 0.01)

            summary = write_aggregate_bundle(results_root, write_csv_enabled=True, write_latex_enabled=True)
            aggregate_dir = results_root / 'aggregate'
            assert aggregate_dir.joinpath('ieee_aggregate_summary.json').exists()
            assert aggregate_dir.joinpath('ieee_aggregate_summary.csv').exists()
            assert aggregate_dir.joinpath('ieee_aggregate_summary.tex').exists()
            assert aggregate_dir.joinpath('ieee_per_seed_summary.json').exists()
            assert aggregate_dir.joinpath('ieee_per_seed_summary.csv').exists()
            assert summary['completed_trials'] == 2
            assert summary['total_trials'] == 2
            assert summary['domains']['synthetic']['ens']['f1_at_opt']['mean'] > 0

    def test_write_aggregate_bundle_supports_nested_trials_layout(self):
        with _results_root_fixture() as results_root:
            trials_root = results_root / 'trials'
            _seed_fixture(trials_root, 42, 0.00)
            _seed_fixture(trials_root, 52, 0.02)

            summary = write_aggregate_bundle(
                results_root,
                write_csv_enabled=True,
                write_latex_enabled=True,
                aggregate_dir=results_root / 'aggregate',
            )

            assert resolve_seed_root(results_root) == trials_root
            assert summary['completed_trials'] == 2
            assert summary['total_trials'] == 2
            assert summary['domains']['real']['ens']['f1_at_opt']['n'] == 2

    def test_write_aggregate_bundle_excludes_missing_metrics_from_n(self):
        with _results_root_fixture() as results_root:
            _seed_fixture(results_root, 42, 0.00)
            partial_dir = results_root / 'seed_52'
            partial_dir.mkdir(parents=True, exist_ok=True)

            summary = write_aggregate_bundle(results_root)

            assert summary['completed_trials'] == 1
            assert summary['total_trials'] == 2
            assert summary['domains']['synthetic']['ens']['f1_at_opt']['n'] == 1
            assert summary['domains']['synthetic']['ens']['f1_at_opt']['mean'] == pytest.approx(0.84)

    def test_wrapper_scripts_run_against_fixture_results(self):
        with _results_root_fixture() as results_root:
            trials_root = results_root / 'trials'
            _seed_fixture(trials_root, 0, 0.00)
            _seed_fixture(trials_root, 7, 0.01)
            aggregate_dir = results_root / 'aggregate'
            figure_dir = results_root / 'paper_figures'

            aggregate = subprocess.run(
                [
                    sys.executable,
                    'aggregate_results.py',
                    '--results-root',
                    str(results_root),
                    '--aggregate-dir',
                    str(aggregate_dir),
                    '--latex',
                    '--csv',
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            assert 'Aggregate complete' in aggregate.stdout
            assert '[AGGREGATE][START]' in aggregate.stdout
            assert '[AGGREGATE][END]' in aggregate.stdout

            figures = subprocess.run(
                [
                    sys.executable,
                    'generate_paper_figures.py',
                    '--results-dir',
                    str(results_root),
                    '--aggregate-json',
                    str(aggregate_dir / 'ieee_aggregate_summary.json'),
                    '--out-dir',
                    str(figure_dir),
                    '--dpi',
                    '300',
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            assert 'Figure generation complete' in figures.stdout
            assert '[FIGURE][START]' in figures.stdout
            assert '[FIGURE][END]' in figures.stdout
            assert figure_dir.exists()
            assert any(figure_dir.glob('*.pdf'))
