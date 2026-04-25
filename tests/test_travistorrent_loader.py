from pathlib import Path

import pandas as pd

from CostGuard_PADE_FULL import load_travistorrent


def test_load_travistorrent_handles_unsorted_build_times(tmp_path: Path):
    csv_path = tmp_path / 'travistorrent.csv'
    out_dir = tmp_path / 'real_data'
    csv_path.write_text(
        (
            "tr_build_id,gh_project_name,git_branch,gh_build_started_at,tr_status,tr_duration,tr_log_buildduration,"
            "tr_log_num_tests_run,tr_log_num_tests_failed,tr_log_num_tests_ok,tr_log_num_test_suites_failed,"
            "git_diff_src_churn,git_diff_test_churn,gh_diff_files_added,gh_diff_files_deleted,gh_diff_files_modified,"
            "gh_team_size,gh_num_commits_in_push,gh_num_issue_comments,gh_num_pr_comments,gh_repo_num_commits,"
            "gh_sloc,gh_test_lines_per_kloc,gh_lang,gh_by_core_team_member,gh_is_pr\n"
            "1,proj_a,main,2025-01-04T00:00:00,passed,10,10,100,0,100,0,1,1,1,0,1,5,2,0,0,1000,12000,80,Python,FALSE,FALSE\n"
            "2,proj_a,main,2025-01-01T00:00:00,passed,10,10,100,0,100,0,1,1,1,0,1,5,2,0,0,1000,12000,80,Python,FALSE,FALSE\n"
            "3,proj_a,main,2025-01-06T00:00:00,passed,100,100,100,0,100,0,1,1,1,0,1,5,2,0,0,1000,12000,80,Python,FALSE,FALSE\n"
            "4,proj_a,main,2025-01-02T00:00:00,passed,10,10,100,0,100,0,1,1,1,0,1,5,2,0,0,1000,12000,80,Python,FALSE,FALSE\n"
            "5,proj_a,main,2025-01-05T00:00:00,passed,10,10,100,0,100,0,1,1,1,0,1,5,2,0,0,1000,12000,80,Python,FALSE,FALSE\n"
            "6,proj_a,main,2025-01-03T00:00:00,passed,10,10,100,0,100,0,1,1,1,0,1,5,2,0,0,1000,12000,80,Python,FALSE,FALSE\n"
        ),
        encoding='utf-8',
    )

    rows = load_travistorrent(csv_path, out_dir)

    assert rows == 6
    focus_df = pd.read_csv(out_dir / 'pipeline_stage_telemetry.csv')
    assert focus_df['anomaly_window_active'].sum() == 1
    assert (
        focus_df.loc[focus_df['run_id'] == 'tt_3', 'anomaly_window_active'].iloc[0]
        == 1
    )


def test_load_travistorrent_validates_required_columns(tmp_path: Path):
    csv_path = tmp_path / 'travistorrent_missing_col.csv'
    out_dir = tmp_path / 'real_data'
    csv_path.write_text(
        (
            "tr_build_id,gh_project_name,git_branch,gh_build_started_at,tr_status,tr_log_buildduration,"
            "tr_log_num_tests_run,tr_log_num_tests_failed,tr_log_num_tests_ok,tr_log_num_test_suites_failed,"
            "git_diff_src_churn,git_diff_test_churn,gh_diff_files_added,gh_diff_files_deleted,gh_diff_files_modified,"
            "gh_team_size,gh_num_commits_in_push,gh_sloc,gh_test_lines_per_kloc,gh_lang,gh_by_core_team_member,gh_is_pr\n"
            "1,proj_a,main,2025-01-04T00:00:00,passed,10,100,0,100,0,1,1,1,0,1,5,2,12000,80,Python,FALSE,FALSE\n"
        ),
        encoding='utf-8',
    )

    try:
        load_travistorrent(csv_path, out_dir)
    except ValueError as exc:
        message = str(exc)
        assert 'TRAVISTORRENT' in message
        assert 'tr_duration' in message
    else:
        raise AssertionError('expected missing-column validation to raise ValueError')
