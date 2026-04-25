from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import CostGuard_PADE_FULL as cg
import costguard_runtime as rt


def _tiny_task_b_tensors():
    torch.manual_seed(7)
    x_train = torch.randn(8, cg.SEQ_LEN, cg.N_CHANNELS, dtype=torch.float32)
    c_train = torch.randn(8, cg.N_CTX, dtype=torch.float32)
    y_train = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)

    x_val = torch.randn(4, cg.SEQ_LEN, cg.N_CHANNELS, dtype=torch.float32)
    c_val = torch.randn(4, cg.N_CTX, dtype=torch.float32)
    y_val = torch.tensor([0, 1, 0, 1], dtype=torch.float32)

    x_test = torch.randn(4, cg.SEQ_LEN, cg.N_CHANNELS, dtype=torch.float32)
    c_test = torch.randn(4, cg.N_CTX, dtype=torch.float32)
    y_test = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    return x_train, c_train, y_train, x_val, c_val, y_val, x_test, c_test, y_test


def test_parse_args_supports_patience_verbosity_and_notifier(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "CostGuard_PADE_FULL.py",
            "--patience",
            "7",
            "--quiet-epochs",
            "--disable-notifier",
        ],
    )
    args = cg.parse_args()

    assert args.patience == 7
    assert args.verbose_epochs is False
    assert args.disable_notifier is True


def test_default_artifact_dirs_resolve_inside_results_workspace(tmp_path: Path):
    results_root = tmp_path / "results"

    synthetic_raw = cg._default_raw_dir(results_root, "synthetic")
    real_ml = cg._default_ml_ready_dir(results_root, "real")
    brain_dir = cg._default_brain_dir(results_root)

    assert synthetic_raw == results_root / "_workspace" / "synthetic_raw"
    assert real_ml == results_root / "_workspace" / "ml_ready_real"
    assert brain_dir == results_root / "_workspace" / "costguard_brain"
    assert synthetic_raw.parent.exists()


def test_notify_short_circuits_when_disabled(monkeypatch):
    def _unexpected_lookup(cls, key: str, default: str = "") -> str:
        raise AssertionError(f"credential lookup should not happen when notifier is disabled: {key}")

    monkeypatch.setattr(cg.CredentialResolver, "get", classmethod(_unexpected_lookup))
    cg._set_runtime_controls(verbose_epochs=True, disable_notifier=True)
    cg._NOTIFIER_LAST_SENT.clear()

    cg.notify("BLOCK", 0.95, 1.25, summary="suppressed for ieee run")

    cg._set_runtime_controls(verbose_epochs=True, disable_notifier=False)


def test_train_lstm_persists_checkpoint_layout_and_resume_state(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cg, "load_task_b_tensors", _tiny_task_b_tensors)
    monkeypatch.setattr(cg, "_probe_lstm_batch_size", lambda cfg, xtr, ctr, hardware=None: 2)

    cfg = cg.LSTMConfig(
        epochs=1,
        batch_size=2,
        lr=1e-3,
        hidden_dim=8,
        num_layers=1,
        ctx_proj_dim=8,
        patience=3,
        seed=7,
    )
    _, hist_first, _, _, _, _, _, _ = cg.train_lstm(
        cfg,
        tmp_path,
        resume_epoch=False,
        domain_label="Synthetic",
        verbose_epochs=False,
    )

    assert hist_first["best_epoch"] >= 1
    assert (tmp_path / "checkpoints" / "lstm_ckpt.pt").exists()
    assert (tmp_path / "checkpoints" / "lstm_best.pt").exists()
    assert (tmp_path / "predictions" / "lstm_val_logits.npy").exists()
    assert (tmp_path / "predictions" / "lstm_test_logits.npy").exists()

    cfg_resume = cg.LSTMConfig(
        epochs=2,
        batch_size=2,
        lr=1e-3,
        hidden_dim=8,
        num_layers=1,
        ctx_proj_dim=8,
        patience=3,
        seed=7,
    )
    _, hist_resume, _, _, _, _, _, _ = cg.train_lstm(
        cfg_resume,
        tmp_path,
        resume_epoch=True,
        domain_label="Synthetic",
        verbose_epochs=False,
    )

    assert hist_resume["resume_loaded"] is True
    assert hist_resume["start_epoch"] == 2
    assert hist_resume["last_epoch"] >= 2
    assert hist_resume["adaptive"]["batch_size"] == 2
    assert hist_resume["adaptive"]["num_workers"] == 0


def test_train_lstm_auto_resumes_from_latest_checkpoint_without_flag(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cg, "load_task_b_tensors", _tiny_task_b_tensors)
    monkeypatch.setattr(cg, "_probe_lstm_batch_size", lambda cfg, xtr, ctr, hardware=None: 2)

    initial_cfg = cg.LSTMConfig(
        epochs=1,
        batch_size=2,
        lr=1e-3,
        hidden_dim=8,
        num_layers=1,
        ctx_proj_dim=8,
        patience=3,
        seed=11,
    )
    cg.train_lstm(
        initial_cfg,
        tmp_path,
        resume_epoch=False,
        domain_label="Synthetic",
        verbose_epochs=False,
    )

    resumed_cfg = cg.LSTMConfig(
        epochs=2,
        batch_size=2,
        lr=1e-3,
        hidden_dim=8,
        num_layers=1,
        ctx_proj_dim=8,
        patience=3,
        seed=11,
    )
    _, hist_resume, _, _, _, _, _, _ = cg.train_lstm(
        resumed_cfg,
        tmp_path,
        resume_epoch=False,
        domain_label="Synthetic",
        verbose_epochs=False,
    )

    assert hist_resume["resume_loaded"] is True
    assert hist_resume["start_epoch"] == 2
    assert hist_resume["last_epoch"] >= 2


def test_train_lstm_checkpoint_payload_is_full_state_and_retains_last_k(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cg, "load_task_b_tensors", _tiny_task_b_tensors)
    monkeypatch.setattr(cg, "_probe_lstm_batch_size", lambda cfg, xtr, ctr, hardware=None: 2)

    cfg = cg.LSTMConfig(
        epochs=3,
        batch_size=2,
        lr=1e-3,
        hidden_dim=8,
        num_layers=1,
        ctx_proj_dim=8,
        patience=3,
        checkpoint_keep_last_k=2,
        seed=13,
    )
    cg.train_lstm(
        cfg,
        tmp_path,
        resume_epoch=False,
        domain_label="Synthetic",
        verbose_epochs=False,
    )

    checkpoint_dir = tmp_path / "checkpoints"
    latest_payload = torch.load(checkpoint_dir / "lstm_ckpt.pt", map_location=cg.DEVICE, weights_only=False)
    required_keys = {
        "epoch",
        "epoch_completed",
        "checkpoint_type",
        "model",
        "optimizer",
        "scheduler",
        "best_f1",
        "best_epoch",
        "patience_ctr",
        "patience_state",
        "hist",
        "torch_rng_state",
        "numpy_rng_state",
        "python_rng_state",
    }

    assert required_keys.issubset(latest_payload.keys())
    assert latest_payload["checkpoint_type"] == "latest"
    assert latest_payload["epoch"] == 3
    assert latest_payload["epoch_completed"] is True
    assert latest_payload["patience_state"]["limit"] == 3
    assert len(list((checkpoint_dir / "lstm_latest_history").glob("*.pt"))) == 2
    assert len(list((checkpoint_dir / "lstm_best_history").glob("*.pt"))) <= 2


def test_derive_lstm_settings_uses_grad_accumulation_for_safe_batch(monkeypatch):
    monkeypatch.setattr(cg, "_probe_lstm_batch_size", lambda cfg, xtr, ctr, hardware=None: 16)

    x_train = torch.randn(32, cg.SEQ_LEN, cg.N_CHANNELS, dtype=torch.float32)
    c_train = torch.randn(32, cg.N_CTX, dtype=torch.float32)
    cfg = cg.LSTMConfig(batch_size=64)
    hardware = cg.HardwareProfile(
        ram_gb=8.0,
        ram_total_gb=16.0,
        vram_gb=3.0,
        vram_total_gb=4.0,
        cpu_cores=8,
    )

    settings = cg._derive_lstm_settings(cfg, x_train, c_train, hardware)

    assert settings.batch_size == 16
    assert settings.eval_batch_size == 32
    assert settings.grad_accum_steps == 4
    assert settings.oom_retry_limit == cg.DEFAULT_OOM_RETRY_LIMIT
    assert settings.monitor_every_epochs == cg.SAFE_MONITOR_EVERY_EPOCHS
    assert settings.num_workers == 0


def test_train_lstm_retries_after_simulated_oom(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cg, "load_task_b_tensors", _tiny_task_b_tensors)
    monkeypatch.setattr(cg, "_probe_lstm_batch_size", lambda cfg, xtr, ctr, hardware=None: 8)

    original_forward = cg.FocalLoss.forward
    state = {"raised": False}

    def flaky_forward(self, logits, targets):
        if not state["raised"]:
            state["raised"] = True
            raise RuntimeError("CUDA out of memory. simulated retry path")
        return original_forward(self, logits, targets)

    monkeypatch.setattr(cg.FocalLoss, "forward", flaky_forward)

    cfg = cg.LSTMConfig(
        epochs=1,
        batch_size=8,
        lr=1e-3,
        hidden_dim=8,
        num_layers=1,
        ctx_proj_dim=8,
        patience=3,
        seed=17,
    )
    _, hist, _, _, _, _, _, _ = cg.train_lstm(
        cfg,
        tmp_path,
        resume_epoch=False,
        domain_label="Synthetic",
        verbose_epochs=False,
    )

    oom_events = hist.get("oom_events", [])
    assert isinstance(oom_events, list) and len(oom_events) == 1
    assert oom_events[0]["epoch"] == 1
    assert hist.get("aborted_due_to_oom") is not True
    assert hist["adaptive"]["batch_size"] == 4
    assert hist["adaptive"]["grad_accum_steps"] == 2
    assert (tmp_path / "checkpoints" / "lstm_ckpt.pt").exists()


def test_epoch_status_line_includes_val_loss():
    line = cg._epoch_status_line(
        seed=42,
        domain_label="Synthetic",
        model_name="LSTM",
        epoch=3,
        total_epochs=10,
        train_loss=0.1234,
        val_loss=0.2345,
        val_f1=0.6789,
        best_f1=0.7001,
        patience_ctr=2,
        patience=10,
        lr=1e-3,
        status="checkpoint=best",
    )

    assert "train_loss=0.1234" in line
    assert "val_loss=0.2345" in line
    assert "val_f1=0.6789" in line
    assert "best=0.7001" in line


def test_assert_cached_inputs_ready_requires_ml_ready_cache(tmp_path: Path):
    with pytest.raises(SystemExit) as excinfo:
        cg._assert_cached_inputs_ready(
            raw_dir=tmp_path / "raw",
            ml_dir=tmp_path / "ml_ready",
            require_raw=False,
            require_ml_ready=True,
            context="--skip-preprocess",
        )

    assert "data-preparation command first" in str(excinfo.value)
    assert "ML-ready cache missing" in str(excinfo.value)


def test_assert_cached_inputs_ready_passes_when_required_cache_exists(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    ml_dir = tmp_path / "ml_ready"
    task_b = ml_dir / "task_B"
    raw_dir.mkdir(parents=True, exist_ok=True)
    task_b.mkdir(parents=True, exist_ok=True)

    telemetry = "run_id,stage_name,anomaly_window_active\n" + "\n".join(f"r{i},build,0" for i in range(500))
    graphs = "graph_id,src_stage,dst_stage,label_cost_anomalous\n" + "\n".join(f"g{i},0,1,0" for i in range(40))
    sequences = "run_id,label_budget_breach\n" + "\n".join(f"r{i},0" for i in range(40))
    (raw_dir / "pipeline_stage_telemetry.csv").write_text(telemetry, encoding="utf-8")
    (raw_dir / "pipeline_graphs.csv").write_text(graphs, encoding="utf-8")
    (raw_dir / "node_stats.csv").write_text("stage_name,anomaly_rate\nbuild,0\n" + ("#" * 300), encoding="utf-8")
    (raw_dir / "lstm_training_sequences.csv").write_text(sequences, encoding="utf-8")
    for name, shape in {
        "X_train.npy": (2, cg.SEQ_LEN, cg.N_CHANNELS),
        "X_val.npy": (1, cg.SEQ_LEN, cg.N_CHANNELS),
        "X_test.npy": (1, cg.SEQ_LEN, cg.N_CHANNELS),
        "X_ctx_train.npy": (2, cg.N_CTX),
        "y_train.npy": (2,),
        "y_val.npy": (1,),
        "y_test.npy": (1,),
    }.items():
        np.save(task_b / name, np.zeros(shape, dtype=np.float32))
    (task_b / "config.json").write_text('{"n_train": 2}', encoding="utf-8")

    cg._assert_cached_inputs_ready(
        raw_dir=raw_dir,
        ml_dir=ml_dir,
        require_raw=True,
        require_ml_ready=True,
        context="training-only",
    )


def test_format_log_event_renders_structured_tags():
    rendered = rt.format_log_event("Seed 42", "Synthetic", "GAT", "START", batch_size=64, path="C:/tmp/run 1")

    assert rendered.startswith("[Seed 42][Synthetic][GAT][START]")
    assert "batch_size=64" in rendered
    assert 'path="C:/tmp/run 1"' in rendered
