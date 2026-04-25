"""
Seed-aware checkpoint and manifest resolution for CostGuard backend inference.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import suppress
from pathlib import Path
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

_SAFE_DEFAULTS: Dict[str, object] = {
    "lstm_temperature": 1.0,
    "gat_temperature": 1.0,
    "lstm_threshold": 0.5,
    "gat_threshold": 0.5,
    "ensemble_strategy": "weighted",
    "oof_meta_model_path": None,
}

_MIN_CHECKPOINT_BYTES = 1024 * 1024
_SEED_ENV = "COSTGUARD_ACTIVE_SEED"


class CheckpointResolver:
    """
    Resolve trained model artifacts with seed-aware precedence while preserving
    backward compatibility with legacy layout.
    """

    def __init__(self, project_root: Optional[Path] = None) -> None:
        if project_root is not None:
            self._root = Path(project_root).resolve()
        else:
            self._root = Path(__file__).resolve().parent.parent.parent
        self._bundled_ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
        self._watcher_thread: Optional[threading.Thread] = None
        self._stop_watcher = threading.Event()
        logger.info(
            "CheckpointResolver initialised: project_root=%s bundled=%s",
            self._root,
            self._bundled_ckpt_dir,
        )

    def get_manifest(self, data_mode: str = "synthetic", seed: Optional[int] = None) -> Optional[Dict]:
        """
        Resolve inference manifest for the requested data mode and optional seed.
        Search order:
          1) results/trials/seed_<seed>/<mode>
          2) results/seed_<seed>/<mode>
          3) results/<mode> (legacy)
        """
        seed_value = self._resolve_seed(seed)

        for mode_root in self._mode_roots(data_mode, seed_value):
            manifest_path = mode_root / "inference_manifest.json"
            if not manifest_path.exists():
                continue
            with suppress(Exception):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                logger.debug("Manifest loaded from %s", manifest_path)
                return manifest

        for mode_root in self._mode_roots(data_mode, seed_value):
            best_scores_path = mode_root / "best_scores.json"
            if not best_scores_path.exists():
                continue
            with suppress(Exception):
                scores = json.loads(best_scores_path.read_text(encoding="utf-8"))
                run_dir = self._latest_run_dir_in(mode_root)
                config: Dict[str, object] = {}
                if run_dir is not None:
                    run_config_path = run_dir / "run_config.json"
                    if run_config_path.exists():
                        with suppress(Exception):
                            config = json.loads(run_config_path.read_text(encoding="utf-8"))

                best_run = str(scores.get("best_run") or (run_dir.name if run_dir else "run_1"))
                ckpt_dir = mode_root / best_run / "checkpoints"
                manifest = {
                    "lstm_checkpoint": str(ckpt_dir / "lstm_best.pt"),
                    "gat_checkpoint": str(ckpt_dir / "gat_best.pt"),
                    "lstm_temperature": config.get("lstm_temperature", 1.0),
                    "gat_temperature": config.get("gat_temperature", 1.0),
                    "lstm_threshold": config.get("lstm_threshold", 0.5),
                    "gat_threshold": config.get("gat_threshold", 0.5),
                    "ensemble_strategy": scores.get("ensemble_strategy", "weighted"),
                    "oof_meta_model_path": config.get("oof_meta_model_path"),
                    "data_mode": data_mode,
                    "run_id": best_run,
                    "trained_at": scores.get("trained_at"),
                    "lstm_f1_at_opt": scores.get("lstm_f1", 0.0),
                    "gat_f1_at_opt": scores.get("gat_f1", 0.0),
                    "ensemble_f1_at_opt": scores.get("ensemble_f1", 0.0),
                    "_source": f"best_scores_fallback:{mode_root}",
                    "_seed_context": seed_value,
                }
                logger.info(
                    "Manifest synthesised from %s for mode=%s seed=%s",
                    best_scores_path,
                    data_mode,
                    seed_value if seed_value is not None else "none",
                )
                return manifest

        logger.warning(
            "No manifest or best_scores.json found for mode=%s seed=%s",
            data_mode,
            seed_value if seed_value is not None else "none",
        )
        return None

    def get_lstm_checkpoint(self, data_mode: str = "synthetic", seed: Optional[int] = None) -> Optional[Path]:
        return self._resolve_checkpoint(
            data_mode=data_mode,
            manifest_key="lstm_checkpoint",
            filename_in_run="lstm_best.pt",
            bundled_name="lstm.pt",
            seed=seed,
        )

    def get_gat_checkpoint(self, data_mode: str = "synthetic", seed: Optional[int] = None) -> Optional[Path]:
        return self._resolve_checkpoint(
            data_mode=data_mode,
            manifest_key="gat_checkpoint",
            filename_in_run="gat_best.pt",
            bundled_name="gat.pt",
            seed=seed,
        )

    def get_calibration_params(self, data_mode: str = "synthetic", seed: Optional[int] = None) -> Dict:
        manifest = self.get_manifest(data_mode, seed=seed)
        if manifest is None:
            return dict(_SAFE_DEFAULTS)

        params = dict(_SAFE_DEFAULTS)
        for key in _SAFE_DEFAULTS:
            if key in manifest:
                params[key] = manifest[key]

        if params.get("oof_meta_model_path"):
            oof_path = Path(str(params["oof_meta_model_path"]))
            if not oof_path.is_absolute():
                oof_path = self._root / oof_path
            params["oof_meta_model_path"] = str(oof_path) if oof_path.exists() else None
        return params

    def get_latest_run_dir(self, data_mode: str = "synthetic", seed: Optional[int] = None) -> Optional[Path]:
        seed_value = self._resolve_seed(seed)
        for mode_root in self._mode_roots(data_mode, seed_value):
            run_dir = self._latest_run_dir_in(mode_root)
            if run_dir is not None:
                logger.debug(
                    "Latest run dir for mode=%s seed=%s resolved to %s",
                    data_mode,
                    seed_value if seed_value is not None else "none",
                    run_dir,
                )
                return run_dir
        return None

    def watch_for_new_checkpoints(
        self,
        callback: Callable,
        poll_interval: int = 30,
    ) -> None:
        if self._watcher_thread is not None and self._watcher_thread.is_alive():
            self._stop_watcher.set()
            self._watcher_thread.join(timeout=5)

        self._stop_watcher.clear()

        def _watch_loop() -> None:
            last_mtimes: Dict[str, float] = {}
            while not self._stop_watcher.is_set():
                for mode in ("synthetic", "real", "bitbrains"):
                    manifest_path = self._root / "results" / mode / "inference_manifest.json"
                    if not manifest_path.exists():
                        continue
                    mtime = manifest_path.stat().st_mtime
                    if last_mtimes.get(mode, 0.0) < mtime:
                        last_mtimes[mode] = mtime
                        with suppress(Exception):
                            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                            callback(manifest)
                self._stop_watcher.wait(timeout=poll_interval)

        self._watcher_thread = threading.Thread(
            target=_watch_loop,
            name="checkpoint-watcher",
            daemon=True,
        )
        self._watcher_thread.start()

    def stop_watcher(self) -> None:
        self._stop_watcher.set()

    def _resolve_checkpoint(
        self,
        data_mode: str,
        manifest_key: str,
        filename_in_run: str,
        bundled_name: str,
        seed: Optional[int] = None,
    ) -> Optional[Path]:
        manifest = self.get_manifest(data_mode, seed=seed)
        if manifest and manifest.get(manifest_key):
            candidate = Path(str(manifest[manifest_key]))
            if not candidate.is_absolute():
                candidate = self._root / candidate
            if self._is_valid_checkpoint(candidate):
                return candidate
            logger.warning("Manifest points to invalid checkpoint: %s", candidate)

        run_dir = self.get_latest_run_dir(data_mode, seed=seed)
        if run_dir is not None:
            candidate = run_dir / "checkpoints" / filename_in_run
            if self._is_valid_checkpoint(candidate):
                return candidate

        for bundled in (self._bundled_ckpt_dir / filename_in_run, self._bundled_ckpt_dir / bundled_name):
            if self._is_valid_checkpoint(bundled):
                logger.warning("Falling back to bundled checkpoint: %s", bundled)
                return bundled
        return None

    def _mode_roots(self, data_mode: str, seed: Optional[int]) -> list[Path]:
        roots: list[Path] = []
        if seed is not None:
            roots.append(self._root / "results" / "trials" / f"seed_{seed}" / data_mode)
            roots.append(self._root / "results" / f"seed_{seed}" / data_mode)
        roots.append(self._root / "results" / data_mode)

        deduped: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            if root not in seen:
                seen.add(root)
                deduped.append(root)
        return deduped

    @staticmethod
    def _seed_from_value(seed_value: object) -> Optional[int]:
        if seed_value is None:
            return None
        if isinstance(seed_value, int):
            return seed_value
        try:
            text = str(seed_value).strip()
            if not text:
                return None
            parsed = int(text)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    def _resolve_seed(self, seed: Optional[int]) -> Optional[int]:
        resolved = self._seed_from_value(seed)
        if resolved is not None:
            return resolved
        return self._seed_from_value(os.getenv(_SEED_ENV))

    @staticmethod
    def _latest_run_dir_in(mode_root: Path) -> Optional[Path]:
        if not mode_root.exists():
            return None
        run_dirs: list[tuple[int, Path]] = []
        with suppress(Exception):
            for entry in mode_root.iterdir():
                if entry.is_dir() and entry.name.startswith("run_"):
                    with suppress(ValueError):
                        run_dirs.append((int(entry.name.split("_", 1)[1]), entry))
        if not run_dirs:
            return None
        run_dirs.sort(key=lambda item: item[0])
        return run_dirs[-1][1]

    @staticmethod
    def _is_valid_checkpoint(path: Path) -> bool:
        if not isinstance(path, Path):
            return False
        try:
            return path.exists() and path.stat().st_size >= _MIN_CHECKPOINT_BYTES
        except Exception:
            return False


_resolver: Optional[CheckpointResolver] = None


def get_resolver(project_root: Optional[Path] = None) -> CheckpointResolver:
    global _resolver
    if _resolver is None:
        _resolver = CheckpointResolver(project_root=project_root)
    return _resolver
