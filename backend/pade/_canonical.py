"""Lazy loader for the canonical CostGuard IEEE engine."""
from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


ROOT_ENGINE_PATH = Path(__file__).resolve().parents[2] / "CostGuard_PADE_FULL.py"


@lru_cache(maxsize=1)
def get_canonical_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("costguard_root_engine", ROOT_ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load canonical CostGuard engine from {ROOT_ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("costguard_root_engine", module)
    spec.loader.exec_module(module)
    return module


def canonical_attr(name: str):
    return getattr(get_canonical_module(), name)
