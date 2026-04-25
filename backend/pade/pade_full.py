"""Compatibility shim over the canonical CostGuard IEEE engine."""
from __future__ import annotations

try:
    from ._canonical import canonical_attr, get_canonical_module
except ImportError:
    from _canonical import canonical_attr, get_canonical_module


def __getattr__(name: str):
    return canonical_attr(name)


def __dir__():
    return sorted(set(dir(get_canonical_module())))
