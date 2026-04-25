#!/usr/bin/env python3
"""Compatibility wrapper for the unified CostGuard analytics engine."""
from __future__ import annotations

from costguard_analytics import aggregate_cli


if __name__ == "__main__":
    raise SystemExit(aggregate_cli())
