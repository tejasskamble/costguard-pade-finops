#!/usr/bin/env python3
"""Import IEEE post-run artifacts into PostgreSQL with dry-run-by-default safety."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

import asyncpg

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import settings  # noqa: E402
from postrun.import_service import build_postrun_snapshot, import_snapshot_to_db  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import CostGuard IEEE post-run artifacts into Postgres.",
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help="Override results root directory (defaults to POSTRUN_RESULTS_ROOT/results).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(settings.POSTRUN_IMPORT_CHUNK_SIZE or 100_000),
        help="CSV import chunk size for bounded processing.",
    )
    parser.add_argument(
        "--min-ensemble-f1",
        type=float,
        default=float(settings.POSTRUN_MIN_ENSEMBLE_F1),
        help="Inline quality gate threshold used during snapshot validation.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write to database (default is dry-run and read-only).",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Postgres DSN override (defaults to config DATABASE_URL).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for writing the output JSON report.",
    )
    return parser


def _emit(payload: Dict[str, Any], output_json: str | None) -> None:
    content = json.dumps(payload, indent=2, default=str)
    print(content)
    if output_json:
        output_path = Path(output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content + "\n", encoding="utf-8")


async def _run(args: argparse.Namespace) -> int:
    snapshot = build_postrun_snapshot(
        results_root_override=args.results_root,
        chunk_size=args.chunk_size,
        min_ensemble_f1=args.min_ensemble_f1,
    )
    if not args.apply:
        dry_run_payload = await import_snapshot_to_db(None, snapshot, dry_run=True)
        _emit(
            {
                "status": "DRY_RUN_ONLY",
                "snapshot_summary": snapshot.get("summary", {}),
                "quality_gate": snapshot.get("quality_gate", {}),
                "import_preview": dry_run_payload,
            },
            args.output_json,
        )
        return 0

    db_url = args.db_url or settings.DATABASE_URL
    conn = await asyncpg.connect(dsn=db_url, timeout=30)
    try:
        payload = await import_snapshot_to_db(conn, snapshot, dry_run=False)
    finally:
        await conn.close()

    _emit(payload, args.output_json)
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
