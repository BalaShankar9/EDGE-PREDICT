#!/usr/bin/env python3
"""
Backfill script — load historical data for specified seasons.

Usage
-----
    python scripts/backfill.py
    python scripts/backfill.py --seasons 2020-21 2021-22
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sharpedge.orchestrator import run_backfill, DEFAULT_SEASONS  # noqa: E402


_STATUS_ICONS = {
    "healthy": "\u2705",   # checkmark
    "degraded": "\u26a0\ufe0f",  # warning
    "down": "\u274c",      # X
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill historical data for SharpEdge.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help=(
            "Seasons to backfill (e.g. 2020-21 2021-22). "
            f"Defaults to: {', '.join(DEFAULT_SEASONS)}"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    seasons = args.seasons
    print(f"\nSharpEdge Backfill — seasons: {seasons or DEFAULT_SEASONS}\n")

    results = run_backfill(seasons=seasons)

    # Print summary
    print("\n" + "=" * 50)
    print("BACKFILL SUMMARY")
    print("=" * 50)
    for r in results:
        icon = _STATUS_ICONS.get(r.status, "?")
        print(f"  {icon}  {r.source:<20s}  {r.rows:>5d} rows  [{r.status}]")
        for err in r.errors:
            print(f"       -> {err}")
    print("=" * 50)

    # Exit with error code if any source is down
    if any(r.status == "down" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
