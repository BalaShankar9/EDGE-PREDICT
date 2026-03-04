#!/usr/bin/env python3
"""
Health check script — verify each collector can reach its data source.

Usage
-----
    python scripts/health_check.py
"""

import logging
import sys
from pathlib import Path

# Ensure src is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sharpedge.collectors.football_data_uk import FootballDataUKCollector  # noqa: E402
from sharpedge.collectors.club_elo import ClubELOCollector  # noqa: E402
from sharpedge.collectors.understat import UnderstatCollector  # noqa: E402
from sharpedge.collectors.fbref import FBrefCollector  # noqa: E402
from sharpedge.collectors.forebet import ForebetCollector  # noqa: E402
from sharpedge.collectors.open_meteo import OpenMeteoCollector  # noqa: E402


_STATUS_ICONS = {
    "ok": "\u2705",
    "down": "\u274c",
}


def _check_collector(collector, **kwargs) -> tuple[str, str, int]:
    """Test a single collector with a minimal fetch.

    Returns (source_name, status, row_count).
    """
    try:
        df = collector.collect(**kwargs)
        if df.empty:
            return collector.source_name, "down", 0
        return collector.source_name, "ok", len(df)
    except Exception as exc:
        logging.error(f"[{collector.source_name}] Health check failed: {exc}")
        return collector.source_name, "down", 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("\nSharpEdge Health Check\n" + "=" * 40)

    checks = [
        (FootballDataUKCollector(), {"league": "E0", "season": "2024-25"}),
        (ClubELOCollector(), {}),
        (UnderstatCollector(), {"league": "EPL", "season": "2024"}),
        (FBrefCollector(), {"league": "ENG-Premier League", "season": "2024-2025"}),
        (ForebetCollector(), {}),
        (OpenMeteoCollector(), {"venue": "London", "date": "2024-01-01"}),
    ]

    all_ok = True
    for collector, kwargs in checks:
        name, status, rows = _check_collector(collector, **kwargs)
        icon = _STATUS_ICONS.get(status, "?")
        print(f"  {icon}  {name:<20s}  {rows:>5d} rows  [{status}]")
        if status != "ok":
            all_ok = False

    print("=" * 40)
    if all_ok:
        print("All sources healthy.\n")
    else:
        print("Some sources are down!\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
