"""
Orchestrator — coordinates data collection, validation, and alerting.

Provides ``run_daily_collection()`` for scheduled runs and
``run_backfill()`` for historical data loading.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from sharpedge.collectors.football_data_uk import FootballDataUKCollector
from sharpedge.collectors.club_elo import ClubELOCollector
from sharpedge.collectors.understat import UnderstatCollector
from sharpedge.collectors.fbref import FBrefCollector
from sharpedge.collectors.forebet import ForebetCollector
from sharpedge.collectors.open_meteo import OpenMeteoCollector
from sharpedge.collectors.predictz import PredictZCollector
from sharpedge.collectors.windrawwin import WinDrawWinCollector
from sharpedge.collectors.footystats import FootyStatsCollector
from sharpedge.collectors.football_data_org import FootballDataOrgCollector
from sharpedge.validation.schema import validate_schema
from sharpedge.validation.statistical import validate_statistical
from sharpedge.alerts.telegram import send_alert_sync

logger = logging.getLogger(__name__)

# Default seasons for backfill
DEFAULT_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

# Map source names to validation record types
_SOURCE_TYPE_MAP: dict[str, str] = {
    "football_data_uk": "match",
    "club_elo": "elo",
    "understat": "xg",
    "fbref": "match",
    "forebet": "prediction",
    "predictz": "prediction",
    "windrawwin": "prediction",
    "footystats": "prediction",
    "football_data_org": "match",
    "open_meteo": "match",  # weather data doesn't have its own schema
}


@dataclass
class CollectionResult:
    """Outcome of a single collector run."""

    source: str
    rows: int
    status: str  # "healthy", "degraded", "down"
    errors: list[str] = field(default_factory=list)


def _infer_type(source_name: str) -> str:
    """Map a collector's source_name to a validation record type."""
    return _SOURCE_TYPE_MAP.get(source_name, "match")


def _validate_dataframe(
    df: pd.DataFrame, source_name: str
) -> CollectionResult:
    """Run schema + statistical validation and return a CollectionResult."""
    errors: list[str] = []
    status = "healthy"

    if df.empty:
        return CollectionResult(
            source=source_name, rows=0, status="down",
            errors=["Collector returned empty DataFrame"],
        )

    # Schema validation
    record_type = _infer_type(source_name)
    schema_result = validate_schema(df, record_type)
    if not schema_result.passed:
        errors.extend(schema_result.errors)
        status = "degraded"

    # Statistical validation
    stat_result = validate_statistical(df)
    if not stat_result.passed:
        errors.extend(stat_result.errors)
        status = "degraded"

    # Warnings don't downgrade status but are logged
    for w in schema_result.warnings + stat_result.warnings:
        logger.warning(f"[{source_name}] {w}")

    return CollectionResult(
        source=source_name, rows=len(df), status=status, errors=errors,
    )


def _run_collector(collector, **kwargs) -> CollectionResult:
    """Execute a single collector with validation and alerting."""
    source = collector.source_name
    try:
        df = collector.collect(**kwargs)
        result = _validate_dataframe(df, source)
    except Exception as exc:
        logger.error(f"[{source}] Unexpected error: {exc}")
        result = CollectionResult(
            source=source, rows=0, status="down", errors=[str(exc)],
        )

    if result.status == "down":
        try:
            send_alert_sync(
                f"Source <b>{source}</b> is DOWN\n"
                f"Errors: {', '.join(result.errors)}"
            )
        except Exception:
            logger.warning(f"[{source}] Failed to send Telegram alert")

    return result


def run_daily_collection() -> list[CollectionResult]:
    """Run all collectors for daily update (current season only).

    Collectors run in dependency order:
    1. Football-Data UK (match results + odds)
    2. ClubELO (current ratings)
    3. Understat (xG data)
    4. FBref (advanced stats)
    5. Forebet (predictions)
    6. PredictZ (predictions)
    7. WinDrawWin (predictions)
    8. FootyStats (stats + predictions)
    9. football-data.org (fixtures + standings)
    10. Open-Meteo (weather — requires match venues)

    Returns
    -------
    list[CollectionResult]
    """
    results: list[CollectionResult] = []

    collectors_and_kwargs = [
        (FootballDataUKCollector(), {}),
        (ClubELOCollector(), {}),
        (UnderstatCollector(), {}),
        (FBrefCollector(), {}),
        (ForebetCollector(), {}),
        (PredictZCollector(), {}),
        (WinDrawWinCollector(), {}),
        (FootyStatsCollector(), {}),
        (FootballDataOrgCollector(), {}),
        (OpenMeteoCollector(), {}),
    ]

    for collector, kwargs in collectors_and_kwargs:
        logger.info(f"--- Running {collector.source_name} ---")
        result = _run_collector(collector, **kwargs)
        results.append(result)
        logger.info(
            f"[{result.source}] {result.status} — {result.rows} rows, "
            f"{len(result.errors)} errors"
        )

    # Summary alert if any source is down
    down_sources = [r.source for r in results if r.status == "down"]
    if down_sources:
        try:
            send_alert_sync(
                f"Daily collection complete with failures.\n"
                f"DOWN: {', '.join(down_sources)}"
            )
        except Exception:
            logger.warning("Failed to send summary alert")

    return results


def run_backfill(
    seasons: Optional[list[str]] = None,
) -> list[CollectionResult]:
    """Backfill historical data for specified seasons.

    Runs all 10 collectors for each season. Collectors that don't support
    season-specific collection are run once per season anyway (they'll
    return their default data range).
    """
    if seasons is None:
        seasons = DEFAULT_SEASONS

    results: list[CollectionResult] = []

    season_aware = [
        FootballDataUKCollector,
        UnderstatCollector,
        FBrefCollector,
        ForebetCollector,
        PredictZCollector,
        WinDrawWinCollector,
        FootyStatsCollector,
    ]
    season_agnostic = [
        ClubELOCollector,
        FootballDataOrgCollector,
        OpenMeteoCollector,
    ]

    for season in seasons:
        logger.info(f"=== Backfilling season {season} ===")

        for collector_cls in season_aware:
            result = _run_collector(collector_cls(), season=season)
            results.append(result)

        for collector_cls in season_agnostic:
            result = _run_collector(collector_cls())
            results.append(result)

    return results
