"""
Freshness validation layer for SharpEdge data pipelines.

Ensures that collected data is recent enough to be useful for
predictions and betting decisions.
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from sharpedge.validation.schema import ValidationResult

logger = logging.getLogger(__name__)

# Candidate column names that may contain a collection timestamp
_TIMESTAMP_COLUMNS = ("collected_at", "timestamp", "fetched_at", "created_at")


def validate_freshness(
    df: pd.DataFrame, max_age_hours: int = 24
) -> ValidationResult:
    """Check whether the data in *df* is fresh enough.

    Looks for a timestamp column (see ``_TIMESTAMP_COLUMNS``) and verifies
    that the most recent entry is no older than *max_age_hours*.

    Parameters
    ----------
    df : pd.DataFrame
    max_age_hours : int
        Maximum acceptable age in hours (default 24).

    Returns
    -------
    ValidationResult
    """
    errors: list[str] = []
    warnings: list[str] = []

    if df.empty:
        warnings.append("DataFrame is empty; cannot assess freshness")
        return ValidationResult(passed=True, errors=errors, warnings=warnings)

    ts_col = _find_timestamp_column(df)
    if ts_col is None:
        warnings.append(
            "No timestamp column found; cannot assess freshness"
        )
        return ValidationResult(passed=True, errors=errors, warnings=warnings)

    try:
        timestamps = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    except Exception:
        warnings.append(
            f"Could not parse column '{ts_col}' as datetime; skipping freshness check"
        )
        return ValidationResult(passed=True, errors=errors, warnings=warnings)

    valid_ts = timestamps.dropna()
    if valid_ts.empty:
        warnings.append(
            f"All values in '{ts_col}' are null/unparseable; cannot assess freshness"
        )
        return ValidationResult(passed=True, errors=errors, warnings=warnings)

    most_recent = valid_ts.max()
    now = datetime.now(timezone.utc)
    age_hours = (now - most_recent).total_seconds() / 3600

    if age_hours > max_age_hours:
        errors.append(
            f"Data is {age_hours:.1f}h old (max allowed: {max_age_hours}h). "
            f"Most recent timestamp: {most_recent}"
        )
    elif age_hours > max_age_hours * 0.75:
        warnings.append(
            f"Data is {age_hours:.1f}h old — approaching staleness threshold "
            f"({max_age_hours}h)"
        )

    passed = len(errors) == 0
    return ValidationResult(passed=passed, errors=errors, warnings=warnings)


def _find_timestamp_column(df: pd.DataFrame) -> str | None:
    """Return the first recognised timestamp column in *df*."""
    for candidate in _TIMESTAMP_COLUMNS:
        if candidate in df.columns:
            return candidate
    return None
