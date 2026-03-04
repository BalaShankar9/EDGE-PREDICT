"""
Statistical validation layer for SharpEdge data pipelines.

Checks that numeric values fall within expected ranges and detects
duplicate match records.
"""

import logging

import pandas as pd

from sharpedge.validation.schema import ValidationResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Expected value ranges  (min_inclusive, max_inclusive)               #
# ------------------------------------------------------------------ #

RANGES: dict[str, tuple[float, float]] = {
    "home_xg": (0.0, 8.0),
    "away_xg": (0.0, 8.0),
    "home_goals": (0, 15),
    "away_goals": (0, 15),
    "elo": (800, 2200),
    "prob_home": (0.0, 1.0),
    "prob_draw": (0.0, 1.0),
    "prob_away": (0.0, 1.0),
    "odds_home": (1.01, 100.0),
    "odds_draw": (1.01, 100.0),
    "odds_away": (1.01, 100.0),
    "temperature_c": (-20.0, 50.0),
}


def validate_statistical(df: pd.DataFrame) -> ValidationResult:
    """Run statistical plausibility checks on *df*.

    Checks
    ------
    1. **Range checks** — for every column present in ``RANGES``, values
       outside the expected range trigger an error when >5 % of rows are
       affected, otherwise a warning.
    2. **Duplicate detection** — if ``home_team_id``, ``away_team_id``, and
       a date column are all present, flag exact duplicate matches.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    ValidationResult
    """
    errors: list[str] = []
    warnings: list[str] = []
    n_rows = len(df)

    if n_rows == 0:
        return ValidationResult(passed=True, errors=errors, warnings=warnings)

    # 1. Range checks -------------------------------------------------------
    for col, (lo, hi) in RANGES.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        out_of_range = ((series < lo) | (series > hi)) & series.notna()
        oor_count = int(out_of_range.sum())
        if oor_count == 0:
            continue
        oor_pct = oor_count / n_rows * 100
        msg = (
            f"Column '{col}': {oor_count} values ({oor_pct:.1f}%) "
            f"outside expected range [{lo}, {hi}]"
        )
        if oor_pct > 5:
            errors.append(msg)
        else:
            warnings.append(msg)

    # 2. Duplicate detection ------------------------------------------------
    date_col = _find_date_column(df)
    if (
        date_col
        and "home_team_id" in df.columns
        and "away_team_id" in df.columns
    ):
        subset = [date_col, "home_team_id", "away_team_id"]
        dup_count = int(df.duplicated(subset=subset, keep="first").sum())
        if dup_count > 0:
            errors.append(
                f"Found {dup_count} duplicate match(es) on "
                f"[{', '.join(subset)}]"
            )

    passed = len(errors) == 0
    return ValidationResult(passed=passed, errors=errors, warnings=warnings)


def _find_date_column(df: pd.DataFrame) -> str | None:
    """Return the first date-like column name found in *df*."""
    for candidate in ("match_date", "date", "Date"):
        if candidate in df.columns:
            return candidate
    return None
