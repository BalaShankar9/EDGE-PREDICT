"""
Schema validation layer for SharpEdge data pipelines.

Validates DataFrames against predefined schemas — checking required columns,
null ratios, and expected data types.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Outcome of a validation check."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ #
#  Schema definitions per record type                                 #
# ------------------------------------------------------------------ #

SCHEMAS: dict[str, dict] = {
    "match": {
        "required": [
            "match_date",
            "home_team_id",
            "away_team_id",
            "league",
            "season",
        ],
        "numeric": ["home_goals", "away_goals"],
    },
    "xg": {
        "required": [
            "date",
            "home_team_id",
            "away_team_id",
            "home_xg",
            "away_xg",
        ],
        "numeric": ["home_xg", "away_xg"],
    },
    "odds": {
        "required": ["match_date", "home_team_id", "away_team_id"],
        "numeric": [],
    },
    "prediction": {
        "required": ["home_team", "away_team", "source"],
        "numeric": ["prob_home", "prob_draw", "prob_away"],
    },
    "elo": {
        "required": ["club", "elo"],
        "numeric": ["elo"],
    },
}


def validate_schema(df: pd.DataFrame, record_type: str) -> ValidationResult:
    """Validate a DataFrame against the schema for *record_type*.

    Checks:
      1. Required columns exist.
      2. Null ratio — error if >10 %, warning if >0 %.
      3. Numeric columns contain numeric-compatible data.

    Parameters
    ----------
    df : pd.DataFrame
        Data to validate.
    record_type : str
        Key into ``SCHEMAS`` (e.g. ``"match"``, ``"xg"``).

    Returns
    -------
    ValidationResult
    """
    errors: list[str] = []
    warnings: list[str] = []

    if record_type not in SCHEMAS:
        warnings.append(
            f"No schema defined for record type '{record_type}'; skipping validation"
        )
        return ValidationResult(passed=True, errors=errors, warnings=warnings)

    schema = SCHEMAS[record_type]
    required: list[str] = schema.get("required", [])
    numeric: list[str] = schema.get("numeric", [])

    # 1. Required columns --------------------------------------------------
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        # Cannot continue further checks if columns are missing
        return ValidationResult(passed=False, errors=errors, warnings=warnings)

    # 2. Null checks --------------------------------------------------------
    n_rows = len(df)
    if n_rows > 0:
        for col in required:
            null_count = int(df[col].isna().sum())
            if null_count == 0:
                continue
            null_pct = null_count / n_rows * 100
            if null_pct > 10:
                errors.append(
                    f"Column '{col}' has {null_pct:.1f}% nulls (>{10}% threshold)"
                )
            else:
                warnings.append(
                    f"Column '{col}' has {null_pct:.1f}% nulls"
                )

    # 3. Numeric type checks ------------------------------------------------
    for col in numeric:
        if col not in df.columns:
            continue  # optional numeric column not present
        if not np.issubdtype(df[col].dtype, np.number):
            # Try to coerce — if it fails, flag as error
            try:
                pd.to_numeric(df[col], errors="raise")
            except (ValueError, TypeError):
                errors.append(
                    f"Column '{col}' expected numeric but got {df[col].dtype}"
                )

    passed = len(errors) == 0
    return ValidationResult(passed=passed, errors=errors, warnings=warnings)
