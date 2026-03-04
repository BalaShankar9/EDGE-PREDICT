"""
Cross-source validation layer for SharpEdge data pipelines.

Compares data from two independent sources to detect discrepancies
in scores, xG values, and team identifiers.
"""

import logging

import pandas as pd

from sharpedge.validation.schema import ValidationResult

logger = logging.getLogger(__name__)

# Tolerance for floating-point xG comparisons
_XG_TOLERANCE = 0.5


def validate_cross_source(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    source1_name: str,
    source2_name: str,
) -> ValidationResult:
    """Cross-validate data between two sources.

    Checks
    ------
    1. **Score agreement** — if both DataFrames contain ``home_goals`` and
       ``away_goals``, matched rows must agree on the scoreline.
    2. **xG agreement** — if both contain ``home_xg`` / ``away_xg``, values
       must be within ``_XG_TOLERANCE``.
    3. **Team-ID consistency** — ``home_team_id`` and ``away_team_id``
       should resolve to the same values across sources.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames from the two sources.
    source1_name, source2_name : str
        Human-readable source labels for error messages.

    Returns
    -------
    ValidationResult
    """
    errors: list[str] = []
    warnings: list[str] = []

    if df1.empty or df2.empty:
        warnings.append("One or both DataFrames are empty; skipping cross-source checks")
        return ValidationResult(passed=True, errors=errors, warnings=warnings)

    # Attempt to merge on common key columns
    merge_keys = _find_merge_keys(df1, df2)
    if not merge_keys:
        warnings.append(
            "No common key columns found for cross-source merge; skipping"
        )
        return ValidationResult(passed=True, errors=errors, warnings=warnings)

    merged = pd.merge(
        df1, df2, on=merge_keys, suffixes=("_src1", "_src2"), how="inner"
    )

    if merged.empty:
        warnings.append(
            f"No matching rows between {source1_name} and {source2_name}"
        )
        return ValidationResult(passed=True, errors=errors, warnings=warnings)

    # 1. Score agreement ----------------------------------------------------
    for goal_col in ("home_goals", "away_goals"):
        c1, c2 = f"{goal_col}_src1", f"{goal_col}_src2"
        if c1 in merged.columns and c2 in merged.columns:
            mismatches = (merged[c1] != merged[c2]).sum()
            if mismatches > 0:
                errors.append(
                    f"Score mismatch on '{goal_col}': {mismatches} rows differ "
                    f"between {source1_name} and {source2_name}"
                )

    # 2. xG agreement ------------------------------------------------------
    for xg_col in ("home_xg", "away_xg"):
        c1, c2 = f"{xg_col}_src1", f"{xg_col}_src2"
        if c1 in merged.columns and c2 in merged.columns:
            diff = (merged[c1] - merged[c2]).abs()
            big_diff = (diff > _XG_TOLERANCE).sum()
            if big_diff > 0:
                errors.append(
                    f"xG mismatch on '{xg_col}': {big_diff} rows differ by "
                    f">{_XG_TOLERANCE} between {source1_name} and {source2_name}"
                )

    # 3. Team-ID consistency ------------------------------------------------
    for tid_col in ("home_team_id", "away_team_id"):
        c1, c2 = f"{tid_col}_src1", f"{tid_col}_src2"
        if c1 in merged.columns and c2 in merged.columns:
            mismatches = (merged[c1] != merged[c2]).sum()
            if mismatches > 0:
                warnings.append(
                    f"Team ID mismatch on '{tid_col}': {mismatches} rows differ "
                    f"between {source1_name} and {source2_name}"
                )

    passed = len(errors) == 0
    return ValidationResult(passed=passed, errors=errors, warnings=warnings)


def _find_merge_keys(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> list[str]:
    """Identify common key columns suitable for merging."""
    candidates = [
        ["match_date", "home_team_id", "away_team_id"],
        ["date", "home_team_id", "away_team_id"],
    ]
    for keys in candidates:
        if all(k in df1.columns and k in df2.columns for k in keys):
            return keys
    # Fallback: any shared columns that look like keys
    shared = [
        c
        for c in df1.columns
        if c in df2.columns and c.endswith(("_id", "_date", "date"))
    ]
    return shared if shared else []
