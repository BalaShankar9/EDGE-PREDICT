"""Fixture enrichment — merges odds, predictions, and team intel onto raw fixtures."""
import logging
from hashlib import md5

import pandas as pd

logger = logging.getLogger(__name__)


def enrich_fixtures(
    fixtures: list[dict],
    odds: list[dict] | None = None,
    predictions: list[dict] | None = None,
    team_intel: list[dict] | None = None,
) -> list[dict]:
    """Merge supplementary data onto raw fixture dicts.

    Uses MD5-keyed lookups by home|away|date or home|away fallback.

    Parameters
    ----------
    fixtures : raw fixture dicts with home_team, away_team, match_date
    odds : bookmaker odds dicts
    predictions : competitor prediction dicts
    team_intel : team news / injury / form dicts

    Returns
    -------
    list of enriched fixture dicts
    """
    odds_map: dict[str, dict] = {}
    pred_map: dict[str, dict] = {}
    intel_map: dict[str, dict] = {}

    def _key(home: str, away: str, date: str = "") -> str:
        raw = f"{home}|{away}|{date}".lower().strip()
        return md5(raw.encode()).hexdigest()

    # Index supplementary data
    if odds:
        for o in odds:
            k = _key(o.get("home_team", ""), o.get("away_team", ""), o.get("match_date", ""))
            odds_map[k] = o

    if predictions:
        for p in predictions:
            k = _key(p.get("home_team", ""), p.get("away_team", ""), p.get("match_date", ""))
            pred_map[k] = p

    if team_intel:
        for t in team_intel:
            k = _key(t.get("team", ""), "", "")
            intel_map[k] = t

    enriched = []
    for fixture in fixtures:
        f = fixture.copy()
        home = f.get("home_team", "")
        away = f.get("away_team", "")
        date = f.get("match_date", "")

        # Try exact match first, then without date
        k_exact = _key(home, away, date)
        k_fuzzy = _key(home, away)

        # Odds
        o = odds_map.get(k_exact) or odds_map.get(k_fuzzy)
        if o:
            f["best_odds_home"] = o.get("odds_home", 0)
            f["best_odds_draw"] = o.get("odds_draw", 0)
            f["best_odds_away"] = o.get("odds_away", 0)
            f["bookmaker"] = o.get("bookmaker", "unknown")

        # Predictions
        p = pred_map.get(k_exact) or pred_map.get(k_fuzzy)
        if p:
            f["meta_prediction"] = p.get("predicted_result", "")
            f["meta_home_prob"] = p.get("prob_home", 0)
            f["meta_agreement"] = p.get("agreement", 0)

        enriched.append(f)

    logger.info(
        f"Enriched {len(enriched)} fixtures: "
        f"{sum(1 for e in enriched if 'best_odds_home' in e)} with odds, "
        f"{sum(1 for e in enriched if 'meta_prediction' in e)} with predictions"
    )
    return enriched
