"""
Persistence layer — writes collector DataFrames into PostgreSQL.

Each ``ingest_*`` function maps a collector's DataFrame columns to SQLAlchemy
models and performs idempotent upserts (safe to re-run).

Usage from the orchestrator::

    from sharpedge.db.ingest import ingest_dataframe
    ingest_dataframe(df, source_name="football_data_uk")
"""

import logging
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from sharpedge.db.engine import get_session
from sharpedge.db.models import (
    CompetitorPrediction,
    EloRating,
    League,
    Match,
    MatchOdds,
    MatchStats,
    MatchXG,
    RawStagingRecord,
    Season,
    Team,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers: get-or-create lookups with in-memory caching per call
# ---------------------------------------------------------------------------

def _get_or_create_team(session: Session, name: str, _cache: dict) -> int:
    """Return team ID, creating if needed. Uses _cache to avoid repeated queries."""
    if name in _cache:
        return _cache[name]
    team = session.query(Team).filter(Team.canonical_name == name).first()
    if not team:
        team = Team(canonical_name=name)
        session.add(team)
        session.flush()
    _cache[name] = team.id
    return team.id


def _get_or_create_league(session: Session, name: str, _cache: dict) -> int:
    if name in _cache:
        return _cache[name]
    league = session.query(League).filter(League.name == name).first()
    if not league:
        league = League(name=name)
        session.add(league)
        session.flush()
    _cache[name] = league.id
    return league.id


def _get_or_create_season(
    session: Session, league_id: int, label: str, _cache: dict
) -> int:
    key = (league_id, label)
    if key in _cache:
        return _cache[key]
    season = (
        session.query(Season)
        .filter(Season.league_id == league_id, Season.label == label)
        .first()
    )
    if not season:
        season = Season(league_id=league_id, label=label)
        session.add(season)
        session.flush()
    _cache[key] = season.id
    return season.id


def _parse_date(val) -> datetime:
    """Parse date from various formats football-data uses."""
    if isinstance(val, (datetime,)):
        return val.date() if hasattr(val, "date") else val
    if isinstance(val, pd.Timestamp):
        return val.date()
    s = str(val).strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {val!r}")


def _safe_int(val):
    """Convert to int or return None."""
    if pd.isna(val):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_float(val):
    """Convert to float or return None."""
    if pd.isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _row_to_dict(row: pd.Series) -> dict:
    """Convert a DataFrame row to a plain JSON-serialisable dict."""
    result = {}
    for k, v in row.items():
        if pd.isna(v) if not isinstance(v, (list, dict)) else False:
            result[k] = None
        elif isinstance(v, pd.Timestamp):
            result[k] = v.isoformat()
        elif hasattr(v, "item"):
            # numpy scalar → python native
            result[k] = v.item()
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Football-Data UK: matches, stats, odds
# ---------------------------------------------------------------------------

def ingest_football_data_uk(df: pd.DataFrame, session: Session | None = None) -> int:
    """Ingest Football-Data UK CSV data → teams, leagues, seasons, matches, stats, odds."""
    if df.empty:
        return 0

    own_session = session is None
    if own_session:
        session = get_session()

    team_cache: dict = {}
    league_cache: dict = {}
    season_cache: dict = {}
    inserted = 0

    try:
        for _, row in df.iterrows():
            # Skip rows without essential data
            home_name = str(row.get("HomeTeam", "")).strip()
            away_name = str(row.get("AwayTeam", "")).strip()
            if not home_name or not away_name or home_name == "nan":
                continue

            # Parse date
            try:
                match_date = _parse_date(row["Date"])
            except (ValueError, KeyError):
                continue

            # Get or create teams
            home_team_id = _get_or_create_team(session, home_name, team_cache)
            away_team_id = _get_or_create_team(session, away_name, team_cache)

            # Get or create league + season
            league_name = str(row.get("league", "Unknown"))
            season_label = str(row.get("season", "Unknown"))
            league_id = _get_or_create_league(session, league_name, league_cache)
            season_id = _get_or_create_season(
                session, league_id, season_label, season_cache
            )

            # Upsert match (unique on season_id, match_date, home_team_id, away_team_id)
            existing = (
                session.query(Match)
                .filter(
                    Match.season_id == season_id,
                    Match.match_date == match_date,
                    Match.home_team_id == home_team_id,
                    Match.away_team_id == away_team_id,
                )
                .first()
            )
            if existing:
                match = existing
                # Update scores if they changed
                match.home_goals = _safe_int(row.get("FTHG"))
                match.away_goals = _safe_int(row.get("FTAG"))
                match.result = str(row.get("FTR", "")).strip() or None
            else:
                match = Match(
                    season_id=season_id,
                    match_date=match_date,
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    home_goals=_safe_int(row.get("FTHG")),
                    away_goals=_safe_int(row.get("FTAG")),
                    home_goals_ht=_safe_int(row.get("HTHG")),
                    away_goals_ht=_safe_int(row.get("HTAG")),
                    result=str(row.get("FTR", "")).strip() or None,
                    referee=str(row.get("Referee", "")).strip() or None,
                    status="played",
                )
                session.add(match)
                session.flush()
                inserted += 1

            # Upsert match stats (one per match)
            if not match.stats and any(
                pd.notna(row.get(c)) for c in ["HS", "AS", "HST", "AST"]
            ):
                stats = MatchStats(
                    match_id=match.id,
                    source="football_data_uk",
                    home_shots=_safe_int(row.get("HS")),
                    away_shots=_safe_int(row.get("AS")),
                    home_shots_on_target=_safe_int(row.get("HST")),
                    away_shots_on_target=_safe_int(row.get("AST")),
                    home_fouls=_safe_int(row.get("HF")),
                    away_fouls=_safe_int(row.get("AF")),
                    home_corners=_safe_int(row.get("HC")),
                    away_corners=_safe_int(row.get("AC")),
                    home_yellow_cards=_safe_int(row.get("HY")),
                    away_yellow_cards=_safe_int(row.get("AY")),
                    home_red_cards=_safe_int(row.get("HR")),
                    away_red_cards=_safe_int(row.get("AR")),
                )
                session.add(stats)

            # Upsert odds — one row per bookmaker
            _BOOKMAKERS = {
                "Bet365": ("B365H", "B365D", "B365A"),
                "Pinnacle": ("PSH", "PSD", "PSA"),
                "WilliamHill": ("WHH", "WHD", "WHA"),
                "MarketMax": ("MaxH", "MaxD", "MaxA"),
                "MarketAvg": ("AvgH", "AvgD", "AvgA"),
            }
            for bookie, (h_col, d_col, a_col) in _BOOKMAKERS.items():
                odds_h = _safe_float(row.get(h_col))
                odds_d = _safe_float(row.get(d_col))
                odds_a = _safe_float(row.get(a_col))
                if odds_h is None and odds_d is None and odds_a is None:
                    continue

                existing_odds = (
                    session.query(MatchOdds)
                    .filter(
                        MatchOdds.match_id == match.id,
                        MatchOdds.bookmaker == bookie,
                        MatchOdds.market == "1X2",
                        MatchOdds.odds_type == "pre-match",
                    )
                    .first()
                )
                if not existing_odds:
                    session.add(
                        MatchOdds(
                            match_id=match.id,
                            bookmaker=bookie,
                            market="1X2",
                            odds_type="pre-match",
                            odds_home=odds_h,
                            odds_draw=odds_d,
                            odds_away=odds_a,
                        )
                    )

        session.commit()
        logger.info(f"[football_data_uk] Ingested {inserted} new matches")
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()

    return inserted


# ---------------------------------------------------------------------------
# Club ELO: elo_ratings
# ---------------------------------------------------------------------------

def ingest_club_elo(df: pd.DataFrame, session: Session | None = None) -> int:
    """Ingest ClubELO data → teams + elo_ratings."""
    if df.empty:
        return 0

    own_session = session is None
    if own_session:
        session = get_session()

    team_cache: dict = {}
    inserted = 0

    try:
        for _, row in df.iterrows():
            # Team name from Club or club column
            club = str(row.get("Club", row.get("club", ""))).strip()
            if not club or club == "nan":
                continue

            team_id = _get_or_create_team(session, club, team_cache)

            # Date from snapshot_date or From column
            date_str = row.get("snapshot_date", row.get("From", ""))
            try:
                rating_date = _parse_date(date_str)
            except (ValueError, KeyError):
                continue

            elo_val = _safe_float(row.get("Elo", row.get("elo", None)))
            if elo_val is None:
                continue

            rank_val = _safe_int(row.get("Rank", row.get("rank", None)))

            # Upsert (unique on team_id, rating_date, source)
            existing = (
                session.query(EloRating)
                .filter(
                    EloRating.team_id == team_id,
                    EloRating.rating_date == rating_date,
                    EloRating.source == "club_elo",
                )
                .first()
            )
            if not existing:
                session.add(
                    EloRating(
                        team_id=team_id,
                        rating_date=rating_date,
                        elo=elo_val,
                        rank=rank_val,
                        source="club_elo",
                    )
                )
                inserted += 1

        session.commit()
        logger.info(f"[club_elo] Ingested {inserted} new ELO ratings")
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()

    return inserted


# ---------------------------------------------------------------------------
# Understat: match xG
# ---------------------------------------------------------------------------

def ingest_understat(df: pd.DataFrame, session: Session | None = None) -> int:
    """Ingest Understat xG data → match_xg (requires matches to exist)."""
    if df.empty:
        return 0

    own_session = session is None
    if own_session:
        session = get_session()

    team_cache: dict = {}
    inserted = 0

    try:
        for _, row in df.iterrows():
            home_name = str(row.get("home_team", "")).strip()
            away_name = str(row.get("away_team", "")).strip()
            if not home_name or not away_name:
                continue

            try:
                match_date = _parse_date(row.get("date", ""))
            except (ValueError, KeyError):
                continue

            # Look up team IDs
            home_team_id = _get_or_create_team(session, home_name, team_cache)
            away_team_id = _get_or_create_team(session, away_name, team_cache)

            # Find the match
            match = (
                session.query(Match)
                .filter(
                    Match.match_date == match_date,
                    Match.home_team_id == home_team_id,
                    Match.away_team_id == away_team_id,
                )
                .first()
            )
            if not match:
                continue

            # Upsert xG (unique on match_id + source)
            existing = (
                session.query(MatchXG)
                .filter(
                    MatchXG.match_id == match.id,
                    MatchXG.source == "understat",
                )
                .first()
            )
            if not existing:
                session.add(
                    MatchXG(
                        match_id=match.id,
                        source="understat",
                        home_xg=_safe_float(row.get("home_xg")),
                        away_xg=_safe_float(row.get("away_xg")),
                    )
                )
                inserted += 1

        session.commit()
        logger.info(f"[understat] Ingested {inserted} new xG records")
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()

    return inserted


# ---------------------------------------------------------------------------
# Competitor predictions: forebet, predictz, windrawwin, footystats
# ---------------------------------------------------------------------------

_RESULT_MAP = {"1": "H", "X": "D", "2": "A", "H": "H", "D": "D", "A": "A"}


def ingest_predictions(
    df: pd.DataFrame, source_name: str, session: Session | None = None
) -> int:
    """Ingest prediction data → competitor_predictions (requires matches to exist)."""
    if df.empty:
        return 0

    own_session = session is None
    if own_session:
        session = get_session()

    team_cache: dict = {}
    inserted = 0

    try:
        for _, row in df.iterrows():
            home_name = str(row.get("home_team", "")).strip()
            away_name = str(row.get("away_team", "")).strip()
            if not home_name or not away_name:
                continue

            # Try to parse date from scraped_date
            date_val = row.get("scraped_date", row.get("date", ""))
            try:
                match_date = _parse_date(date_val)
            except (ValueError, KeyError):
                continue

            home_team_id = _get_or_create_team(session, home_name, team_cache)
            away_team_id = _get_or_create_team(session, away_name, team_cache)

            # Find the match
            match = (
                session.query(Match)
                .filter(
                    Match.match_date == match_date,
                    Match.home_team_id == home_team_id,
                    Match.away_team_id == away_team_id,
                )
                .first()
            )
            if not match:
                continue

            # Normalise predicted result
            raw_result = str(row.get("predicted_result", "")).strip()
            predicted_result = _RESULT_MAP.get(raw_result, raw_result[:1] if raw_result else None)

            # Normalise probabilities (some sources use 0-100, we store 0-1)
            prob_h = _safe_float(row.get("prob_home"))
            prob_d = _safe_float(row.get("prob_draw"))
            prob_a = _safe_float(row.get("prob_away"))
            if prob_h is not None and prob_h > 1:
                prob_h /= 100
            if prob_d is not None and prob_d > 1:
                prob_d /= 100
            if prob_a is not None and prob_a > 1:
                prob_a /= 100

            # Upsert (unique on match_id + source)
            existing = (
                session.query(CompetitorPrediction)
                .filter(
                    CompetitorPrediction.match_id == match.id,
                    CompetitorPrediction.source == source_name,
                )
                .first()
            )
            if not existing:
                session.add(
                    CompetitorPrediction(
                        match_id=match.id,
                        source=source_name,
                        predicted_result=predicted_result,
                        prob_home=prob_h,
                        prob_draw=prob_d,
                        prob_away=prob_a,
                        predicted_score_home=_safe_int(row.get("predicted_score_home")),
                        predicted_score_away=_safe_int(row.get("predicted_score_away")),
                        confidence=_safe_float(row.get("confidence")),
                        scraped_at=datetime.now(),
                    )
                )
                inserted += 1

        session.commit()
        logger.info(f"[{source_name}] Ingested {inserted} new predictions")
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()

    return inserted


# ---------------------------------------------------------------------------
# BetExplorer: odds → MatchOdds
# ---------------------------------------------------------------------------

def ingest_betexplorer_odds(df: pd.DataFrame, session: Session | None = None) -> int:
    """Ingest BetExplorer odds data → MatchOdds table.

    Expected columns: home_team, away_team, date (or match_date),
    odds_home, odds_draw, odds_away.  Optional: market, odds_type.
    """
    if df.empty:
        return 0

    own_session = session is None
    if own_session:
        session = get_session()

    team_cache: dict = {}
    inserted = 0

    try:
        for _, row in df.iterrows():
            home_name = str(row.get("home_team", "")).strip()
            away_name = str(row.get("away_team", "")).strip()
            if not home_name or not away_name:
                continue

            date_val = row.get("match_date", row.get("date", ""))
            try:
                match_date = _parse_date(date_val)
            except (ValueError, KeyError):
                continue

            home_team_id = _get_or_create_team(session, home_name, team_cache)
            away_team_id = _get_or_create_team(session, away_name, team_cache)

            match = (
                session.query(Match)
                .filter(
                    Match.match_date == match_date,
                    Match.home_team_id == home_team_id,
                    Match.away_team_id == away_team_id,
                )
                .first()
            )
            if not match:
                continue

            market = str(row.get("market", "1X2")).strip() or "1X2"
            odds_type = str(row.get("odds_type", "pre-match")).strip() or "pre-match"

            existing = (
                session.query(MatchOdds)
                .filter(
                    MatchOdds.match_id == match.id,
                    MatchOdds.bookmaker == "BetExplorer",
                    MatchOdds.market == market,
                    MatchOdds.odds_type == odds_type,
                )
                .first()
            )
            if not existing:
                session.add(
                    MatchOdds(
                        match_id=match.id,
                        bookmaker="BetExplorer",
                        market=market,
                        odds_type=odds_type,
                        odds_home=_safe_float(row.get("odds_home")),
                        odds_draw=_safe_float(row.get("odds_draw")),
                        odds_away=_safe_float(row.get("odds_away")),
                    )
                )
                inserted += 1

        session.commit()
        logger.info(f"[betexplorer] Ingested {inserted} new odds records")
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()

    return inserted


# ---------------------------------------------------------------------------
# OddsPortal: closing odds → MatchOdds
# ---------------------------------------------------------------------------

def ingest_oddsportal_odds(df: pd.DataFrame, session: Session | None = None) -> int:
    """Ingest OddsPortal closing odds data → MatchOdds table.

    Expected columns: home_team, away_team, date (or match_date),
    odds_home, odds_draw, odds_away.  Optional: market, odds_type.
    """
    if df.empty:
        return 0

    own_session = session is None
    if own_session:
        session = get_session()

    team_cache: dict = {}
    inserted = 0

    try:
        for _, row in df.iterrows():
            home_name = str(row.get("home_team", "")).strip()
            away_name = str(row.get("away_team", "")).strip()
            if not home_name or not away_name:
                continue

            date_val = row.get("match_date", row.get("date", ""))
            try:
                match_date = _parse_date(date_val)
            except (ValueError, KeyError):
                continue

            home_team_id = _get_or_create_team(session, home_name, team_cache)
            away_team_id = _get_or_create_team(session, away_name, team_cache)

            match = (
                session.query(Match)
                .filter(
                    Match.match_date == match_date,
                    Match.home_team_id == home_team_id,
                    Match.away_team_id == away_team_id,
                )
                .first()
            )
            if not match:
                continue

            market = str(row.get("market", "1X2")).strip() or "1X2"
            # OddsPortal typically provides closing odds
            odds_type = str(row.get("odds_type", "closing")).strip() or "closing"

            existing = (
                session.query(MatchOdds)
                .filter(
                    MatchOdds.match_id == match.id,
                    MatchOdds.bookmaker == "OddsPortal",
                    MatchOdds.market == market,
                    MatchOdds.odds_type == odds_type,
                )
                .first()
            )
            if not existing:
                session.add(
                    MatchOdds(
                        match_id=match.id,
                        bookmaker="OddsPortal",
                        market=market,
                        odds_type=odds_type,
                        odds_home=_safe_float(row.get("odds_home")),
                        odds_draw=_safe_float(row.get("odds_draw")),
                        odds_away=_safe_float(row.get("odds_away")),
                    )
                )
                inserted += 1

        session.commit()
        logger.info(f"[oddsportal] Ingested {inserted} new odds records")
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()

    return inserted


# ---------------------------------------------------------------------------
# Betfair Exchange: exchange data → MatchOdds
# ---------------------------------------------------------------------------

def ingest_betfair_exchange(df: pd.DataFrame, session: Session | None = None) -> int:
    """Ingest Betfair exchange data → MatchOdds with bookmaker='BetfairExchange'.

    Expected columns: home_team, away_team, date (or match_date),
    odds_home, odds_draw, odds_away.  Optional: market, odds_type, volume.
    """
    if df.empty:
        return 0

    own_session = session is None
    if own_session:
        session = get_session()

    team_cache: dict = {}
    inserted = 0

    try:
        for _, row in df.iterrows():
            home_name = str(row.get("home_team", "")).strip()
            away_name = str(row.get("away_team", "")).strip()
            if not home_name or not away_name:
                continue

            date_val = row.get("match_date", row.get("date", ""))
            try:
                match_date = _parse_date(date_val)
            except (ValueError, KeyError):
                continue

            home_team_id = _get_or_create_team(session, home_name, team_cache)
            away_team_id = _get_or_create_team(session, away_name, team_cache)

            match = (
                session.query(Match)
                .filter(
                    Match.match_date == match_date,
                    Match.home_team_id == home_team_id,
                    Match.away_team_id == away_team_id,
                )
                .first()
            )
            if not match:
                continue

            market = str(row.get("market", "1X2")).strip() or "1X2"
            odds_type = str(row.get("odds_type", "exchange")).strip() or "exchange"

            existing = (
                session.query(MatchOdds)
                .filter(
                    MatchOdds.match_id == match.id,
                    MatchOdds.bookmaker == "BetfairExchange",
                    MatchOdds.market == market,
                    MatchOdds.odds_type == odds_type,
                )
                .first()
            )
            if not existing:
                session.add(
                    MatchOdds(
                        match_id=match.id,
                        bookmaker="BetfairExchange",
                        market=market,
                        odds_type=odds_type,
                        odds_home=_safe_float(row.get("odds_home")),
                        odds_draw=_safe_float(row.get("odds_draw")),
                        odds_away=_safe_float(row.get("odds_away")),
                    )
                )
                inserted += 1

        session.commit()
        logger.info(f"[betfair_exchange] Ingested {inserted} new exchange odds records")
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()

    return inserted


# ---------------------------------------------------------------------------
# Sofascore: fixtures → Match table (scheduled matches)
# ---------------------------------------------------------------------------

def ingest_sofascore(df: pd.DataFrame, session: Session | None = None) -> int:
    """Ingest Sofascore fixture data → Match table (creates scheduled matches).

    Expected columns: home_team, away_team, date (or match_date).
    Optional: kickoff_time, competition, season, status.
    """
    if df.empty:
        return 0

    own_session = session is None
    if own_session:
        session = get_session()

    team_cache: dict = {}
    league_cache: dict = {}
    season_cache: dict = {}
    inserted = 0

    try:
        for _, row in df.iterrows():
            home_name = str(row.get("home_team", "")).strip()
            away_name = str(row.get("away_team", "")).strip()
            if not home_name or not away_name or home_name == "nan":
                continue

            date_val = row.get("match_date", row.get("date", ""))
            try:
                match_date = _parse_date(date_val)
            except (ValueError, KeyError):
                continue

            home_team_id = _get_or_create_team(session, home_name, team_cache)
            away_team_id = _get_or_create_team(session, away_name, team_cache)

            league_name = str(row.get("competition", row.get("league", "Unknown"))).strip()
            season_label = str(row.get("season", "Unknown")).strip()
            league_id = _get_or_create_league(session, league_name, league_cache)
            season_id = _get_or_create_season(
                session, league_id, season_label, season_cache
            )

            existing = (
                session.query(Match)
                .filter(
                    Match.match_date == match_date,
                    Match.home_team_id == home_team_id,
                    Match.away_team_id == away_team_id,
                )
                .first()
            )
            if not existing:
                match_status = str(row.get("status", "scheduled")).strip() or "scheduled"
                session.add(
                    Match(
                        season_id=season_id,
                        match_date=match_date,
                        home_team_id=home_team_id,
                        away_team_id=away_team_id,
                        status=match_status,
                    )
                )
                inserted += 1

        session.commit()
        logger.info(f"[sofascore] Ingested {inserted} new scheduled matches")
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()

    return inserted


# ---------------------------------------------------------------------------
# FlashScore: results → update Match scores
# ---------------------------------------------------------------------------

def ingest_flashscore_results(df: pd.DataFrame, session: Session | None = None) -> int:
    """Ingest FlashScore result data → updates Match scores in place.

    Expected columns: home_team, away_team, date (or match_date),
    home_goals (or home_score), away_goals (or away_score).
    Optional: result (H/D/A).

    Only updates existing Match rows — does not create new matches.
    """
    if df.empty:
        return 0

    own_session = session is None
    if own_session:
        session = get_session()

    team_cache: dict = {}
    updated = 0

    try:
        for _, row in df.iterrows():
            home_name = str(row.get("home_team", "")).strip()
            away_name = str(row.get("away_team", "")).strip()
            if not home_name or not away_name:
                continue

            date_val = row.get("match_date", row.get("date", ""))
            try:
                match_date = _parse_date(date_val)
            except (ValueError, KeyError):
                continue

            home_team_id = _get_or_create_team(session, home_name, team_cache)
            away_team_id = _get_or_create_team(session, away_name, team_cache)

            match = (
                session.query(Match)
                .filter(
                    Match.match_date == match_date,
                    Match.home_team_id == home_team_id,
                    Match.away_team_id == away_team_id,
                )
                .first()
            )
            if not match:
                continue

            home_goals = _safe_int(row.get("home_goals", row.get("home_score")))
            away_goals = _safe_int(row.get("away_goals", row.get("away_score")))

            if home_goals is not None:
                match.home_goals = home_goals
            if away_goals is not None:
                match.away_goals = away_goals

            # Derive result from score if not provided
            raw_result = str(row.get("result", "")).strip()
            if raw_result in _RESULT_MAP:
                match.result = _RESULT_MAP[raw_result]
            elif home_goals is not None and away_goals is not None:
                if home_goals > away_goals:
                    match.result = "H"
                elif home_goals < away_goals:
                    match.result = "A"
                else:
                    match.result = "D"

            match.status = "played"
            updated += 1

        session.commit()
        logger.info(f"[flashscore] Updated scores for {updated} matches")
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()

    return updated


# ---------------------------------------------------------------------------
# Raw staging: generic helper
# ---------------------------------------------------------------------------

def _ingest_raw_staging(
    df: pd.DataFrame,
    source_name: str,
    record_type: str,
    session: Session | None = None,
) -> int:
    """Generic ingest into raw_staging_records for sources without dedicated tables."""
    if df.empty:
        return 0

    own_session = session is None
    if own_session:
        session = get_session()

    inserted = 0
    now = datetime.now()

    try:
        for _, row in df.iterrows():
            session.add(
                RawStagingRecord(
                    source=source_name,
                    record_type=record_type,
                    raw_data=_row_to_dict(row),
                    status="ingested",
                    created_at=now,
                )
            )
            inserted += 1

        session.commit()
        logger.info(
            f"[{source_name}] Staged {inserted} raw {record_type} records"
        )
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()

    return inserted


# ---------------------------------------------------------------------------
# Transfermarkt: injury/absence data → raw_staging_records
# ---------------------------------------------------------------------------

def ingest_transfermarkt_injuries(
    df: pd.DataFrame, session: Session | None = None
) -> int:
    """Ingest Transfermarkt injury data → raw_staging_records[record_type='injury']."""
    return _ingest_raw_staging(df, "transfermarkt", "injury", session)


# ---------------------------------------------------------------------------
# Lineups → raw_staging_records
# ---------------------------------------------------------------------------

def ingest_lineups(df: pd.DataFrame, session: Session | None = None) -> int:
    """Ingest lineup data → raw_staging_records[record_type='lineup']."""
    return _ingest_raw_staging(df, "lineup_scraper", "lineup", session)


# ---------------------------------------------------------------------------
# Team news → raw_staging_records
# ---------------------------------------------------------------------------

def ingest_team_news(df: pd.DataFrame, session: Session | None = None) -> int:
    """Ingest team news data → raw_staging_records[record_type='team_news']."""
    return _ingest_raw_staging(df, "team_news", "team_news", session)


# ---------------------------------------------------------------------------
# Referee stats → raw_staging_records
# ---------------------------------------------------------------------------

def ingest_referee_stats(df: pd.DataFrame, session: Session | None = None) -> int:
    """Ingest referee stats → raw_staging_records[record_type='referee_stats']."""
    return _ingest_raw_staging(df, "referee_stats", "referee_stats", session)


# ---------------------------------------------------------------------------
# Manager records → raw_staging_records
# ---------------------------------------------------------------------------

def ingest_manager_records(df: pd.DataFrame, session: Session | None = None) -> int:
    """Ingest manager records → raw_staging_records[record_type='manager_records']."""
    return _ingest_raw_staging(df, "manager_records", "manager_records", session)


# ---------------------------------------------------------------------------
# Wages (Capology) → raw_staging_records
# ---------------------------------------------------------------------------

def ingest_wages(df: pd.DataFrame, session: Session | None = None) -> int:
    """Ingest Capology wages data → raw_staging_records[record_type='wages']."""
    return _ingest_raw_staging(df, "capology", "wages", session)


# ---------------------------------------------------------------------------
# Advanced metrics (WhoScored / AdvancedMetrics) → raw_staging_records
# ---------------------------------------------------------------------------

def ingest_advanced_metrics(df: pd.DataFrame, session: Session | None = None) -> int:
    """Ingest advanced metrics → raw_staging_records[record_type='advanced_metrics']."""
    return _ingest_raw_staging(df, "advanced_metrics", "advanced_metrics", session)


# ---------------------------------------------------------------------------
# Dispatch: route DataFrame to the right ingest function
# ---------------------------------------------------------------------------

_PREDICTION_SOURCES = {"forebet", "predictz", "windrawwin", "footystats"}


def ingest_dataframe(df: pd.DataFrame, source_name: str) -> int:
    """Route a collector's DataFrame to the appropriate ingest function.

    Returns the number of new rows inserted (or updated for score patching).

    Supported sources
    -----------------
    football_data_uk, club_elo, understat,
    forebet, predictz, windrawwin, footystats   (→ competitor_predictions)
    betexplorer                                  (→ match_odds)
    oddsportal                                   (→ match_odds)
    betfair_exchange                             (→ match_odds)
    sofascore                                    (→ matches as scheduled)
    flashscore                                   (→ updates match scores)
    transfermarkt                                (→ raw_staging: injury)
    lineup_scraper, fotmob                       (→ raw_staging: lineup)
    team_news                                    (→ raw_staging: team_news)
    referee_stats                                (→ raw_staging: referee_stats)
    manager_records                              (→ raw_staging: manager_records)
    capology                                     (→ raw_staging: wages)
    advanced_metrics, whoscored                  (→ raw_staging: advanced_metrics)
    crowd_sentiment, european_fatigue            (→ raw_staging: <source_name>)
    """
    if df is None or df.empty:
        return 0

    if source_name == "football_data_uk":
        return ingest_football_data_uk(df)
    elif source_name == "club_elo":
        return ingest_club_elo(df)
    elif source_name == "understat":
        return ingest_understat(df)
    elif source_name in _PREDICTION_SOURCES:
        return ingest_predictions(df, source_name)
    elif source_name == "betexplorer":
        return ingest_betexplorer_odds(df)
    elif source_name == "oddsportal":
        return ingest_oddsportal_odds(df)
    elif source_name == "betfair_exchange":
        return ingest_betfair_exchange(df)
    elif source_name == "sofascore":
        return ingest_sofascore(df)
    elif source_name == "flashscore":
        return ingest_flashscore_results(df)
    elif source_name == "transfermarkt":
        return ingest_transfermarkt_injuries(df)
    elif source_name in ("lineup_scraper", "fotmob"):
        return ingest_lineups(df)
    elif source_name == "team_news":
        return ingest_team_news(df)
    elif source_name == "referee_stats":
        return ingest_referee_stats(df)
    elif source_name == "manager_records":
        return ingest_manager_records(df)
    elif source_name == "capology":
        return ingest_wages(df)
    elif source_name in ("advanced_metrics", "whoscored"):
        return ingest_advanced_metrics(df)
    elif source_name in ("crowd_sentiment", "european_fatigue"):
        # Stage raw for downstream processing
        return _ingest_raw_staging(df, source_name, source_name)
    elif source_name in ("fbref", "football_data_org", "open_meteo",
                         "world_football", "prediction_aggregator",
                         "soccerway"):
        # These sources need more complex mapping or aren't critical for training
        logger.debug(f"[{source_name}] Ingestion not yet implemented, skipping")
        return 0
    else:
        logger.warning(f"[{source_name}] Unknown source, skipping ingestion")
        return 0
