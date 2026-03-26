"""Prediction endpoints (public)."""
from datetime import date, datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from sharpedge.api.deps import get_db
from sharpedge.db.models import League as LeagueModel, Match, MatchOdds, Prediction, Season, Team

router = APIRouter(tags=["predictions"])


def _serialize_prediction(p: Prediction) -> dict:
    return {
        "id": p.id, "match_date": str(p.match_date),
        "home_team": p.home_team, "away_team": p.away_team, "league": p.league,
        "prob_home": p.prob_home, "prob_draw": p.prob_draw, "prob_away": p.prob_away,
        "prob_over": p.prob_over, "prob_under": p.prob_under,
        "prob_btts_yes": p.prob_btts_yes, "prob_btts_no": p.prob_btts_no,
    }


def _serialize_historical_match(m: Match) -> dict:
    """Serialize a Match row from the training data as a historical prediction."""
    home_name = m.home_team.canonical_name if m.home_team else "Unknown"
    away_name = m.away_team.canonical_name if m.away_team else "Unknown"
    league_name = m.season.league.name if m.season and m.season.league else "Unknown"

    # Get best available odds
    odds = None
    if m.odds:
        for o in m.odds:
            if o.odds_home and o.odds_draw and o.odds_away:
                odds = {
                    "home": round(o.odds_home, 2),
                    "draw": round(o.odds_draw, 2),
                    "away": round(o.odds_away, 2),
                    "bookmaker": o.bookmaker,
                }
                break

    return {
        "match_date": str(m.match_date),
        "home_team": home_name,
        "away_team": away_name,
        "league": league_name,
        "home_goals": m.home_goals,
        "away_goals": m.away_goals,
        "result": m.result,
        "kick_off_time": m.kick_off_time,
        "odds": odds,
    }


@router.get("/predictions/today")
async def predictions_today(db: Session = Depends(get_db)):
    today = date.today()
    preds = db.query(Prediction).filter(Prediction.match_date == today).all()
    return {"status": "ok", "data": [_serialize_prediction(p) for p in preds],
            "meta": {"count": len(preds), "generated_at": datetime.now().isoformat()}}


@router.get("/predictions/upcoming")
async def predictions_upcoming(
    days: int = Query(7, ge=1, le=30),
    db: Session = Depends(get_db),
):
    """Return all predictions from today through today+N days."""
    start = date.today()
    end = start + timedelta(days=days)
    preds = (
        db.query(Prediction)
        .filter(Prediction.match_date >= start, Prediction.match_date <= end)
        .order_by(Prediction.match_date, Prediction.league)
        .all()
    )
    return {
        "status": "ok",
        "data": [_serialize_prediction(p) for p in preds],
        "meta": {"count": len(preds), "generated_at": datetime.now().isoformat()},
    }


@router.get("/predictions/history")
async def predictions_history(
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
    league: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Return historical matches from the training data with results and odds."""
    from sqlalchemy.orm import aliased
    HomeTeam = aliased(Team, name="home_t")
    AwayTeam = aliased(Team, name="away_t")

    query = (
        db.query(Match)
        .join(HomeTeam, Match.home_team_id == HomeTeam.id)
        .join(AwayTeam, Match.away_team_id == AwayTeam.id)
        .join(Match.season)
        .join(Season.league)
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
            joinedload(Match.season).joinedload(Season.league),
            joinedload(Match.odds),
        )
        .filter(Match.home_goals.isnot(None))
        .distinct(Match.match_date, Match.home_team_id, Match.away_team_id)
    )

    if league:
        query = query.filter(LeagueModel.name.ilike(f"%{league}%"))

    if date_from:
        try:
            query = query.filter(Match.match_date >= date.fromisoformat(date_from))
        except ValueError:
            pass
    if date_to:
        try:
            query = query.filter(Match.match_date <= date.fromisoformat(date_to))
        except ValueError:
            pass

    if search:
        query = query.filter(
            HomeTeam.canonical_name.ilike(f"%{search}%")
            | AwayTeam.canonical_name.ilike(f"%{search}%")
        )

    total = query.count()
    matches = (
        query.order_by(Match.match_date.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    return {
        "status": "ok",
        "data": [_serialize_historical_match(m) for m in matches],
        "meta": {
            "count": len(matches),
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
            "generated_at": datetime.now().isoformat(),
        },
    }


@router.get("/predictions/stats")
async def predictions_stats(db: Session = Depends(get_db)):
    """Return summary stats for hero section."""
    today = date.today()
    upcoming_count = (
        db.query(func.count(Prediction.id))
        .filter(Prediction.match_date >= today)
        .scalar()
    )
    historical_count = (
        db.query(func.count(func.distinct(
            func.concat(Match.match_date, '-', Match.home_team_id, '-', Match.away_team_id)
        )))
        .filter(Match.home_goals.isnot(None))
        .scalar()
    )
    leagues_count = (
        db.query(func.count(func.distinct(Prediction.league)))
        .filter(Prediction.match_date >= today)
        .scalar()
    )
    return {
        "status": "ok",
        "data": {
            "upcoming_predictions": upcoming_count or 0,
            "historical_matches": historical_count or 0,
            "leagues_count": leagues_count or 0,
        },
        "meta": {"generated_at": datetime.now().isoformat()},
    }


@router.get("/predictions/{pred_date}")
async def predictions_by_date(pred_date: str, db: Session = Depends(get_db)):
    try:
        target = date.fromisoformat(pred_date)
    except ValueError:
        return {"status": "error", "data": {"error": "Invalid date format"}, "meta": {}}
    preds = db.query(Prediction).filter(Prediction.match_date == target).all()
    return {"status": "ok", "data": [_serialize_prediction(p) for p in preds],
            "meta": {"count": len(preds), "generated_at": datetime.now().isoformat()}}
