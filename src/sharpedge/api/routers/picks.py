"""Picks endpoints (public)."""
from datetime import date, datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from sharpedge.api.deps import get_db
from sharpedge.db.models import DailyPick

router = APIRouter(tags=["picks"])


def _serialize_pick(p: DailyPick) -> dict:
    return {
        "id": p.id, "match_date": str(p.match_date),
        "home_team": p.home_team, "away_team": p.away_team, "league": p.league,
        "pick_market": p.pick_market, "pick_selection": p.pick_selection,
        "model_prob": p.model_prob, "best_odds": p.best_odds,
        "bookmaker": p.bookmaker, "edge": p.edge, "tier": p.tier,
        "meta_agreement": p.meta_agreement,
        "result": p.result, "profit_loss": p.profit_loss,
    }


@router.get("/picks/today")
async def picks_today(db: Session = Depends(get_db)):
    today = date.today()
    picks = db.query(DailyPick).filter(DailyPick.match_date == today).all()
    return {"status": "ok", "data": [_serialize_pick(p) for p in picks],
            "meta": {"count": len(picks), "generated_at": datetime.now().isoformat()}}


@router.get("/picks/upcoming")
async def picks_upcoming(
    days: int = Query(7, ge=1, le=30),
    db: Session = Depends(get_db),
):
    """Return all picks from today through today+N days (resolved and unresolved)."""
    start = date.today()
    end = start + timedelta(days=days)
    picks = (
        db.query(DailyPick)
        .filter(DailyPick.match_date >= start, DailyPick.match_date <= end)
        .order_by(DailyPick.match_date, DailyPick.tier)
        .all()
    )
    return {
        "status": "ok",
        "data": [_serialize_pick(p) for p in picks],
        "meta": {"count": len(picks), "generated_at": datetime.now().isoformat()},
    }


@router.get("/picks/date/{pick_date}")
async def picks_by_date(pick_date: str, db: Session = Depends(get_db)):
    """Return all picks for a specific date (resolved or unresolved)."""
    try:
        target = date.fromisoformat(pick_date)
    except ValueError:
        return {"status": "error", "data": {"error": "Invalid date format"}, "meta": {}}
    picks = (
        db.query(DailyPick)
        .filter(DailyPick.match_date == target)
        .order_by(DailyPick.tier)
        .all()
    )
    return {
        "status": "ok",
        "data": [_serialize_pick(p) for p in picks],
        "meta": {"count": len(picks), "generated_at": datetime.now().isoformat()},
    }


@router.get("/picks/history")
async def picks_history(
    tier: Optional[str] = Query(None),
    league: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    db: Session = Depends(get_db),
):
    query = db.query(DailyPick).filter(DailyPick.result.isnot(None))
    if tier:
        query = query.filter(DailyPick.tier == tier)
    if league:
        query = query.filter(DailyPick.league == league)
    picks = query.order_by(DailyPick.match_date.desc()).offset(offset).limit(limit).all()
    return {"status": "ok", "data": [_serialize_pick(p) for p in picks],
            "meta": {"count": len(picks), "generated_at": datetime.now().isoformat()}}
