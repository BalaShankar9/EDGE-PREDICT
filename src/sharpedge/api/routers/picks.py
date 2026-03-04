"""Picks endpoints (public)."""
from datetime import date, datetime
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


@router.get("/picks/history")
async def picks_history(
    tier: str = Query(None), league: str = Query(None),
    limit: int = Query(50, le=200), offset: int = Query(0),
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
