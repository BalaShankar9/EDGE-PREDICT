"""Track record endpoints (public)."""
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sharpedge.api.deps import get_db
from sharpedge.db.models import DailyPick
from sharpedge.pipeline.track_record import calculate_track_record

router = APIRouter(tags=["track_record"])


def _load_resolved_picks(db: Session) -> list[dict]:
    picks = db.query(DailyPick).filter(DailyPick.result.isnot(None)).all()
    return [{"tier": p.tier, "league": p.league, "result": p.result,
             "profit_loss": p.profit_loss, "best_odds": p.best_odds,
             "match_date": str(p.match_date)} for p in picks]


@router.get("/track-record")
async def track_record_overall(db: Session = Depends(get_db)):
    picks = _load_resolved_picks(db)
    record = calculate_track_record(picks)
    return {"status": "ok", "data": record, "meta": {"generated_at": datetime.now().isoformat()}}


@router.get("/track-record/by-tier")
async def track_record_by_tier(db: Session = Depends(get_db)):
    picks = _load_resolved_picks(db)
    record = calculate_track_record(picks)
    return {"status": "ok", "data": record.get("by_tier", {}), "meta": {"generated_at": datetime.now().isoformat()}}


@router.get("/track-record/by-league")
async def track_record_by_league(db: Session = Depends(get_db)):
    picks = _load_resolved_picks(db)
    record = calculate_track_record(picks)
    return {"status": "ok", "data": record.get("by_league", {}), "meta": {"generated_at": datetime.now().isoformat()}}
