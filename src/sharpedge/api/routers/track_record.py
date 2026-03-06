"""Track record endpoints (public)."""
import logging
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from sharpedge.api.deps import get_db
from sharpedge.db.models import DailyPick
from sharpedge.pipeline.track_record import calculate_track_record

logger = logging.getLogger(__name__)

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


@router.get("/track-record/monthly")
async def track_record_monthly(db: Session = Depends(get_db)):
    try:
        resolved = (
            db.query(DailyPick)
            .filter(DailyPick.result.isnot(None))
            .order_by(DailyPick.match_date)
            .all()
        )
        monthly: dict[str, dict] = defaultdict(lambda: {"picks": 0, "wins": 0, "profit": 0.0, "staked": 0.0})
        for pick in resolved:
            month_key = pick.match_date.strftime("%Y-%m")
            monthly[month_key]["picks"] += 1
            if pick.result == "win":
                monthly[month_key]["wins"] += 1
            monthly[month_key]["profit"] += pick.profit_loss or 0.0
            monthly[month_key]["staked"] += pick.stake_flat or 1.0

        series = []
        cumulative_profit = 0.0
        for month_key in sorted(monthly.keys()):
            m = monthly[month_key]
            cumulative_profit += m["profit"]
            roi = (m["profit"] / m["staked"] * 100) if m["staked"] else 0.0
            series.append({
                "month": month_key,
                "picks": m["picks"],
                "wins": m["wins"],
                "profit": round(m["profit"], 2),
                "roi": round(roi, 2),
                "cumulative_profit": round(cumulative_profit, 2),
            })
        return {
            "status": "ok",
            "data": series,
            "meta": {"count": len(series), "generated_at": datetime.now(timezone.utc).isoformat()},
        }
    except Exception as e:
        logger.error(f"Error computing monthly track record: {e}")
        return {"status": "error", "data": {"error": str(e)[:200]}, "meta": {}}
