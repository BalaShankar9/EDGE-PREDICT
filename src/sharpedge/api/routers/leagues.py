"""Leagues router — league list with fixture/pick counts."""
import logging
from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from sharpedge.api.deps import get_db
from sharpedge.db.models import DailyPick, Prediction

logger = logging.getLogger(__name__)
router = APIRouter(tags=["leagues"])


def _slugify(name: str) -> str:
    return name.lower().replace(" ", "-").replace(".", "")


@router.get("/leagues")
def list_leagues(db: Session = Depends(get_db)):
    try:
        today = date.today()
        league_preds = (
            db.query(Prediction.league, func.count(Prediction.id).label("prediction_count"))
            .filter(Prediction.match_date >= today)
            .group_by(Prediction.league)
            .all()
        )
        league_picks = (
            db.query(DailyPick.league, func.count(DailyPick.id).label("pick_count"))
            .filter(DailyPick.match_date >= today)
            .group_by(DailyPick.league)
            .all()
        )
        pick_counts = {row.league: row.pick_count for row in league_picks}
        leagues = []
        for row in league_preds:
            leagues.append({
                "name": row.league,
                "slug": _slugify(row.league),
                "prediction_count": row.prediction_count,
                "pick_count": pick_counts.get(row.league, 0),
            })
        leagues.sort(key=lambda x: x["pick_count"], reverse=True)
        return {
            "status": "ok",
            "data": leagues,
            "meta": {"count": len(leagues), "generated_at": datetime.now(timezone.utc).isoformat()},
        }
    except Exception as e:
        logger.error(f"Error listing leagues: {e}")
        return {"status": "error", "data": {"error": str(e)[:200]}, "meta": {}}


@router.get("/league/{slug}/predictions")
def get_league_predictions(slug: str, db: Session = Depends(get_db)):
    try:
        today = date.today()
        predictions = (
            db.query(Prediction)
            .filter(Prediction.match_date >= today)
            .order_by(Prediction.match_date)
            .all()
        )
        league_preds = []
        league_name = None
        for pred in predictions:
            if _slugify(pred.league) == slug:
                league_name = pred.league
                league_preds.append(pred)
        if not league_name:
            return {"status": "error", "data": {"error": f"League not found: {slug}"}, "meta": {}}
        picks = (
            db.query(DailyPick)
            .filter(DailyPick.league == league_name, DailyPick.match_date >= today)
            .all()
        )
        pick_map = {f"{p.home_team}|{p.away_team}|{p.match_date}": p for p in picks}
        data = []
        for pred in league_preds:
            key = f"{pred.home_team}|{pred.away_team}|{pred.match_date}"
            pick = pick_map.get(key)
            entry = {
                "home_team": pred.home_team,
                "away_team": pred.away_team,
                "match_date": str(pred.match_date),
                "prob_home": pred.prob_home,
                "prob_draw": pred.prob_draw,
                "prob_away": pred.prob_away,
            }
            if pick:
                entry["pick"] = {
                    "selection": pick.pick_selection,
                    "tier": pick.tier,
                    "best_odds": pick.best_odds,
                    "edge": pick.edge,
                }
            data.append(entry)
        return {
            "status": "ok",
            "data": {"league": league_name, "slug": slug, "predictions": data},
            "meta": {"count": len(data), "generated_at": datetime.now(timezone.utc).isoformat()},
        }
    except Exception as e:
        logger.error(f"Error getting league predictions: {e}")
        return {"status": "error", "data": {"error": str(e)[:200]}, "meta": {}}
