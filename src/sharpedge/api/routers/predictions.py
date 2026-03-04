"""Prediction endpoints (public)."""
from datetime import date, datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sharpedge.api.deps import get_db
from sharpedge.db.models import Prediction

router = APIRouter(tags=["predictions"])


def _serialize_prediction(p: Prediction) -> dict:
    return {
        "id": p.id, "match_date": str(p.match_date),
        "home_team": p.home_team, "away_team": p.away_team, "league": p.league,
        "prob_home": p.prob_home, "prob_draw": p.prob_draw, "prob_away": p.prob_away,
        "prob_over": p.prob_over, "prob_under": p.prob_under,
        "prob_btts_yes": p.prob_btts_yes, "prob_btts_no": p.prob_btts_no,
    }


@router.get("/predictions/today")
async def predictions_today(db: Session = Depends(get_db)):
    today = date.today()
    preds = db.query(Prediction).filter(Prediction.match_date == today).all()
    return {"status": "ok", "data": [_serialize_prediction(p) for p in preds],
            "meta": {"count": len(preds), "generated_at": datetime.now().isoformat()}}


@router.get("/predictions/{pred_date}")
async def predictions_by_date(pred_date: str, db: Session = Depends(get_db)):
    try:
        target = date.fromisoformat(pred_date)
    except ValueError:
        return {"status": "error", "data": {"error": "Invalid date format"}, "meta": {}}
    preds = db.query(Prediction).filter(Prediction.match_date == target).all()
    return {"status": "ok", "data": [_serialize_prediction(p) for p in preds],
            "meta": {"count": len(preds), "generated_at": datetime.now().isoformat()}}
