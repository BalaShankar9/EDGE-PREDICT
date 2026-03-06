"""Match detail router — full prediction + context for a single match."""
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from sharpedge.api.deps import get_db
from sharpedge.db.models import DailyPick, Prediction

logger = logging.getLogger(__name__)
router = APIRouter(tags=["matches"])


@router.get("/match/{home_team}/{away_team}/{match_date}")
def get_match_detail(home_team: str, away_team: str, match_date: str, db: Session = Depends(get_db)):
    try:
        prediction = (
            db.query(Prediction)
            .filter(
                Prediction.home_team == home_team,
                Prediction.away_team == away_team,
                Prediction.match_date == match_date,
            )
            .order_by(Prediction.created_at.desc())
            .first()
        )
        if not prediction:
            return {"status": "error", "data": {"error": "Match not found"}, "meta": {}}
        pick = (
            db.query(DailyPick)
            .filter(
                DailyPick.home_team == home_team,
                DailyPick.away_team == away_team,
                DailyPick.match_date == match_date,
            )
            .order_by(DailyPick.id.desc())
            .first()
        )
        match_data = {
            "home_team": prediction.home_team,
            "away_team": prediction.away_team,
            "match_date": str(prediction.match_date),
            "league": prediction.league,
            "probabilities": {
                "home": prediction.prob_home,
                "draw": prediction.prob_draw,
                "away": prediction.prob_away,
                "over_25": prediction.prob_over,
                "under_25": prediction.prob_under,
                "btts_yes": prediction.prob_btts_yes,
                "btts_no": prediction.prob_btts_no,
            },
            "xgboost_probs": prediction.xgboost_probs,
            "poisson_probs": prediction.poisson_probs,
            "ensemble_weights": prediction.ensemble_weights,
        }
        if pick:
            match_data["pick"] = {
                "market": pick.pick_market,
                "selection": pick.pick_selection,
                "model_prob": pick.model_prob,
                "best_odds": pick.best_odds,
                "bookmaker": pick.bookmaker,
                "edge": pick.edge,
                "tier": pick.tier,
                "meta_agreement": pick.meta_agreement,
                "risk_flags": pick.risk_flags or [],
                "result": pick.result,
                "profit_loss": pick.profit_loss,
            }
        return {
            "status": "ok",
            "data": match_data,
            "meta": {"generated_at": datetime.now(timezone.utc).isoformat()},
        }
    except Exception as e:
        logger.error(f"Error getting match detail: {e}")
        return {"status": "error", "data": {"error": str(e)[:200]}, "meta": {}}
