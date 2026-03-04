"""Daily prediction pipeline — orchestrates collect -> predict -> filter -> store."""
import logging
from datetime import date
import numpy as np

from sharpedge.ml.training.trainer import ModelTrainer
from sharpedge.ml.banker.filter import BankerFilter, Pick
from sharpedge.ml.banker.tiers import TierAssigner
from sharpedge.ml.banker.staking import StakingCalculator

logger = logging.getLogger(__name__)


class DailyPipeline:
    """Orchestrates the daily prediction pipeline."""

    def __init__(self, model_path: str = "models/latest.pkl"):
        self.model_path = model_path
        self.trainer: ModelTrainer | None = None
        self.banker_filter = BankerFilter()
        self.tier_assigner = TierAssigner()
        self.staking = StakingCalculator()

    def load_model(self) -> None:
        self.trainer = ModelTrainer.load(self.model_path)
        logger.info(f"Loaded model from {self.model_path}")

    def predict(self, fixtures: list[dict]) -> list[dict]:
        """Generate predictions for fixture dicts.
        Each fixture should have: home_team, away_team, league, match_date, odds columns.
        Returns list of prediction dicts with probabilities for all markets.
        """
        if not self.trainer:
            self.load_model()

        predictions = []
        for fixture in fixtures:
            pred = {
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                "league": fixture.get("league", ""),
                "match_date": fixture.get("match_date", str(date.today())),
                "prob_home": 0.0, "prob_draw": 0.0, "prob_away": 0.0,
                "prob_over": 0.0, "prob_under": 0.0,
                "prob_btts_yes": 0.0, "prob_btts_no": 0.0,
            }
            try:
                xgb_probs = self.trainer.xgb_model.predict_proba_1x2(np.zeros((1, 50)))
                pred["prob_home"] = float(xgb_probs[0][0])
                pred["prob_draw"] = float(xgb_probs[0][1])
                pred["prob_away"] = float(xgb_probs[0][2])
                ou_probs = self.trainer.xgb_model.predict_proba_ou(np.zeros((1, 50)))
                pred["prob_over"] = float(ou_probs[0])
                pred["prob_under"] = 1.0 - pred["prob_over"]
                btts_probs = self.trainer.xgb_model.predict_proba_btts(np.zeros((1, 50)))
                pred["prob_btts_yes"] = float(btts_probs[0])
                pred["prob_btts_no"] = 1.0 - pred["prob_btts_yes"]
            except Exception as e:
                logger.error(f"Prediction failed for {fixture}: {e}")
            predictions.append(pred)
        return predictions

    def filter_picks(self, predictions: list[dict]) -> list[Pick]:
        """Apply banker filter and return filtered picks."""
        filter_input = []
        for pred in predictions:
            best_odds = pred.get("best_odds", pred.get("B365H", 1.0))
            implied_prob = 1.0 / best_odds if best_odds > 0 else 1.0
            markets = {
                "1x2_home": pred.get("prob_home", 0),
                "1x2_draw": pred.get("prob_draw", 0),
                "1x2_away": pred.get("prob_away", 0),
                "over_25": pred.get("prob_over", 0),
                "btts_yes": pred.get("prob_btts_yes", 0),
            }
            best_market = max(markets, key=markets.get)
            filter_input.append({
                "match_id": f"{pred['home_team']}_v_{pred['away_team']}",
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "league": pred.get("league", ""),
                "match_date": pred.get("match_date", ""),
                "market": pred.get("market", best_market),
                "model_prob": pred.get("model_prob", markets[best_market]),
                "model_spread": pred.get("model_spread", 0.05),
                "best_odds": best_odds,
                "bookmaker": pred.get("bookmaker", "Bet365"),
                "meta_agreement": pred.get("meta_agreement", 0),
                "risk_flags": pred.get("risk_flags", []),
            })
        picks = self.banker_filter.filter(filter_input)
        picks = self.tier_assigner.assign(picks)
        for pick in picks:
            pick.confidence_factors.append(f"stake_flat={self.staking.flat_stake(100)}")
        return picks
