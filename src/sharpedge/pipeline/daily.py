"""Daily prediction pipeline — orchestrates collect -> predict -> filter -> store."""
import logging
from datetime import date

import numpy as np
import pandas as pd

from sharpedge.ml.features.pipeline import FeaturePipeline
from sharpedge.ml.models.ensemble import EnsemblePredictor
from sharpedge.ml.models.calibration import ProbabilityCalibrator
from sharpedge.ml.training.trainer import ModelTrainer
from sharpedge.ml.banker.filter import BankerFilter, Pick
from sharpedge.ml.banker.tiers import TierAssigner
from sharpedge.ml.banker.staking import StakingCalculator
from sharpedge.ml.banker.league_calibration import adjust_confidence

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

    def predict(
        self,
        fixtures: list[dict],
        historical_matches: pd.DataFrame | None = None,
        elo_df: pd.DataFrame | None = None,
        xg_df: pd.DataFrame | None = None,
        predictions_df: pd.DataFrame | None = None,
        injuries_df: pd.DataFrame | None = None,
    ) -> list[dict]:
        """Generate predictions for fixture dicts.

        Parameters
        ----------
        fixtures : list of dicts, each with keys:
            home_team_id, away_team_id, league, match_date,
            and optionally odds columns (B365H, B365D, B365A, etc.)
        historical_matches : past match data for feature computation
            (rolling form, H2H, etc.). If None, features will be NaN-filled.
        elo_df : ELO ratings for feature computation. If None, ELO features
            will be NaN-filled.
        xg_df : xG data from Understat. If None, xG features will be NaN-filled.
        predictions_df : Competitor predictions. If None, meta features will be NaN-filled.
        injuries_df : Injury data from Transfermarkt. If None, injury features
            default to 0 (no injury advantage assumed).

        Returns
        -------
        list of prediction dicts with probabilities for all markets
        """
        if not self.trainer:
            self.load_model()

        # Build feature rows by appending fixtures to historical data
        fixture_rows = []
        for fixture in fixtures:
            # Normalize match_date to date object (handles ISO strings like "2026-03-08T15:00:00Z")
            raw_date = fixture.get("match_date", str(date.today()))
            try:
                match_date = pd.to_datetime(raw_date).date()
            except Exception:
                match_date = date.today()

            fixture_rows.append({
                "home_team_id": fixture.get("home_team_id", fixture.get("home_team", "")),
                "away_team_id": fixture.get("away_team_id", fixture.get("away_team", "")),
                "match_date": match_date,
                "league": fixture.get("league", ""),
                "FTHG": np.nan,  # Unknown — match hasn't happened
                "FTAG": np.nan,
                "FTR": "",
                "B365H": fixture.get("B365H", np.nan),
                "B365D": fixture.get("B365D", np.nan),
                "B365A": fixture.get("B365A", np.nan),
                "PSH": fixture.get("PSH", np.nan),
                "PSD": fixture.get("PSD", np.nan),
                "PSA": fixture.get("PSA", np.nan),
                "WHH": fixture.get("WHH", np.nan),
                "WHD": fixture.get("WHD", np.nan),
                "WHA": fixture.get("WHA", np.nan),
                "MaxH": fixture.get("MaxH", np.nan),
                "MaxD": fixture.get("MaxD", np.nan),
                "MaxA": fixture.get("MaxA", np.nan),
                "AvgH": fixture.get("AvgH", np.nan),
                "AvgD": fixture.get("AvgD", np.nan),
                "AvgA": fixture.get("AvgA", np.nan),
                "HS": np.nan, "AS": np.nan,
                "HST": np.nan, "AST": np.nan,
                "HF": np.nan, "AF": np.nan,
                "HC": np.nan, "AC": np.nan,
                "HY": np.nan, "AY": np.nan,
                "HR": np.nan, "AR": np.nan,
                "HTHG": np.nan, "HTAG": np.nan,
                "Referee": fixture.get("Referee", ""),
                "season": fixture.get("season", ""),
            })

        fixture_df = pd.DataFrame(fixture_rows)

        # Combine with historical matches so rolling features can be computed
        if historical_matches is not None:
            # Ensure consistent date types (DB returns datetime, fixtures are date)
            if "match_date" in historical_matches.columns:
                historical_matches["match_date"] = pd.to_datetime(
                    historical_matches["match_date"]
                ).dt.date
            combined_df = pd.concat([historical_matches, fixture_df], ignore_index=True)
        else:
            combined_df = fixture_df

        # Build features on the full dataset
        pipe = FeaturePipeline()
        feature_matrix = pipe.build(
            combined_df, elo_df=elo_df, xg_df=xg_df,
            predictions_df=predictions_df, injuries_df=injuries_df,
        )

        # Extract only the fixture rows (last N rows)
        # Keep NaN — XGBoost handles missing values natively
        n_fixtures = len(fixtures)
        X_fixtures = feature_matrix.iloc[-n_fixtures:].values.astype(np.float32)

        # Augment features with Dixon-Coles team strengths (same as training)
        if self.trainer and self.trainer.poisson_model:
            dc_features = np.full((n_fixtures, 4), np.nan, dtype=np.float32)
            for i, fixture in enumerate(fixtures):
                home_id = fixture.get("home_team_id", fixture.get("home_team", ""))
                away_id = fixture.get("away_team_id", fixture.get("away_team", ""))
                dc_features[i, 0] = self.trainer.poisson_model.attack_strength.get(home_id, np.nan)
                dc_features[i, 1] = self.trainer.poisson_model.defence_strength.get(home_id, np.nan)
                dc_features[i, 2] = self.trainer.poisson_model.attack_strength.get(away_id, np.nan)
                dc_features[i, 3] = self.trainer.poisson_model.defence_strength.get(away_id, np.nan)
            X_fixtures = np.hstack([X_fixtures, dc_features])

        # Generate predictions
        predictions = []
        for i, fixture in enumerate(fixtures):
            X_row = X_fixtures[i : i + 1]
            pred = {
                "home_team": fixture.get("home_team_id", fixture.get("home_team", "")),
                "away_team": fixture.get("away_team_id", fixture.get("away_team", "")),
                "league": fixture.get("league", ""),
                "match_date": fixture.get("match_date", str(date.today())),
                "prob_home": 0.0, "prob_draw": 0.0, "prob_away": 0.0,
                "prob_over": 0.0, "prob_under": 0.0,
                "prob_btts_yes": 0.0, "prob_btts_no": 0.0,
            }
            # Carry forward odds for filter_picks edge calculation
            for odds_col in ("B365H", "B365D", "B365A", "MaxH", "MaxD", "MaxA",
                             "AvgH", "AvgD", "AvgA", "over_25_odds", "under_25_odds"):
                if odds_col in fixture:
                    pred[odds_col] = fixture[odds_col]
            try:
                # XGBoost predictions
                xgb_probs = self.trainer.xgb_model.predict_proba_1x2(X_row)

                # Poisson predictions
                home_id = fixture.get("home_team_id", fixture.get("home_team", ""))
                away_id = fixture.get("away_team_id", fixture.get("away_team", ""))
                poisson_probs = self.trainer.poisson_model.predict_proba_1x2(
                    home_id, away_id
                )

                # Bivariate Poisson predictions
                bvp_probs = np.array([1/3, 1/3, 1/3])
                if self.trainer.bvp_model is not None:
                    bvp_probs = self.trainer.bvp_model.predict_proba_1x2(
                        home_id, away_id
                    )

                # CatBoost predictions
                if getattr(self.trainer, "catboost_model", None) is not None:
                    catboost_probs = self.trainer.catboost_model.predict_proba_1x2(X_row)
                else:
                    catboost_probs = np.array([[1/3, 1/3, 1/3]])

                # LightGBM predictions
                if getattr(self.trainer, "lgbm_model", None) is not None:
                    lgbm_probs = self.trainer.lgbm_model.predict_proba_1x2(X_row)
                else:
                    lgbm_probs = np.array([[1/3, 1/3, 1/3]])

                # Stacking meta-learner (preferred) or weighted ensemble (fallback)
                base_preds = {
                    "xgboost": xgb_probs,
                    "poisson": poisson_probs.reshape(1, 3),
                    "bivariate_poisson": bvp_probs.reshape(1, 3),
                    "catboost": catboost_probs,
                    "lightgbm": lgbm_probs,
                }
                if self.trainer.stacking is not None:
                    combined = self.trainer.stacking.predict(base_preds)
                else:
                    combined = self.trainer.ensemble.predict(base_preds)

                # Calibrate
                calibrated = self.trainer.calibrator.calibrate(combined)

                # Two-stage adaptive blend with OvR specialist:
                #   <0.75: pure baseline (OvR weight = 0)
                #   0.75-0.80: gentle ramp 0→0.5 (preserves Platinum accuracy)
                #   0.80+: aggressive ramp 0.5→0.9 (maximizes Diamond accuracy)
                # Validated: Diamond 91.3%, Platinum 82.5%, ROI +8.3%
                if self.trainer.ovr_model is not None:
                    ovr_probs = self.trainer.ovr_model.predict_proba_1x2(X_row)
                    baseline_max = float(calibrated[0].max())
                    if baseline_max < 0.75:
                        w_ovr = 0.0
                    elif baseline_max < 0.80:
                        w_ovr = 0.5 * (baseline_max - 0.75) / 0.05
                    else:
                        w_ovr = 0.5 + 0.4 * min((baseline_max - 0.80) / 0.10, 1.0)
                    blended = (1.0 - w_ovr) * calibrated + w_ovr * ovr_probs
                    blended = blended / blended.sum(axis=1, keepdims=True)
                else:
                    blended = calibrated

                # Skip league calibration for high-confidence picks
                # (experiment shows it hurts accuracy at ≥0.80: 88.5% → 85.8%)
                blended_max = float(blended[0].max())
                if blended_max < 0.75:
                    # League calibration helps at moderate confidence
                    league = fixture.get("league", "")
                    probs_1x2 = np.array([
                        adjust_confidence(float(blended[0][c]), league)
                        for c in range(3)
                    ])
                    total = probs_1x2.sum()
                    if total > 0:
                        probs_1x2 = probs_1x2 / total
                else:
                    # High confidence: use raw blend (preserves sharpness)
                    probs_1x2 = blended[0]

                # Apply draw probability floor (draws are ~26% of outcomes)
                draw_floor = getattr(self.trainer, "draw_floor", 0.18)
                if probs_1x2[1] < draw_floor:
                    deficit = draw_floor - probs_1x2[1]
                    probs_1x2 = probs_1x2.copy()
                    probs_1x2[1] = draw_floor
                    # Redistribute deficit proportionally from H and A
                    ha_sum = probs_1x2[0] + probs_1x2[2]
                    if ha_sum > 0:
                        probs_1x2[0] -= deficit * (probs_1x2[0] / ha_sum)
                        probs_1x2[2] -= deficit * (probs_1x2[2] / ha_sum)
                    probs_1x2 = np.clip(probs_1x2, 0.01, None)
                    probs_1x2 = probs_1x2 / probs_1x2.sum()

                pred["prob_home"] = float(probs_1x2[0])
                pred["prob_draw"] = float(probs_1x2[1])
                pred["prob_away"] = float(probs_1x2[2])

                # O/U — weighted XGBoost 60% + DC 40% (BVP excluded)
                xgb_ou = float(self.trainer.xgb_model.predict_proba_ou(X_row)[0])
                dc_ou = self.trainer.poisson_model.predict_proba_ou(
                    home_id, away_id, 2.5
                )
                raw_ou = 0.6 * xgb_ou + 0.4 * dc_ou
                if getattr(self.trainer, "ou_calibrator", None) is not None:
                    raw_ou = float(
                        self.trainer.ou_calibrator.predict(np.array([raw_ou]))[0]
                    )
                pred["prob_over"] = raw_ou
                pred["prob_under"] = 1.0 - raw_ou

                # BTTS — weighted XGBoost 60% + DC 40%
                xgb_btts = float(self.trainer.xgb_model.predict_proba_btts(X_row)[0])
                dc_btts = self.trainer.poisson_model.predict_proba_btts(
                    home_id, away_id
                )
                raw_btts = 0.6 * xgb_btts + 0.4 * dc_btts
                if getattr(self.trainer, "btts_calibrator", None) is not None:
                    raw_btts = float(
                        self.trainer.btts_calibrator.predict(np.array([raw_btts]))[0]
                    )
                pred["prob_btts_yes"] = raw_btts
                pred["prob_btts_no"] = 1.0 - raw_btts

                # Add confidence metadata
                probs_1x2 = calibrated[0]
                sorted_probs = np.sort(probs_1x2)
                pred["model_spread"] = float(sorted_probs[2] - sorted_probs[1])

            except Exception as e:
                logger.error(f"Prediction failed for {fixture}: {e}")

            predictions.append(pred)

        return predictions

    def filter_picks(self, predictions: list[dict]) -> list[Pick]:
        """Apply banker filter and return filtered picks."""
        # Map market → best odds column (prefer Max, fall back to B365, then Avg)
        _ODDS_FOR_MARKET = {
            "1x2_home": ["MaxH", "B365H", "AvgH"],
            "1x2_draw": ["MaxD", "B365D", "AvgD"],
            "1x2_away": ["MaxA", "B365A", "AvgA"],
            "over_25": ["over_25_odds"],
        }

        filter_input = []
        for pred in predictions:
            markets = {
                "1x2_home": pred.get("prob_home", 0),
                "1x2_draw": pred.get("prob_draw", 0),
                "1x2_away": pred.get("prob_away", 0),
                "over_25": pred.get("prob_over", 0),
                "btts_yes": pred.get("prob_btts_yes", 0),
            }
            best_market = max(markets, key=markets.get)

            # Find best odds for the selected market
            best_odds = pred.get("best_odds", 1.0)
            if best_odds <= 1.0:
                for col in _ODDS_FOR_MARKET.get(best_market, []):
                    val = pred.get(col)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        best_odds = float(val)
                        break

            filter_input.append({
                "match_id": f"{pred['home_team']}_v_{pred['away_team']}",
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "league": pred.get("league", ""),
                "match_date": pred.get("match_date", ""),
                "market": pred.get("market", best_market),
                "model_prob": pred.get("model_prob", markets[best_market]),
                "model_spread": pred.get("model_spread", 0.0),
                "best_odds": best_odds,
                "bookmaker": pred.get("bookmaker", "BetExplorer"),
                "risk_flags": pred.get("risk_flags", []),
            })
        picks = self.banker_filter.filter(filter_input)
        picks = self.tier_assigner.assign(picks)
        for pick in picks:
            pick.confidence_factors.append(f"stake_flat={self.staking.flat_stake(100)}")
        return picks
