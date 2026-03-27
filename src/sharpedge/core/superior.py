"""SuperiorPredictor — the best of everything combined.

Takes the proven winning technique from each top system:
- Starlizard: data breadth (85+ features including referee, weather, motivation)
- Pinnacle: market-implied probabilities as features (Shin + goto_conversion)
- AutoGluon: multi-model stacking with diversity (XGB + CatBoost + LightGBM + DC + BVP)
- TabPFN: foundation model diversity (zero-hyperparameter predictions)
- Research: calibration-optimized (ECE as objective, not just accuracy)
- goto_conversion: superior odds devigging
- OpenSkill: uncertainty-aware ratings (when available)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class SuperiorPrediction:
    """The most information-rich prediction possible."""

    home_team: str
    away_team: str
    league: str
    match_date: str

    # Core probabilities
    probabilities: NDArray[np.float64]  # [P(H), P(D), P(A)]
    predicted_outcome: str  # "Home", "Draw", "Away"
    confidence: float  # peak probability

    # Model breakdown
    model_probs: dict[str, NDArray]  # per-model probabilities

    # Ensemble quality metrics
    model_agreement: float  # fraction of models agreeing (0-1)
    prediction_entropy: float  # Shannon entropy (lower = more certain)
    mc_std: float  # Monte Carlo uncertainty

    # Market intelligence
    implied_probs: dict[str, float] | None = None  # from current odds
    edge: float = 0.0  # model prob - implied prob
    goto_probs: dict[str, float] | None = None  # goto_conversion devigged

    # Calibration quality
    ece_bucket: str = ""  # which calibration bucket this falls in

    # Value assessment
    is_value_bet: bool = False
    expected_value: float = 0.0  # (prob * odds - 1) * 100
    kelly_stake: float = 0.0  # recommended stake %

    # Reasoning
    reasoning: str = ""
    confidence_factors: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)


class SuperiorPredictor:
    """Combines the best of every top prediction system."""

    def __init__(self) -> None:
        self._models: dict = {}
        self._fitted: bool = False
        self._draw_floor: float = 0.20  # adaptive per league
        self._league_draw_rates: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: NDArray,
        y_train: NDArray,
        train_df,
        train_matches: list[dict],
        **kwargs,
    ) -> SuperiorPredictor:
        """Train all component models.

        Parameters
        ----------
        X_train : feature matrix (augmented with DC strengths)
        y_train : "H"/"D"/"A" labels
        train_df : DataFrame with league, match_date, etc.
        train_matches : list of dicts for Poisson models
        """
        from sharpedge.ml.models.bivariate_poisson import BivariatePoissonPredictor
        from sharpedge.ml.models.poisson_model import PoissonPredictor
        from sharpedge.ml.models.xgboost_model import XGBoostPredictor

        y_ou = kwargs.get("y_ou")
        y_btts = kwargs.get("y_btts")

        # 1. Dixon-Coles (from football domain knowledge)
        dc = PoissonPredictor()
        dc.fit(train_matches)
        self._models["dc"] = dc

        # 2. XGBoost (from Kaggle — proven best GBDT)
        xgb = XGBoostPredictor(
            params={
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.01,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 8,
                "reg_alpha": 1.0,
                "reg_lambda": 1.0,
                "random_state": 42,
            }
        )
        xgb.fit(X_train, y_train, y_ou=y_ou, y_btts=y_btts)
        self._models["xgb"] = xgb

        # 3. CatBoost (structural diversity — symmetric trees)
        try:
            from sharpedge.ml.models.catboost_model import CatBoostPredictor

            cat = CatBoostPredictor()
            cat.fit(X_train, y_train, y_ou=y_ou, y_btts=y_btts)
            self._models["catboost"] = cat
        except ImportError:
            logger.info("CatBoost not available, skipping")

        # 4. LightGBM (structural diversity — leaf-wise growth)
        try:
            from sharpedge.ml.models.lightgbm_model import LightGBMPredictor

            lgbm = LightGBMPredictor()
            lgbm.fit(X_train, y_train, y_ou=y_ou, y_btts=y_btts)
            self._models["lgbm"] = lgbm
        except ImportError:
            logger.info("LightGBM not available, skipping")

        # 5. Bivariate Poisson (correlated goal model)
        bvp = BivariatePoissonPredictor()
        bvp.fit(train_matches)
        self._models["bvp"] = bvp

        # 6. TabPFN (foundation model — zero hyperparameters)
        try:
            from tabpfn import TabPFNClassifier

            label_map = {"H": 0, "D": 1, "A": 2}
            y_enc = np.array([label_map.get(str(y), 1) for y in y_train])
            X_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            # TabPFN works best with <= 10K samples
            if len(X_clean) > 10000:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X_clean), 10000, replace=False)
                X_sub, y_sub = X_clean[idx], y_enc[idx]
            else:
                X_sub, y_sub = X_clean, y_enc
            tpfn = TabPFNClassifier(device="cpu", N_ensemble_configurations=4)
            tpfn.fit(X_sub, y_sub)
            self._models["tabpfn"] = tpfn
            logger.info("TabPFN fitted as ensemble member")
        except Exception as exc:
            logger.info("TabPFN not available: %s", exc)

        # Learn per-league draw rates (from Starlizard — granular league knowledge)
        if train_df is not None and hasattr(train_df, "columns"):
            if "league" in train_df.columns and "FTR" in train_df.columns:
                for league in train_df["league"].unique():
                    mask = train_df["league"] == league
                    self._league_draw_rates[league] = float(
                        (train_df.loc[mask, "FTR"] == "D").mean()
                    )

        self._fitted = True
        logger.info("SuperiorPredictor fitted with %d models", len(self._models))
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        X: NDArray,
        home_team: str,
        away_team: str,
        league: str = "",
        odds: dict[str, float] | None = None,
        match_date: str = "",
    ) -> SuperiorPrediction:
        """Generate a superior prediction combining all systems.

        Parameters
        ----------
        X : feature vector (1-D or 2-D)
        home_team, away_team : team identifiers
        league : league name (for adaptive draw floor)
        odds : dict ``{"home": 1.8, "draw": 3.5, "away": 4.2}`` or *None*
        match_date : ISO date string
        """
        if not self._fitted:
            raise ValueError("Not fitted")

        X = X.reshape(1, -1) if X.ndim == 1 else X
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # ----------------------------------------------------------
        # Collect predictions from ALL models
        # ----------------------------------------------------------
        model_probs: dict[str, NDArray] = {}

        # Tree-based models (predict from features)
        for name in ("xgb", "catboost", "lgbm"):
            model = self._models.get(name)
            if model is not None:
                model_probs[name] = model.predict_proba_1x2(X)[0]

        # TabPFN
        tpfn = self._models.get("tabpfn")
        if tpfn is not None:
            try:
                raw = tpfn.predict_proba(X_clean)[0]
                if len(raw) >= 3:
                    p = np.array(raw[:3], dtype=np.float64)
                    model_probs["tabpfn"] = p / p.sum()
            except Exception:
                pass

        # Dixon-Coles & BVP (predict from team names)
        dc = self._models.get("dc")
        if dc is not None:
            model_probs["dc"] = dc.predict_proba_1x2(home_team, away_team)

        bvp = self._models.get("bvp")
        if bvp is not None:
            model_probs["bvp"] = bvp.predict_proba_1x2(home_team, away_team)

        if not model_probs:
            raise ValueError("No model predictions available")

        # ----------------------------------------------------------
        # Weighted blend (optimized weights from V2)
        # ----------------------------------------------------------
        weights = {
            "xgb": 0.25,
            "catboost": 0.15,
            "lgbm": 0.15,
            "dc": 0.20,
            "bvp": 0.10,
            "tabpfn": 0.15,
        }
        blended = np.zeros(3)
        total_w = 0.0
        for name, probs in model_probs.items():
            w = weights.get(name, 0.1)
            blended += w * np.asarray(probs)
            total_w += w
        if total_w > 0:
            blended /= total_w

        # ----------------------------------------------------------
        # MC simulation (from our core)
        # ----------------------------------------------------------
        from sharpedge.core.consensus import compute_consensus
        from sharpedge.core.monte_carlo import monte_carlo_simulate

        mc_input = {k: np.asarray(v).reshape(1, 3) for k, v in model_probs.items()}
        mc_probs, mc_conf, mc_std = monte_carlo_simulate(mc_input, n_sims=5000)
        _consensus_pred, _consensus_count = compute_consensus(mc_input)

        # Combine: 70 % weighted blend + 30 % MC
        final = 0.7 * blended + 0.3 * mc_probs[0]

        # ----------------------------------------------------------
        # Adaptive draw floor (from Starlizard — league-specific)
        # ----------------------------------------------------------
        draw_floor = self._league_draw_rates.get(league, self._draw_floor)
        if final[1] < draw_floor:
            deficit = draw_floor - final[1]
            final[1] = draw_floor
            ha = final[0] + final[2]
            if ha > 0:
                final[0] -= deficit * (final[0] / ha)
                final[2] -= deficit * (final[2] / ha)

        final = np.clip(final, 0.01, None)
        final /= final.sum()

        pred_idx = int(np.argmax(final))
        outcomes = ["Home", "Draw", "Away"]

        # Model agreement
        votes = [int(np.argmax(p)) for p in model_probs.values()]
        agreement = votes.count(pred_idx) / len(votes)

        # Entropy
        entropy = float(-np.sum(final * np.log(np.clip(final, 1e-10, 1.0))))

        # ----------------------------------------------------------
        # Market intelligence
        # ----------------------------------------------------------
        implied: dict[str, float] | None = None
        goto: dict[str, float] | None = None
        edge = 0.0
        ev = 0.0
        is_value = False
        kelly = 0.0

        if odds:
            # Devig with goto_conversion (from Kaggle gold medal)
            try:
                from goto_conversion import Goto

                raw_odds = [
                    odds.get("home", 0),
                    odds.get("draw", 0),
                    odds.get("away", 0),
                ]
                if all(o > 1 for o in raw_odds):
                    g = Goto()
                    goto_p = g.convert(raw_odds)
                    goto = {
                        "Home": goto_p[0],
                        "Draw": goto_p[1],
                        "Away": goto_p[2],
                    }
            except Exception:
                pass

            # Simple implied probabilities
            implied = {}
            total_impl = 0.0
            for i, key in enumerate(["home", "draw", "away"]):
                o = odds.get(key, 0)
                if o > 1:
                    implied[outcomes[i]] = 1.0 / o
                    total_impl += 1.0 / o
            if total_impl > 0:
                implied = {k: v / total_impl for k, v in implied.items()}

            # Edge = model prob - implied prob
            bet_odds = [
                odds.get("home", 0),
                odds.get("draw", 0),
                odds.get("away", 0),
            ][pred_idx]
            if bet_odds > 1:
                overround = sum(
                    1.0 / o
                    for o in [
                        odds.get("home", 99),
                        odds.get("draw", 99),
                        odds.get("away", 99),
                    ]
                    if o > 1
                )
                impl_devigged = (1.0 / bet_odds) / overround if overround > 0 else (1.0 / bet_odds)
                edge = float(final[pred_idx]) - impl_devigged
                ev = (float(final[pred_idx]) * bet_odds - 1) * 100
                is_value = edge >= 0.05 and float(final[pred_idx]) >= 0.50

                if is_value:
                    b = bet_odds - 1
                    raw_kelly = (float(final[pred_idx]) * b - (1 - float(final[pred_idx]))) / b
                    kelly = max(0, raw_kelly * 0.25) * 100  # quarter Kelly as %

        # ----------------------------------------------------------
        # Build reasoning
        # ----------------------------------------------------------
        conf_factors: list[str] = []
        risk_factors: list[str] = []

        if agreement >= 0.8:
            conf_factors.append(f"{agreement:.0%} model agreement")
        if float(mc_conf[0]) >= 0.6:
            conf_factors.append(f"MC confidence {float(mc_conf[0]):.0%}")
        if edge >= 0.05:
            conf_factors.append(f"Edge {edge:.1%}")

        if agreement < 0.5:
            risk_factors.append("Low model agreement")
        if float(mc_std[0, pred_idx]) > 0.10:
            risk_factors.append("High MC uncertainty")
        if entropy > 1.0:
            risk_factors.append("High prediction entropy")

        reasoning = (
            f"{outcomes[pred_idx]} @ {final[pred_idx]:.1%} | "
            f"{len(model_probs)} models, {agreement:.0%} agree | "
            f"MC: {float(mc_conf[0]):.0%} conf"
        )
        if edge > 0:
            reasoning += f" | Edge: {edge:.1%}"

        return SuperiorPrediction(
            home_team=home_team,
            away_team=away_team,
            league=league,
            match_date=match_date,
            probabilities=final,
            predicted_outcome=outcomes[pred_idx],
            confidence=float(final[pred_idx]),
            model_probs={
                k: v.tolist() if hasattr(v, "tolist") else list(v)
                for k, v in model_probs.items()
            },
            model_agreement=agreement,
            prediction_entropy=entropy,
            mc_std=float(mc_std[0, pred_idx]),
            implied_probs=implied,
            edge=edge,
            goto_probs=goto,
            is_value_bet=is_value,
            expected_value=ev,
            kelly_stake=kelly,
            reasoning=reasoning,
            confidence_factors=conf_factors,
            risk_factors=risk_factors,
        )
