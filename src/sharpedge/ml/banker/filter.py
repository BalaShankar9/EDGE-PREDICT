"""5-stage banker tip filter pipeline.

Selects high-confidence, high-value picks from model predictions.

Stage 1: Minimum Confidence (model_prob >= 0.70)
Stage 2: Model Agreement (spread <= 0.08)
Stage 3: Value Edge (edge >= 0.05)
Stage 4: Meta-Model Confirmation (>= 2 competitor sites agree)
Stage 5: Risk Flag Check (no critical flags)
"""
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Pick:
    """A filtered betting pick."""
    match_id: str
    home_team: str
    away_team: str
    league: str
    match_date: str
    market: str           # "1x2_home", "1x2_away", "1x2_draw", "over_25", "btts_yes"
    model_prob: float
    model_spread: float   # spread across ensemble models
    best_odds: float
    bookmaker: str
    implied_prob: float   # 1 / best_odds
    edge: float           # model_prob - implied_prob
    tier: str = ""        # Set by TierAssigner
    meta_agreement: int = 0
    risk_flags: list[str] = field(default_factory=list)
    confidence_factors: list[str] = field(default_factory=list)


class BankerFilter:
    """5-stage filter that selects high-confidence, high-value picks."""

    def __init__(
        self,
        min_confidence: float = 0.70,
        max_spread: float = 0.08,
        min_edge: float = 0.05,
        min_meta_agreement: int = 2,
    ):
        self.min_confidence = min_confidence
        self.max_spread = max_spread
        self.min_edge = min_edge
        self.min_meta_agreement = min_meta_agreement

    def filter(self, predictions: list[dict]) -> list[Pick]:
        """Apply all 5 filters in sequence.

        Parameters
        ----------
        predictions : list of dicts with keys:
            match_id, home_team, away_team, league, match_date, market,
            model_prob, model_spread, best_odds, bookmaker,
            meta_agreement, risk_flags

        Returns
        -------
        list[Pick] : filtered picks
        """
        picks = []

        for pred in predictions:
            implied_prob = 1.0 / pred["best_odds"] if pred["best_odds"] > 0 else 1.0
            edge = pred["model_prob"] - implied_prob
            confidence_factors = []

            # Stage 1: Minimum confidence
            if pred["model_prob"] < self.min_confidence:
                continue
            confidence_factors.append(f"prob={pred['model_prob']:.2f}")

            # Stage 2: Model agreement (low spread)
            if pred["model_spread"] > self.max_spread:
                continue
            confidence_factors.append(f"spread={pred['model_spread']:.3f}")

            # Stage 3: Value edge
            if edge < self.min_edge:
                continue
            confidence_factors.append(f"edge={edge:.3f}")

            # Stage 4: Meta-model confirmation
            if pred.get("meta_agreement", 0) < self.min_meta_agreement:
                continue
            confidence_factors.append(f"meta={pred['meta_agreement']}")

            # Stage 5: Risk flag check
            risk_flags = pred.get("risk_flags", [])
            critical_flags = [f for f in risk_flags if f.startswith("CRITICAL")]
            if critical_flags:
                continue

            pick = Pick(
                match_id=pred["match_id"],
                home_team=pred["home_team"],
                away_team=pred["away_team"],
                league=pred["league"],
                match_date=pred["match_date"],
                market=pred["market"],
                model_prob=pred["model_prob"],
                model_spread=pred["model_spread"],
                best_odds=pred["best_odds"],
                bookmaker=pred.get("bookmaker", "unknown"),
                implied_prob=implied_prob,
                edge=edge,
                meta_agreement=pred.get("meta_agreement", 0),
                risk_flags=risk_flags,
                confidence_factors=confidence_factors,
            )
            picks.append(pick)

        logger.info(f"Banker filter: {len(predictions)} -> {len(picks)} picks")
        return picks
