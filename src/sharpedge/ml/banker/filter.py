"""Confidence-based banker filter pipeline.

Selects high-confidence picks based on empirically-validated thresholds.

Empirical accuracy by confidence threshold (walk-forward validation, 3504 matches):
  >= 0.55: 67.7% acc (45% of matches)
  >= 0.60: 72.3% acc (33% of matches)
  >= 0.65: 75.7% acc (24% of matches)
  >= 0.70: 77.0% acc (21% of matches)
  >= 0.75: 82.1% acc (9% of matches)
  >= 0.80: 83.2% acc (4% of matches)
  >= 0.85: 89.4% acc (2% of matches)

Stages:
  Stage 1: Market whitelist (only backtest-profitable markets)
  Stage 2: Odds ceiling (reject longshots > MAX_ODDS)
  Stage 3: Per-market confidence (backtest-optimized thresholds)
  Stage 4: Value edge (model_prob - implied_prob >= min_edge)
  Stage 5: Risk flag check (no CRITICAL flags)
"""
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Markets proven profitable in walk-forward backtest
PROFITABLE_MARKETS = frozenset({
    "1x2_home", "1x2_away", "1x2_draw",
    "over_25", "dc_x2",
})

# Per-market confidence thresholds (backtest-optimized)
MARKET_THRESHOLDS = {
    "1x2_home": 0.58,   # Home needs higher bar (52.9% acc, -7.6% ROI at low thresholds)
    "1x2_away": 0.50,   # Away is strongest (+3.9% ROI)
    "1x2_draw": 0.50,   # Rare but profitable when predicted
    "over_25": 0.55,    # O/U needs moderate confidence
    "dc_x2": 0.65,      # DC X2 needs high confidence (+0.3% ROI at 65%+)
}

# Maximum odds (backtest: odds > 2.50 -> -11.3% ROI)
MAX_ODDS = 2.50


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
    model_spread: float   # spread between top two probabilities
    best_odds: float
    bookmaker: str
    implied_prob: float   # 1 / best_odds
    edge: float           # model_prob - implied_prob
    tier: str = ""        # Set by TierAssigner
    meta_agreement: int = 0
    pick_odds: float = 0.0       # Odds at time of pick (for CLV tracking)
    closing_odds: float = 0.0    # Closing odds (filled post-match for CLV)
    clv_pct: float = 0.0         # CLV% = (closing_implied/pick_implied - 1) * 100
    risk_flags: list[str] = field(default_factory=list)
    confidence_factors: list[str] = field(default_factory=list)


class BankerFilter:
    """Confidence-based filter using empirically-validated thresholds."""

    def __init__(
        self,
        min_confidence: float = 0.50,
        min_edge: float = 0.05,
        max_odds: float = MAX_ODDS,
    ):
        self.min_confidence = min_confidence
        self.min_edge = min_edge
        self.max_odds = max_odds

    def filter(self, predictions: list[dict]) -> list[Pick]:
        """Apply backtest-proven filters in order.

        Stages
        ------
        1. Market whitelist  -- reject if market not in PROFITABLE_MARKETS
        2. Odds ceiling      -- reject if best_odds > max_odds or <= 0
        3. Per-market conf.  -- use max(MARKET_THRESHOLDS[market], min_confidence)
        4. Value edge        -- model_prob - implied_prob >= min_edge
        5. Risk flag check   -- reject if any CRITICAL flags

        Parameters
        ----------
        predictions : list of dicts with keys:
            match_id, home_team, away_team, league, match_date, market,
            model_prob, model_spread, best_odds, bookmaker,
            meta_agreement (optional), risk_flags (optional)

        Returns
        -------
        list[Pick] : filtered picks
        """
        picks = []

        for pred in predictions:
            market = pred["market"]
            confidence_factors = []

            # Stage 1: Market whitelist
            if market not in PROFITABLE_MARKETS:
                continue

            # Stage 2: Odds ceiling
            best_odds = pred["best_odds"]
            if best_odds <= 0 or best_odds > self.max_odds:
                continue

            # Stage 3: Per-market confidence threshold
            market_threshold = MARKET_THRESHOLDS.get(market, self.min_confidence)
            effective_threshold = max(market_threshold, self.min_confidence)
            if pred["model_prob"] < effective_threshold:
                continue
            confidence_factors.append(f"prob={pred['model_prob']:.2f}")

            # Stage 4: Value edge
            implied_prob = 1.0 / best_odds
            edge = pred["model_prob"] - implied_prob
            if edge < self.min_edge:
                continue
            confidence_factors.append(f"edge={edge:.3f}")

            # Stage 5: Risk flag check
            risk_flags = pred.get("risk_flags", [])
            critical_flags = [f for f in risk_flags if f.startswith("CRITICAL")]
            if critical_flags:
                continue

            model_spread = pred.get("model_spread", 0.0)
            confidence_factors.append(f"spread={model_spread:.3f}")

            pick = Pick(
                match_id=pred["match_id"],
                home_team=pred["home_team"],
                away_team=pred["away_team"],
                league=pred["league"],
                match_date=pred["match_date"],
                market=market,
                model_prob=pred["model_prob"],
                model_spread=model_spread,
                best_odds=best_odds,
                bookmaker=pred.get("bookmaker", "unknown"),
                implied_prob=implied_prob,
                edge=edge,
                pick_odds=best_odds,  # Record for CLV tracking
                meta_agreement=pred.get("meta_agreement", 0),
                risk_flags=risk_flags,
                confidence_factors=confidence_factors,
            )
            picks.append(pick)

        logger.info(f"Banker filter: {len(predictions)} -> {len(picks)} picks")
        return picks
