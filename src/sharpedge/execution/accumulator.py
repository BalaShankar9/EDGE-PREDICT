"""Smart Accumulator (Parlay) Builder.

Builds scientifically-constructed accumulators from filtered single-match picks.
Uses correlation-adjusted probability math and EV-ranked selection.

Key principles:
- Accumulator EV = product of individual EVs ONLY when legs are independent.
  Real correlations (same league, same country, correlated markets) reduce true
  combined probability below the naive product, so we apply a correlation penalty.
- Kelly sizing for accas uses a 50% haircut vs singles: parlays have higher
  variance and longer losing streaks even at positive EV.
- Chain-of-trust: the acca is only as good as its weakest leg. Ranking penalises
  low-confidence anchors.

Correlation priors (empirical estimates from sports betting research):
  - Same league:               0.30  (league-wide form / referee bias)
  - Same country, diff league: 0.15  (national weather / rule variations)
  - Same market type:          0.10  (systematic model overconfidence per market)
  - Home + Over in same match: 0.40  (high-scoring home wins drive both)
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Optional

from sharpedge.ml.banker.filter import Pick
from sharpedge.execution.staking import AntifragileStaking
from sharpedge.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Correlation priors (empirical, from sports betting literature)
# ---------------------------------------------------------------------------
CORR_SAME_LEAGUE = 0.30
CORR_SAME_COUNTRY = 0.15
CORR_SAME_MARKET_TYPE = 0.10
CORR_HOME_AND_OVER_SAME_MATCH = 0.40

# Accumulator sizing: halve normal Kelly to account for parlay variance
ACCA_KELLY_HAIRCUT = 0.50

# Leg count bounds
ABS_MAX_LEGS = 4
ABS_MIN_LEGS = 2


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AccumulatorLeg:
    """A single pick within an accumulator, annotated with its probability contribution."""
    pick: Pick
    contribution_to_prob: float  # This leg's model_prob, isolated for inspection


@dataclass
class Accumulator:
    """A multi-leg accumulator (parlay) bet."""
    legs: list[AccumulatorLeg]
    combined_odds: float          # Product of all leg odds (decimal)
    combined_prob: float          # Correlation-adjusted combined probability
    edge: float                   # Average edge across legs
    expected_value: float         # EV = combined_prob * combined_odds - 1
    stake_pct: float              # Recommended stake as fraction of bankroll
    correlation_score: float      # Max pairwise correlation (0=independent, 1=identical)
    reasoning: str                # Human-readable explanation

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    @property
    def label(self) -> str:
        labels = {2: "Double", 3: "Treble", 4: "Four-Fold"}
        return labels.get(self.n_legs, f"{self.n_legs}-Fold")


# ---------------------------------------------------------------------------
# Country extraction helper
# ---------------------------------------------------------------------------

def _extract_country(league: str) -> str:
    """Best-effort extraction of country from a league string.

    Expects formats like 'England - Premier League', 'Spain - La Liga',
    'Champions League', etc. Falls back to the full league name if no
    separator is present.
    """
    if " - " in league:
        return league.split(" - ")[0].strip().lower()
    return league.strip().lower()


def _market_base(market: str) -> str:
    """Return the market family: '1x2', 'over_under', 'dc', etc."""
    if market.startswith("1x2"):
        return "1x2"
    if market.startswith("over") or market.startswith("under"):
        return "over_under"
    if market.startswith("dc"):
        return "dc"
    return market


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AccumulatorBuilder:
    """Builds, ranks, and formats smart accumulators from banker picks.

    Usage
    -----
    builder = AccumulatorBuilder(staking_engine)
    accas = builder.build_daily_accas(picks, n_accas=3)
    message = builder.format_telegram(accas)
    """

    def __init__(self, staking: Optional[AntifragileStaking] = None) -> None:
        self.staking = staking or AntifragileStaking(
            bankroll=settings.bankroll_initial,
            kelly_fraction=settings.kelly_fraction,
            max_bet_pct=settings.max_bet_pct,
            max_daily_pct=settings.max_daily_pct,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_accumulators(
        self,
        picks: list[Pick],
        max_legs: int = ABS_MAX_LEGS,
        min_legs: int = ABS_MIN_LEGS,
    ) -> list[Accumulator]:
        """Generate all valid accumulators from *picks* and rank by EV.

        Parameters
        ----------
        picks:
            Filtered banker picks from BankerFilter.
        max_legs:
            Maximum number of legs (hard cap: 4).
        min_legs:
            Minimum number of legs (hard floor: 2).

        Returns
        -------
        List of Accumulator objects ranked by adjusted EV (best first).
        All negative-EV combinations are dropped.
        """
        max_legs = min(max_legs, ABS_MAX_LEGS)
        min_legs = max(min_legs, ABS_MIN_LEGS)

        if len(picks) < min_legs:
            logger.warning(
                "AccumulatorBuilder: only %d picks available (need >= %d for accas).",
                len(picks), min_legs,
            )
            return []

        candidates: list[Accumulator] = []

        for n in range(min_legs, max_legs + 1):
            for combo in itertools.combinations(picks, n):
                acca = self._try_build_acca(list(combo))
                if acca is not None:
                    candidates.append(acca)

        logger.info(
            "AccumulatorBuilder: %d raw combinations -> %d valid accas before ranking.",
            sum(
                len(list(itertools.combinations(picks, n)))
                for n in range(min_legs, max_legs + 1)
            ),
            len(candidates),
        )

        return self.rank_accumulators(candidates)

    def compute_correlation(self, pick_a: Pick, pick_b: Pick) -> float:
        """Estimate pairwise correlation between two picks.

        Returns a value in [0, 1]:
          0.0 = statistically independent (different countries, different markets)
          1.0 = perfectly correlated (should never happen in a valid acca)

        Correlations are additive but capped at 1.0. The priors are empirical
        estimates from sports betting research; they are intentionally conservative
        (i.e. slightly high) to avoid overestimating acca value.
        """
        # Same match: never combine (enforced at combination level, but guard here)
        if pick_a.match_id == pick_b.match_id:
            # Exception: home result + over/under is a known correlated combo
            markets = {pick_a.market, pick_b.market}
            is_home = any(m == "1x2_home" for m in markets)
            is_over = any(m.startswith("over") for m in markets)
            if is_home and is_over:
                return CORR_HOME_AND_OVER_SAME_MATCH
            # Two picks from the same match that aren't the above combo: full correlation
            return 1.0

        corr = 0.0

        # Same league (most impactful: shared referee pool, scheduling, travel)
        if pick_a.league == pick_b.league:
            corr += CORR_SAME_LEAGUE
        else:
            # Same country, different competition
            country_a = _extract_country(pick_a.league)
            country_b = _extract_country(pick_b.league)
            if country_a == country_b:
                corr += CORR_SAME_COUNTRY

        # Same market family: systematic model bias can affect multiple picks
        # the same way (e.g. model consistently under/over-estimates home wins)
        if _market_base(pick_a.market) == _market_base(pick_b.market):
            corr += CORR_SAME_MARKET_TYPE

        return min(corr, 1.0)

    def compute_combined_probability(
        self,
        legs: list[Pick],
        correlations: dict[tuple[int, int], float],
    ) -> float:
        """Correlation-adjusted combined probability.

        Naive approach: prod(p_i)  — assumes full independence.
        We apply a downward penalty proportional to the worst pairwise
        correlation in the combo.

        Formula
        -------
            combined_prob = prod(p_i) * correlation_penalty
            correlation_penalty = 1 - max_corr * 0.5

        The 0.5 weight is conservative: a max correlation of 1.0 halves the
        naive probability, not zeroing it, which guards against extreme
        over-penalisation when we have only one correlated pair in a 4-leg acca.
        """
        # Naive product of individual model probabilities
        naive_product = 1.0
        for leg in legs:
            naive_product *= leg.model_prob

        if not correlations:
            return naive_product

        max_corr = max(correlations.values()) if correlations else 0.0
        # Penalty: 1.0 at max_corr=0, 0.5 at max_corr=1.0
        correlation_penalty = 1.0 - max_corr * 0.5

        return naive_product * correlation_penalty

    def compute_expected_value(self, combined_prob: float, combined_odds: float) -> float:
        """Expected value for a unit stake on the accumulator.

        EV = (combined_prob * combined_odds) - 1

        Interpretation:
          EV > 0: profitable in the long run (positive expectation)
          EV = 0: break-even
          EV < 0: losing bet
        """
        return combined_prob * combined_odds - 1.0

    def rank_accumulators(self, accas: list[Accumulator]) -> list[Accumulator]:
        """Filter and rank accumulators by risk-adjusted EV.

        Scoring function
        ----------------
            score = EV * (1 - correlation_score) * leg_confidence_min

        Rationale:
          - EV: raw expected profit per unit staked.
          - (1 - correlation_score): discount for correlated legs; independent
            accas score higher.
          - leg_confidence_min: the weakest-link principle — a 4-fold with one
            50% leg is fundamentally less reliable than a tight treble.

        Negative-EV accas are removed before ranking.
        """
        positive_ev = [a for a in accas if a.expected_value > 0]

        def score(acca: Accumulator) -> float:
            leg_confidence_min = min(leg.pick.model_prob for leg in acca.legs)
            return acca.expected_value * (1.0 - acca.correlation_score) * leg_confidence_min

        ranked = sorted(positive_ev, key=score, reverse=True)
        logger.debug(
            "AccumulatorBuilder.rank_accumulators: %d candidates -> %d positive-EV -> ranked.",
            len(accas), len(positive_ev),
        )
        return ranked

    def build_daily_accas(
        self,
        picks: list[Pick],
        n_accas: int = 3,
    ) -> list[Accumulator]:
        """Build the top N accumulators for a single betting day.

        Guarantees (where enough picks exist):
          - At least 1 double   (2-leg)
          - At least 1 treble   (3-leg)
          - At least 1 four-fold (4-leg)  — only if >= 4 eligible picks

        The stake for every acca is halved relative to the normal Kelly
        recommendation to account for the higher variance of parlays.

        Parameters
        ----------
        picks:
            Filtered banker picks for today.
        n_accas:
            Maximum number of accumulators to return (default 3).
        """
        all_accas = self.build_accumulators(picks, max_legs=ABS_MAX_LEGS, min_legs=ABS_MIN_LEGS)

        if not all_accas:
            logger.info("AccumulatorBuilder.build_daily_accas: no valid accas generated.")
            return []

        selected: list[Accumulator] = []
        seen_leg_sets: set[frozenset[str]] = set()

        # Helper: try to pull the best acca of a given leg count
        def pick_best_by_legs(n: int) -> Optional[Accumulator]:
            for acca in all_accas:
                if acca.n_legs != n:
                    continue
                leg_key = frozenset(leg.pick.match_id + leg.pick.market for leg in acca.legs)
                if leg_key not in seen_leg_sets:
                    return acca
            return None

        # Guarantee one of each type in priority order
        for target_legs in (2, 3, 4):
            if len(picks) < target_legs:
                continue
            best = pick_best_by_legs(target_legs)
            if best is not None:
                leg_key = frozenset(leg.pick.match_id + leg.pick.market for leg in best.legs)
                seen_leg_sets.add(leg_key)
                selected.append(best)

        # Fill remaining slots with the best available unique accas
        for acca in all_accas:
            if len(selected) >= n_accas:
                break
            leg_key = frozenset(leg.pick.match_id + leg.pick.market for leg in acca.legs)
            if leg_key not in seen_leg_sets:
                seen_leg_sets.add(leg_key)
                selected.append(acca)

        logger.info(
            "AccumulatorBuilder.build_daily_accas: selected %d accas (%s).",
            len(selected),
            ", ".join(a.label for a in selected),
        )
        return selected

    def format_telegram(self, accas: list[Accumulator]) -> str:
        """Format accumulator recommendations for Telegram broadcast.

        Produces a single message covering all provided accas, ordered by EV.
        Each acca block shows all legs, combined odds, EV, stake, and a brief
        rationale.
        """
        if not accas:
            return "\U0001f4ca SharpEdge Accumulators\n\nNo qualifying accumulators today."

        lines: list[str] = [
            "\U0001f3b2 SharpEdge Accumulators",
            f"Top {len(accas)} Smart Parlays for Today\n",
        ]

        for idx, acca in enumerate(accas, start=1):
            ev_pct = acca.expected_value * 100
            ev_sign = "+" if ev_pct >= 0 else ""
            stake_pct_display = acca.stake_pct * 100
            corr_display = acca.correlation_score * 100

            lines.append(f"\U0001f4cc Acca {idx}: {acca.label}")
            lines.append(f"\U0001f3af Combined Odds: {acca.combined_odds:.2f}")
            lines.append(f"\U0001f4c8 EV: {ev_sign}{ev_pct:.1f}%")
            lines.append(f"\U0001f4b0 Stake: {stake_pct_display:.2f}% bankroll")
            lines.append(f"\U0001f517 Correlation: {corr_display:.0f}%")
            lines.append("")

            for i, acca_leg in enumerate(acca.legs, start=1):
                p = acca_leg.pick
                market_display = p.market.upper().replace("_", " ")
                edge_pct = p.edge * 100
                prob_pct = p.model_prob * 100
                lines.append(
                    f"  Leg {i}: {p.home_team} vs {p.away_team}"
                )
                lines.append(
                    f"         {p.league} | {p.match_date}"
                )
                lines.append(
                    f"         \u2714 {market_display} @ {p.best_odds:.2f} "
                    f"({prob_pct:.0f}% | +{edge_pct:.1f}% edge)"
                )
                lines.append("")

            lines.append(f"\U0001f4dd {acca.reasoning}")
            lines.append("\u2500" * 30)
            lines.append("")

        lines.append("\u26a0\ufe0f Accumulators carry higher variance. Stake responsibly.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_build_acca(self, picks: list[Pick]) -> Optional[Accumulator]:
        """Attempt to build a valid Accumulator from a pick combination.

        Validation rules (any failure returns None):
          1. No two legs from the same match (match_id uniqueness).
          2. No two legs from the same league.
          3. If two legs share the same country (but different leagues), their
             markets must differ — avoids stacking correlated market types
             within the same national football ecosystem.

        After validation, computes correlation, combined probability, combined
        odds, EV, and Kelly-derived stake (with ACCA_KELLY_HAIRCUT applied).
        """
        # Rule 1: no two picks from the same match
        match_ids = [p.match_id for p in picks]
        if len(match_ids) != len(set(match_ids)):
            return None

        # Rule 2: no two picks from the same league
        leagues = [p.league for p in picks]
        if len(leagues) != len(set(leagues)):
            return None

        # Rule 3: same country + same market family must not co-exist
        countries = [_extract_country(p.league) for p in picks]
        market_families = [_market_base(p.market) for p in picks]
        for i in range(len(picks)):
            for j in range(i + 1, len(picks)):
                if countries[i] == countries[j] and market_families[i] == market_families[j]:
                    return None

        # Build pairwise correlation map
        correlations: dict[tuple[int, int], float] = {}
        for i in range(len(picks)):
            for j in range(i + 1, len(picks)):
                c = self.compute_correlation(picks[i], picks[j])
                correlations[(i, j)] = c

        max_corr = max(correlations.values()) if correlations else 0.0

        # Combined probability (correlation-adjusted)
        combined_prob = self.compute_combined_probability(picks, correlations)

        # Combined odds: product of all leg odds
        combined_odds = 1.0
        for p in picks:
            combined_odds *= p.best_odds

        # Expected value
        ev = self.compute_expected_value(combined_prob, combined_odds)

        # Average edge across legs
        avg_edge = sum(p.edge for p in picks) / len(picks)

        # Kelly sizing for the accumulator
        # We treat the acca as a single bet: prob = combined_prob, odds = combined_odds
        # Then apply ACCA_KELLY_HAIRCUT on top of the staking engine's Kelly fraction
        raw_kelly = self.staking.kelly_stake(combined_prob, combined_odds)
        stake_pct = (
            raw_kelly
            * self.staking.kelly_fraction
            * ACCA_KELLY_HAIRCUT
        )
        stake_pct = min(stake_pct, self.staking.max_bet_pct * ACCA_KELLY_HAIRCUT)

        # Volume confidence: average model spread (higher = more decisive models)
        volume_confidence = sum(p.model_spread for p in picks) / len(picks)

        # Build score components for reasoning
        combination_score = avg_edge * (1.0 - max_corr) * volume_confidence

        legs_str = " | ".join(
            f"{p.home_team} vs {p.away_team} [{p.market}]" for p in picks
        )
        reasoning = (
            f"{len(picks)}-leg acca | "
            f"EV={ev * 100:.1f}% | "
            f"avg edge={avg_edge * 100:.1f}% | "
            f"max corr={max_corr:.2f} | "
            f"vol_conf={volume_confidence:.3f} | "
            f"score={combination_score:.4f} | "
            f"legs: {legs_str}"
        )

        legs = [
            AccumulatorLeg(pick=p, contribution_to_prob=p.model_prob)
            for p in picks
        ]

        return Accumulator(
            legs=legs,
            combined_odds=combined_odds,
            combined_prob=combined_prob,
            edge=avg_edge,
            expected_value=ev,
            stake_pct=stake_pct,
            correlation_score=max_corr,
            reasoning=reasoning,
        )
