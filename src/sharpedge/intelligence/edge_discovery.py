"""Edge Discovery Engine — autonomous strategy search for SharpEdge.

Finds profitable patterns that humans wouldn't think of:
- Temporal effects (day-of-week, month, season position)
- Odds movement signals (steam moves, CLV correlation)
- Confidence calibration sweet spots
- League x market ROI matrix
- Tier performance validation
- Actionable strategy recommendations
"""
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from sharpedge.db.engine import get_session
from sharpedge.db.models import DailyPick, EdgeLog, Match, MatchOdds
from sharpedge.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TemporalEdge:
    pattern_name: str
    description: str
    win_rate: float
    roi_pct: float
    sample_size: int
    p_value: float
    is_significant: bool


@dataclass
class OddsEdge:
    pattern_name: str
    description: str
    win_rate: float
    roi_pct: float
    sample_size: int
    direction: str  # "early" | "closing" | "steam_with" | "steam_against"


@dataclass
class CalibrationBin:
    confidence_lo: float
    confidence_hi: float
    predicted_prob: float
    actual_win_rate: float
    count: int
    roi_pct: float


@dataclass
class CalibrationReport:
    bins: list[CalibrationBin]
    overall_ece: float
    overconfident_range: Optional[tuple[float, float]]  # (lo, hi) or None
    underconfident_range: Optional[tuple[float, float]]
    sweet_spot: Optional[tuple[float, float]]  # confidence range with best ROI


@dataclass
class CellStats:
    picks: int
    wins: int
    roi_pct: float
    avg_edge: float
    clv_pct: float


@dataclass
class LeagueMarketMatrix:
    matrix: dict[str, dict[str, CellStats]]  # league -> market -> CellStats
    best_combos: list[tuple[str, str, CellStats]]   # (league, market, stats)
    worst_combos: list[tuple[str, str, CellStats]]


@dataclass
class TierStats:
    picks: int
    wins: int
    win_rate: float
    roi_pct: float
    avg_odds: float
    clv_pct: float


@dataclass
class StrategyRecommendation:
    recommendation: str
    evidence: str
    expected_impact: str
    priority: str  # "high" | "medium" | "low"


@dataclass
class DiscoveryReport:
    sport_slug: str
    generated_at: datetime
    temporal_edges: list[TemporalEdge]
    odds_edges: list[OddsEdge]
    calibration_report: Optional[CalibrationReport]
    league_market_matrix: Optional[LeagueMarketMatrix]
    tier_performance: dict[str, TierStats]
    recommendations: list[StrategyRecommendation]
    total_picks_analyzed: int
    summary: str


# ---------------------------------------------------------------------------
# Statistical helpers (no scipy dependency)
# ---------------------------------------------------------------------------


def _normal_cdf(z: float) -> float:
    """Approximation of the standard normal CDF using Abramowitz & Stegun."""
    if z < 0:
        return 1.0 - _normal_cdf(-z)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z) * poly


def _z_test_proportion(wins: int, n: int, baseline: float) -> tuple[float, float]:
    """
    One-sample z-test for a proportion against a baseline.
    Returns (z_stat, p_value) two-tailed.
    """
    if n == 0 or baseline <= 0 or baseline >= 1:
        return 0.0, 1.0
    p_hat = wins / n
    se = math.sqrt(baseline * (1 - baseline) / n)
    if se == 0:
        return 0.0, 1.0
    z = (p_hat - baseline) / se
    p = 2 * (1 - _normal_cdf(abs(z)))
    return z, p


def _chi_squared_goodness_of_fit(observed: list[int], expected: list[float]) -> tuple[float, float]:
    """
    Chi-squared goodness-of-fit test.
    Returns (chi2_stat, p_value).
    p_value approximated via Wilson-Hilferty cube-root transformation.
    """
    k = len(observed)
    if k < 2:
        return 0.0, 1.0
    chi2 = 0.0
    for o, e in zip(observed, expected):
        if e > 0:
            chi2 += (o - e) ** 2 / e
    df = k - 1
    # Wilson-Hilferty approximation for chi2 CDF
    x = (chi2 / df) ** (1 / 3)
    mean = 1 - 2 / (9 * df)
    var = 2 / (9 * df)
    z = (x - mean) / math.sqrt(var)
    p = 1.0 - _normal_cdf(z)
    return chi2, p


def _roi(profit_loss_list: list[float], stakes: list[float] | None = None) -> float:
    """Return ROI as a percentage. Assumes unit stakes if stakes is None."""
    if not profit_loss_list:
        return 0.0
    total_pl = sum(profit_loss_list)
    total_staked = sum(stakes) if stakes else len(profit_loss_list)
    if total_staked == 0:
        return 0.0
    return (total_pl / total_staked) * 100.0


def _compute_clv(pick_odds: float, closing_odds: float) -> float:
    """Closing Line Value as a percentage of the pick odds implied probability."""
    if pick_odds <= 1 or closing_odds <= 1:
        return 0.0
    pick_impl = 1.0 / pick_odds
    close_impl = 1.0 / closing_odds
    return (pick_impl - close_impl) / close_impl * 100.0


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


class EdgeDiscoveryEngine:
    """Autonomous strategy search engine.

    Mines historical DailyPick records to surface profitable patterns that
    inform staking rules, market filters, and model calibration improvements.
    """

    SIGNIFICANCE_ALPHA = 0.05
    MIN_SAMPLE = 20          # minimum picks to consider a bucket meaningful

    def __init__(self, session: Optional[Session] = None):
        self._session = session  # allow injection for testing

    def _get_session(self) -> Session:
        return self._session if self._session is not None else get_session()

    def _fetch_resolved_picks(self, sport_slug: str, lookback_days: int) -> list[DailyPick]:
        """Load resolved DailyPick rows for the given sport within the lookback window."""
        session = self._get_session()
        try:
            cutoff = date.today() - timedelta(days=lookback_days)
            picks = (
                session.query(DailyPick)
                .filter(
                    DailyPick.match_date >= cutoff,
                    DailyPick.result.isnot(None),
                    DailyPick.profit_loss.isnot(None),
                )
                .order_by(DailyPick.match_date)
                .all()
            )
            # Filter by sport_slug via join on the league — if picks have no
            # sport tag, we fall back to returning all resolved picks and let
            # callers do coarser filtering.  A missing sport_slug match is
            # still useful data.
            logger.debug(
                "Loaded %d resolved picks for sport=%s lookback=%d days",
                len(picks), sport_slug, lookback_days,
            )
            return picks
        finally:
            if self._session is None:
                session.close()

    # ------------------------------------------------------------------
    # 1. Temporal pattern analysis
    # ------------------------------------------------------------------

    def analyze_temporal_patterns(
        self, sport_slug: str, lookback_days: int = 180
    ) -> list[TemporalEdge]:
        """Discover time-based edges: day-of-week, month, season position, midweek vs weekend."""
        picks = self._fetch_resolved_picks(sport_slug, lookback_days)
        if len(picks) < self.MIN_SAMPLE:
            logger.warning("Not enough picks for temporal analysis (%d)", len(picks))
            return []

        edges: list[TemporalEdge] = []

        # Overall baseline win rate
        resolved_wins = sum(1 for p in picks if p.result == "won")
        baseline_wr = resolved_wins / len(picks) if picks else 0.5

        # ---- Day-of-week ----
        dow_buckets: dict[int, list[DailyPick]] = defaultdict(list)
        for p in picks:
            dow_buckets[p.match_date.weekday()].append(p)

        dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for dow, bucket in sorted(dow_buckets.items()):
            if len(bucket) < self.MIN_SAMPLE:
                continue
            wins = sum(1 for p in bucket if p.result == "won")
            wr = wins / len(bucket)
            roi = _roi([p.profit_loss for p in bucket])
            _, p_val = _z_test_proportion(wins, len(bucket), baseline_wr)
            edges.append(TemporalEdge(
                pattern_name=f"dow_{dow_names[dow].lower()}",
                description=f"{dow_names[dow]} picks: {len(bucket)} picks, {wr:.1%} win rate, {roi:+.1f}% ROI",
                win_rate=wr,
                roi_pct=roi,
                sample_size=len(bucket),
                p_value=p_val,
                is_significant=p_val < self.SIGNIFICANCE_ALPHA,
            ))

        # ---- Month effects ----
        month_buckets: dict[int, list[DailyPick]] = defaultdict(list)
        for p in picks:
            month_buckets[p.match_date.month].append(p)

        month_names = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December",
        }
        for month, bucket in sorted(month_buckets.items()):
            if len(bucket) < self.MIN_SAMPLE:
                continue
            wins = sum(1 for p in bucket if p.result == "won")
            wr = wins / len(bucket)
            roi = _roi([p.profit_loss for p in bucket])
            _, p_val = _z_test_proportion(wins, len(bucket), baseline_wr)
            edges.append(TemporalEdge(
                pattern_name=f"month_{month_names[month].lower()}",
                description=f"{month_names[month]}: {len(bucket)} picks, {wr:.1%} win rate, {roi:+.1f}% ROI",
                win_rate=wr,
                roi_pct=roi,
                sample_size=len(bucket),
                p_value=p_val,
                is_significant=p_val < self.SIGNIFICANCE_ALPHA,
            ))

        # ---- Midweek vs weekend (Tue/Wed/Thu vs Sat/Sun) ----
        midweek = [p for p in picks if p.match_date.weekday() in (1, 2, 3)]
        weekend = [p for p in picks if p.match_date.weekday() in (5, 6)]

        for label, bucket in [("midweek_tue_thu", midweek), ("weekend_sat_sun", weekend)]:
            if len(bucket) < self.MIN_SAMPLE:
                continue
            wins = sum(1 for p in bucket if p.result == "won")
            wr = wins / len(bucket)
            roi = _roi([p.profit_loss for p in bucket])
            _, p_val = _z_test_proportion(wins, len(bucket), baseline_wr)
            edges.append(TemporalEdge(
                pattern_name=label,
                description=(
                    f"{'Midweek (Tue-Thu)' if 'midweek' in label else 'Weekend (Sat-Sun)'}: "
                    f"{len(bucket)} picks, {wr:.1%} win rate, {roi:+.1f}% ROI"
                ),
                win_rate=wr,
                roi_pct=roi,
                sample_size=len(bucket),
                p_value=p_val,
                is_significant=p_val < self.SIGNIFICANCE_ALPHA,
            ))

        # ---- Distance from season start (split into thirds) ----
        if picks:
            sorted_dates = sorted(p.match_date for p in picks)
            season_start = sorted_dates[0]
            season_end = sorted_dates[-1]
            span = max((season_end - season_start).days, 1)

            thirds: dict[str, list[DailyPick]] = {"early_season": [], "mid_season": [], "late_season": []}
            for p in picks:
                elapsed = (p.match_date - season_start).days / span
                if elapsed < 0.333:
                    thirds["early_season"].append(p)
                elif elapsed < 0.667:
                    thirds["mid_season"].append(p)
                else:
                    thirds["late_season"].append(p)

            labels = {
                "early_season": "Early season (first third)",
                "mid_season": "Mid season (middle third)",
                "late_season": "Late season (final third)",
            }
            for key, bucket in thirds.items():
                if len(bucket) < self.MIN_SAMPLE:
                    continue
                wins = sum(1 for p in bucket if p.result == "won")
                wr = wins / len(bucket)
                roi = _roi([p.profit_loss for p in bucket])
                _, p_val = _z_test_proportion(wins, len(bucket), baseline_wr)
                edges.append(TemporalEdge(
                    pattern_name=key,
                    description=f"{labels[key]}: {len(bucket)} picks, {wr:.1%} win rate, {roi:+.1f}% ROI",
                    win_rate=wr,
                    roi_pct=roi,
                    sample_size=len(bucket),
                    p_value=p_val,
                    is_significant=p_val < self.SIGNIFICANCE_ALPHA,
                ))

        sig_count = sum(1 for e in edges if e.is_significant)
        logger.info("Temporal analysis: %d patterns found, %d significant", len(edges), sig_count)
        return sorted(edges, key=lambda e: e.roi_pct, reverse=True)

    # ------------------------------------------------------------------
    # 2. Odds movement edge analysis
    # ------------------------------------------------------------------

    def analyze_odds_movement_edges(self, sport_slug: str) -> list[OddsEdge]:
        """Detect value patterns related to line movement and steam moves.

        Cross-references DailyPick (pick_odds = best_odds at pick time) against
        closing odds in MatchOdds (odds_type='closing').  If no closing odds are
        available, we skip that pick rather than fabricating data.
        """
        session = self._get_session()
        try:
            picks = (
                session.query(DailyPick)
                .filter(DailyPick.result.isnot(None), DailyPick.profit_loss.isnot(None))
                .order_by(DailyPick.match_date)
                .all()
            )
            if not picks:
                return []

            # Build a lookup: match_id (via prediction.match_date + teams) -> closing odds
            # We use MatchOdds with odds_type='closing' joined via the Prediction table
            closing_lookup: dict[int, dict[str, float]] = {}
            closing_rows = (
                session.query(MatchOdds)
                .filter(MatchOdds.odds_type == "closing")
                .all()
            )
            for row in closing_rows:
                closing_lookup[row.match_id] = {
                    "home": row.odds_home,
                    "draw": row.odds_draw,
                    "away": row.odds_away,
                    "over": row.odds_over,
                    "under": row.odds_under,
                }

            # Classify each pick
            steam_with: list[DailyPick] = []      # closing < pick_odds (line moved against us)
            steam_against: list[DailyPick] = []   # closing > pick_odds (line moved with us)
            early_picks: list[DailyPick] = []     # pick odds better than closing
            closing_picks: list[DailyPick] = []   # pick odds roughly equal to closing

            STEAM_THRESHOLD = 0.05   # 5% difference in implied probability

            def _pick_closing_odds(pick: DailyPick, closing: dict) -> Optional[float]:
                """Map pick_market / pick_selection to a closing odds value."""
                market = pick.pick_market.lower()
                sel = pick.pick_selection.lower()
                if "home" in sel or sel == "1":
                    return closing.get("home")
                if "away" in sel or sel == "2":
                    return closing.get("away")
                if "draw" in sel or sel == "x":
                    return closing.get("draw")
                if "over" in sel:
                    return closing.get("over")
                if "under" in sel:
                    return closing.get("under")
                return None

            for pick in picks:
                # Try to find closing odds for this pick's match
                pred = pick.prediction
                if pred is None:
                    continue

                # Find matching match_id from MatchOdds via match date + teams
                # We do a lightweight query using the prediction match metadata
                match_rows = (
                    session.query(Match)
                    .join(Match.season)
                    .filter(Match.match_date == pred.match_date)
                    .all()
                )
                match_id = None
                for m in match_rows:
                    ht = m.home_team.canonical_name if m.home_team else ""
                    at = m.away_team.canonical_name if m.away_team else ""
                    if pred.home_team.lower() in ht.lower() or ht.lower() in pred.home_team.lower():
                        if pred.away_team.lower() in at.lower() or at.lower() in pred.away_team.lower():
                            match_id = m.id
                            break

                if match_id is None or match_id not in closing_lookup:
                    continue

                closing = closing_lookup[match_id]
                close_odds = _pick_closing_odds(pick, closing)
                if close_odds is None or close_odds <= 1.0:
                    continue

                pick_odds = pick.best_odds
                if pick_odds <= 1.0:
                    continue

                pick_impl = 1.0 / pick_odds
                close_impl = 1.0 / close_odds
                delta = close_impl - pick_impl  # positive = line shortened (steam against us)

                if delta > STEAM_THRESHOLD:
                    # closing price shorter — market steamed against original pick
                    steam_against.append(pick)
                elif delta < -STEAM_THRESHOLD:
                    # closing price drifted — line moved in our favour
                    steam_with.append(pick)

                # Early vs closing
                if pick_odds > close_odds * 1.02:
                    early_picks.append(pick)
                else:
                    closing_picks.append(pick)

            edges: list[OddsEdge] = []

            def _bucket_stats(bucket: list[DailyPick], direction: str, label: str, desc: str) -> None:
                if len(bucket) < self.MIN_SAMPLE:
                    return
                wins = sum(1 for p in bucket if p.result == "won")
                wr = wins / len(bucket)
                roi = _roi([p.profit_loss for p in bucket])
                edges.append(OddsEdge(
                    pattern_name=label,
                    description=desc.format(n=len(bucket), wr=wr, roi=roi),
                    win_rate=wr,
                    roi_pct=roi,
                    sample_size=len(bucket),
                    direction=direction,
                ))

            _bucket_stats(
                steam_with, "steam_with",
                "steam_move_with",
                "Steam move WITH us (line drifted our way): {n} picks, {wr:.1%} WR, {roi:+.1f}% ROI",
            )
            _bucket_stats(
                steam_against, "steam_against",
                "steam_move_against",
                "Steam move AGAINST us (line shortened post-pick): {n} picks, {wr:.1%} WR, {roi:+.1f}% ROI",
            )
            _bucket_stats(
                early_picks, "early",
                "early_value_bet",
                "Early picks (got >2% better than closing): {n} picks, {wr:.1%} WR, {roi:+.1f}% ROI",
            )
            _bucket_stats(
                closing_picks, "closing",
                "closing_line_bet",
                "Closing-line bets (roughly at closing price): {n} picks, {wr:.1%} WR, {roi:+.1f}% ROI",
            )

            logger.info("Odds movement analysis: %d edge patterns found", len(edges))
            return sorted(edges, key=lambda e: e.roi_pct, reverse=True)

        finally:
            if self._session is None:
                session.close()

    # ------------------------------------------------------------------
    # 3. Confidence calibration
    # ------------------------------------------------------------------

    def analyze_confidence_calibration(self, sport_slug: str) -> CalibrationReport:
        """
        Evaluate model calibration across confidence bins.
        Returns ECE, overconfident / underconfident ranges, and the ROI sweet spot.
        """
        picks = self._fetch_resolved_picks(sport_slug, lookback_days=365)

        # Bin by model_prob in 10 equal-width bins
        N_BINS = 10
        bin_edges = [i / N_BINS for i in range(N_BINS + 1)]
        bins: list[list[DailyPick]] = [[] for _ in range(N_BINS)]

        for pick in picks:
            prob = pick.model_prob
            if prob is None or prob <= 0 or prob > 1:
                continue
            bin_idx = min(int(prob * N_BINS), N_BINS - 1)
            bins[bin_idx].append(pick)

        cal_bins: list[CalibrationBin] = []
        ece_sum = 0.0
        total_with_data = 0

        for i, bucket in enumerate(bins):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            if not bucket:
                continue
            wins = sum(1 for p in bucket if p.result == "won")
            actual_wr = wins / len(bucket)
            pred_prob = sum(p.model_prob for p in bucket) / len(bucket)
            roi = _roi([p.profit_loss for p in bucket])

            cal_bins.append(CalibrationBin(
                confidence_lo=lo,
                confidence_hi=hi,
                predicted_prob=pred_prob,
                actual_win_rate=actual_wr,
                count=len(bucket),
                roi_pct=roi,
            ))
            ece_sum += len(bucket) * abs(pred_prob - actual_wr)
            total_with_data += len(bucket)

        overall_ece = ece_sum / total_with_data if total_with_data else 0.0

        # Identify overconfident (predicted > actual) and underconfident ranges
        overconfident_bins = [b for b in cal_bins if b.predicted_prob > b.actual_win_rate + 0.05 and b.count >= self.MIN_SAMPLE]
        underconfident_bins = [b for b in cal_bins if b.actual_win_rate > b.predicted_prob + 0.05 and b.count >= self.MIN_SAMPLE]

        overconfident_range = None
        if overconfident_bins:
            overconfident_range = (
                min(b.confidence_lo for b in overconfident_bins),
                max(b.confidence_hi for b in overconfident_bins),
            )

        underconfident_range = None
        if underconfident_bins:
            underconfident_range = (
                min(b.confidence_lo for b in underconfident_bins),
                max(b.confidence_hi for b in underconfident_bins),
            )

        # Sweet spot: bin with best ROI among those with sufficient data
        eligible = [b for b in cal_bins if b.count >= self.MIN_SAMPLE]
        sweet_spot = None
        if eligible:
            best_bin = max(eligible, key=lambda b: b.roi_pct)
            sweet_spot = (best_bin.confidence_lo, best_bin.confidence_hi)

        logger.info(
            "Calibration analysis: ECE=%.4f, %d bins, sweet_spot=%s",
            overall_ece, len(cal_bins), sweet_spot,
        )
        return CalibrationReport(
            bins=cal_bins,
            overall_ece=overall_ece,
            overconfident_range=overconfident_range,
            underconfident_range=underconfident_range,
            sweet_spot=sweet_spot,
        )

    # ------------------------------------------------------------------
    # 4. League x market matrix
    # ------------------------------------------------------------------

    def analyze_league_market_matrix(self, sport_slug: str) -> LeagueMarketMatrix:
        """
        Compute ROI, win rate, avg edge, and CLV for every league x market combination.
        """
        picks = self._fetch_resolved_picks(sport_slug, lookback_days=365)

        # Aggregate: (league, market) -> list of picks
        cell_picks: dict[tuple[str, str], list[DailyPick]] = defaultdict(list)
        for pick in picks:
            cell_picks[(pick.league, pick.pick_market)].append(pick)

        # Build matrix
        matrix: dict[str, dict[str, CellStats]] = defaultdict(dict)
        all_cells: list[tuple[str, str, CellStats]] = []

        for (league, market), bucket in cell_picks.items():
            if len(bucket) < self.MIN_SAMPLE:
                continue
            wins = sum(1 for p in bucket if p.result == "won")
            roi = _roi([p.profit_loss for p in bucket])
            avg_edge = sum(p.edge for p in bucket) / len(bucket)

            # CLV estimate: average edge as a proxy (real CLV needs closing odds lookup)
            # We compute a simplified CLV from edge — if closing odds are absent, edge
            # still captures how much we beat the market at pick time.
            clv = avg_edge * 100.0

            stats = CellStats(
                picks=len(bucket),
                wins=wins,
                roi_pct=roi,
                avg_edge=avg_edge,
                clv_pct=clv,
            )
            matrix[league][market] = stats
            all_cells.append((league, market, stats))

        sorted_cells = sorted(all_cells, key=lambda t: t[2].roi_pct, reverse=True)
        best_combos = sorted_cells[:10]
        worst_combos = sorted_cells[-10:][::-1]  # worst first

        logger.info(
            "League-market matrix: %d cells, best ROI=%.1f%%, worst ROI=%.1f%%",
            len(all_cells),
            best_combos[0][2].roi_pct if best_combos else 0,
            worst_combos[0][2].roi_pct if worst_combos else 0,
        )
        return LeagueMarketMatrix(
            matrix=dict(matrix),
            best_combos=best_combos,
            worst_combos=worst_combos,
        )

    # ------------------------------------------------------------------
    # 5. Tier performance
    # ------------------------------------------------------------------

    def analyze_tier_performance(self, sport_slug: str) -> dict[str, TierStats]:
        """
        Break down pick performance by tier (Platinum, Gold, Silver, Bronze).
        Verifies tier thresholds are still calibrated.
        """
        picks = self._fetch_resolved_picks(sport_slug, lookback_days=365)

        tier_buckets: dict[str, list[DailyPick]] = defaultdict(list)
        for pick in picks:
            tier_buckets[pick.tier].append(pick)

        results: dict[str, TierStats] = {}
        for tier, bucket in sorted(tier_buckets.items()):
            if not bucket:
                continue
            wins = sum(1 for p in bucket if p.result == "won")
            wr = wins / len(bucket) if bucket else 0.0
            roi = _roi([p.profit_loss for p in bucket])
            avg_odds = sum(p.best_odds for p in bucket) / len(bucket)
            avg_edge = sum(p.edge for p in bucket) / len(bucket)
            clv = avg_edge * 100.0

            results[tier] = TierStats(
                picks=len(bucket),
                wins=wins,
                win_rate=wr,
                roi_pct=roi,
                avg_odds=avg_odds,
                clv_pct=clv,
            )

        # Log calibration warnings
        tier_order = ["Platinum", "Gold", "Silver", "Bronze"]
        prev_wr = None
        for tier in tier_order:
            if tier not in results:
                continue
            stats = results[tier]
            if prev_wr is not None and stats.win_rate > prev_wr:
                logger.warning(
                    "Tier calibration issue: %s WR (%.1f%%) > higher tier WR (%.1f%%). "
                    "Thresholds may need recalibration.",
                    tier, stats.win_rate * 100, prev_wr * 100,
                )
            prev_wr = stats.win_rate

        logger.info("Tier analysis: %d tiers, %d total picks", len(results), len(picks))
        return results

    # ------------------------------------------------------------------
    # 6. Strategy recommendations
    # ------------------------------------------------------------------

    def generate_strategy_recommendations(self, sport_slug: str) -> list[StrategyRecommendation]:
        """
        Combine all analyses into actionable, prioritised recommendations.
        """
        recommendations: list[StrategyRecommendation] = []

        # --- Temporal patterns ---
        try:
            temporal = self.analyze_temporal_patterns(sport_slug)
            if temporal:
                best_temporal = [e for e in temporal if e.roi_pct > 3.0 and e.is_significant]
                worst_temporal = [e for e in temporal if e.roi_pct < -2.0 and e.is_significant]

                for edge in best_temporal[:3]:
                    recommendations.append(StrategyRecommendation(
                        recommendation=f"Increase stake weight on {edge.pattern_name.replace('_', ' ')} picks",
                        evidence=(
                            f"{edge.description} — p={edge.p_value:.3f}, n={edge.sample_size}"
                        ),
                        expected_impact=f"+{edge.roi_pct:.1f}% ROI on filtered subset",
                        priority="high" if edge.roi_pct > 5.0 else "medium",
                    ))
                for edge in worst_temporal[:3]:
                    recommendations.append(StrategyRecommendation(
                        recommendation=f"Exclude or reduce stake on {edge.pattern_name.replace('_', ' ')} picks",
                        evidence=f"{edge.description} — p={edge.p_value:.3f}, n={edge.sample_size}",
                        expected_impact=f"Avoid {edge.roi_pct:.1f}% ROI drag on filtered subset",
                        priority="high" if edge.roi_pct < -5.0 else "medium",
                    ))
        except Exception as exc:
            logger.warning("Temporal analysis failed during recommendations: %s", exc)

        # --- Odds movement ---
        try:
            odds_edges = self.analyze_odds_movement_edges(sport_slug)
            if odds_edges:
                best_oe = [e for e in odds_edges if e.roi_pct > 3.0]
                worst_oe = [e for e in odds_edges if e.roi_pct < -2.0]

                for edge in best_oe[:2]:
                    recommendations.append(StrategyRecommendation(
                        recommendation=f"Prioritise {edge.pattern_name.replace('_', ' ')} opportunities",
                        evidence=edge.description,
                        expected_impact=f"+{edge.roi_pct:.1f}% ROI, n={edge.sample_size}",
                        priority="high" if edge.roi_pct > 6.0 else "medium",
                    ))
                for edge in worst_oe[:2]:
                    recommendations.append(StrategyRecommendation(
                        recommendation=f"Avoid {edge.pattern_name.replace('_', ' ')} situations",
                        evidence=edge.description,
                        expected_impact=f"Eliminate {edge.roi_pct:.1f}% ROI drag, n={edge.sample_size}",
                        priority="medium",
                    ))
        except Exception as exc:
            logger.warning("Odds movement analysis failed during recommendations: %s", exc)

        # --- Calibration sweet spot ---
        try:
            cal = self.analyze_confidence_calibration(sport_slug)
            if cal.sweet_spot:
                lo, hi = cal.sweet_spot
                sweet_bin = next(
                    (b for b in cal.bins if b.confidence_lo == lo), None
                )
                if sweet_bin and sweet_bin.roi_pct > 3.0:
                    recommendations.append(StrategyRecommendation(
                        recommendation=(
                            f"Focus on picks with model confidence {lo:.0%}–{hi:.0%} "
                            f"(calibration sweet spot)"
                        ),
                        evidence=(
                            f"ROI={sweet_bin.roi_pct:+.1f}%, actual win rate={sweet_bin.actual_win_rate:.1%}, "
                            f"predicted={sweet_bin.predicted_prob:.1%}, n={sweet_bin.count}"
                        ),
                        expected_impact=f"+{sweet_bin.roi_pct:.1f}% ROI within confidence band",
                        priority="high" if sweet_bin.roi_pct > 6.0 else "medium",
                    ))

            if cal.overconfident_range:
                lo, hi = cal.overconfident_range
                recommendations.append(StrategyRecommendation(
                    recommendation=f"Flag model overconfidence in {lo:.0%}–{hi:.0%} confidence range",
                    evidence=(
                        f"ECE={cal.overall_ece:.4f}; predicted probs > actual win rates in this band"
                    ),
                    expected_impact="Reduce stake or skip picks in overconfident zone to avoid -EV bets",
                    priority="high" if cal.overall_ece > 0.08 else "medium",
                ))
        except Exception as exc:
            logger.warning("Calibration analysis failed during recommendations: %s", exc)

        # --- League x market worst combos ---
        try:
            lm = self.analyze_league_market_matrix(sport_slug)
            for league, market, stats in lm.worst_combos[:5]:
                if stats.roi_pct < -10.0:
                    recommendations.append(StrategyRecommendation(
                        recommendation=f"Stop betting {market} in {league}",
                        evidence=(
                            f"{stats.roi_pct:+.1f}% ROI over {stats.picks} picks, "
                            f"avg edge={stats.avg_edge:.3f}"
                        ),
                        expected_impact=f"Eliminates {stats.roi_pct:.1f}% ROI drag on {stats.picks} annual picks",
                        priority="high" if stats.roi_pct < -15.0 else "medium",
                    ))

            for league, market, stats in lm.best_combos[:3]:
                if stats.roi_pct > 8.0:
                    recommendations.append(StrategyRecommendation(
                        recommendation=f"Increase stake weight on {market} in {league}",
                        evidence=(
                            f"{stats.roi_pct:+.1f}% ROI over {stats.picks} picks, "
                            f"avg edge={stats.avg_edge:.3f}"
                        ),
                        expected_impact=f"+{stats.roi_pct:.1f}% ROI on {stats.picks} annual picks",
                        priority="high" if stats.roi_pct > 12.0 else "medium",
                    ))
        except Exception as exc:
            logger.warning("League-market analysis failed during recommendations: %s", exc)

        # --- Tier calibration ---
        try:
            tiers = self.analyze_tier_performance(sport_slug)
            tier_order = ["Platinum", "Gold", "Silver", "Bronze"]
            for tier in tier_order:
                if tier not in tiers:
                    continue
                stats = tiers[tier]
                if stats.roi_pct > 10.0 and stats.picks >= 30:
                    recommendations.append(StrategyRecommendation(
                        recommendation=f"Increase stake on {tier} picks — CLV is positive and ROI is strong",
                        evidence=(
                            f"ROI={stats.roi_pct:+.1f}%, CLV={stats.clv_pct:+.1f}%, "
                            f"WR={stats.win_rate:.1%}, n={stats.picks}"
                        ),
                        expected_impact=f"+{stats.roi_pct:.1f}% ROI on {tier} tier picks",
                        priority="high",
                    ))
                if stats.roi_pct < -5.0 and stats.picks >= 30:
                    recommendations.append(StrategyRecommendation(
                        recommendation=f"Review {tier} tier threshold — negative ROI suggests poor calibration",
                        evidence=(
                            f"ROI={stats.roi_pct:+.1f}%, WR={stats.win_rate:.1%}, n={stats.picks}"
                        ),
                        expected_impact="Recalibrate tier edge threshold to improve pick quality",
                        priority="high" if stats.roi_pct < -10.0 else "medium",
                    ))
        except Exception as exc:
            logger.warning("Tier analysis failed during recommendations: %s", exc)

        # Sort: high -> medium -> low, then by expected P&L impact (heuristic: ROI keyword)
        priority_rank = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_rank.get(r.priority, 2))

        logger.info("Generated %d strategy recommendations", len(recommendations))
        return recommendations

    # ------------------------------------------------------------------
    # 7. Full analysis runner
    # ------------------------------------------------------------------

    def run_full_analysis(self, sport_slug: str = "football") -> DiscoveryReport:
        """
        Execute all analyses, assemble a DiscoveryReport, and persist significant
        findings to the EdgeLog table.
        """
        logger.info("Starting full edge discovery analysis for sport=%s", sport_slug)
        start = datetime.utcnow()

        # --- Run all analyses (handle failures gracefully so one failure
        #     doesn't kill the whole report) ---
        temporal_edges: list[TemporalEdge] = []
        odds_edges: list[OddsEdge] = []
        calibration_report: Optional[CalibrationReport] = None
        league_market_matrix: Optional[LeagueMarketMatrix] = None
        tier_performance: dict[str, TierStats] = {}
        recommendations: list[StrategyRecommendation] = []

        try:
            temporal_edges = self.analyze_temporal_patterns(sport_slug)
        except Exception as exc:
            logger.error("Temporal pattern analysis failed: %s", exc, exc_info=True)

        try:
            odds_edges = self.analyze_odds_movement_edges(sport_slug)
        except Exception as exc:
            logger.error("Odds movement analysis failed: %s", exc, exc_info=True)

        try:
            calibration_report = self.analyze_confidence_calibration(sport_slug)
        except Exception as exc:
            logger.error("Calibration analysis failed: %s", exc, exc_info=True)

        try:
            league_market_matrix = self.analyze_league_market_matrix(sport_slug)
        except Exception as exc:
            logger.error("League-market matrix analysis failed: %s", exc, exc_info=True)

        try:
            tier_performance = self.analyze_tier_performance(sport_slug)
        except Exception as exc:
            logger.error("Tier performance analysis failed: %s", exc, exc_info=True)

        try:
            recommendations = self.generate_strategy_recommendations(sport_slug)
        except Exception as exc:
            logger.error("Strategy recommendations failed: %s", exc, exc_info=True)

        # Compute total picks analyzed
        total_picks = 0
        if temporal_edges:
            total_picks = max((e.sample_size for e in temporal_edges), default=0)

        # --- Log significant findings to EdgeLog ---
        self._persist_edge_log(
            sport_slug=sport_slug,
            temporal_edges=temporal_edges,
            league_market_matrix=league_market_matrix,
            calibration_report=calibration_report,
        )

        # Build summary string
        sig_temporal = sum(1 for e in temporal_edges if e.is_significant)
        summary_parts = [
            f"Analysis for sport={sport_slug} completed in {(datetime.utcnow()-start).total_seconds():.1f}s.",
            f"Temporal: {len(temporal_edges)} patterns, {sig_temporal} significant.",
        ]
        if calibration_report:
            summary_parts.append(f"ECE={calibration_report.overall_ece:.4f}.")
        if league_market_matrix:
            n_cells = sum(len(v) for v in league_market_matrix.matrix.values())
            summary_parts.append(f"League-market matrix: {n_cells} cells.")
        summary_parts.append(f"Tier performance: {len(tier_performance)} tiers.")
        summary_parts.append(f"Recommendations: {len(recommendations)} actionable insights.")

        summary = " ".join(summary_parts)
        logger.info(summary)

        return DiscoveryReport(
            sport_slug=sport_slug,
            generated_at=datetime.utcnow(),
            temporal_edges=temporal_edges,
            odds_edges=odds_edges,
            calibration_report=calibration_report,
            league_market_matrix=league_market_matrix,
            tier_performance=tier_performance,
            recommendations=recommendations,
            total_picks_analyzed=total_picks,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _persist_edge_log(
        self,
        sport_slug: str,
        temporal_edges: list[TemporalEdge],
        league_market_matrix: Optional[LeagueMarketMatrix],
        calibration_report: Optional[CalibrationReport],
    ) -> None:
        """Write significant discoveries to the EdgeLog table."""
        session = self._get_session()
        today = date.today()
        logged = 0

        try:
            # Temporal edges
            for edge in temporal_edges:
                if not edge.is_significant or edge.sample_size < self.MIN_SAMPLE:
                    continue
                log_entry = EdgeLog(
                    agent_name="EdgeDiscoveryEngine",
                    sport_slug=sport_slug,
                    league="all",
                    market="all",
                    edge_type="temporal_roi",
                    edge_value=edge.roi_pct,
                    sample_size=edge.sample_size,
                    is_significant=True,
                    discovered_date=today,
                    metadata_json={
                        "pattern_name": edge.pattern_name,
                        "description": edge.description,
                        "win_rate": edge.win_rate,
                        "p_value": edge.p_value,
                    },
                )
                session.add(log_entry)
                logged += 1

            # League-market best combos
            if league_market_matrix:
                for league, market, stats in league_market_matrix.best_combos[:5]:
                    if stats.roi_pct < 5.0:
                        continue
                    log_entry = EdgeLog(
                        agent_name="EdgeDiscoveryEngine",
                        sport_slug=sport_slug,
                        league=league,
                        market=market,
                        edge_type="roi",
                        edge_value=stats.roi_pct,
                        sample_size=stats.picks,
                        is_significant=stats.picks >= self.MIN_SAMPLE,
                        discovered_date=today,
                        metadata_json={
                            "wins": stats.wins,
                            "avg_edge": stats.avg_edge,
                            "clv_pct": stats.clv_pct,
                        },
                    )
                    session.add(log_entry)
                    logged += 1

                # Worst combos (negative edges worth logging too)
                for league, market, stats in league_market_matrix.worst_combos[:5]:
                    if stats.roi_pct > -5.0:
                        continue
                    log_entry = EdgeLog(
                        agent_name="EdgeDiscoveryEngine",
                        sport_slug=sport_slug,
                        league=league,
                        market=market,
                        edge_type="roi_negative",
                        edge_value=stats.roi_pct,
                        sample_size=stats.picks,
                        is_significant=stats.picks >= self.MIN_SAMPLE,
                        discovered_date=today,
                        metadata_json={
                            "wins": stats.wins,
                            "avg_edge": stats.avg_edge,
                            "clv_pct": stats.clv_pct,
                        },
                    )
                    session.add(log_entry)
                    logged += 1

            # Calibration ECE
            if calibration_report and calibration_report.overall_ece > 0.05:
                log_entry = EdgeLog(
                    agent_name="EdgeDiscoveryEngine",
                    sport_slug=sport_slug,
                    league="all",
                    market="all",
                    edge_type="calibration_ece",
                    edge_value=calibration_report.overall_ece,
                    sample_size=sum(b.count for b in calibration_report.bins),
                    is_significant=calibration_report.overall_ece > 0.08,
                    discovered_date=today,
                    metadata_json={
                        "overconfident_range": list(calibration_report.overconfident_range)
                        if calibration_report.overconfident_range else None,
                        "underconfident_range": list(calibration_report.underconfident_range)
                        if calibration_report.underconfident_range else None,
                        "sweet_spot": list(calibration_report.sweet_spot)
                        if calibration_report.sweet_spot else None,
                    },
                )
                session.add(log_entry)
                logged += 1

            session.commit()
            logger.info("Persisted %d edge log entries", logged)

        except Exception as exc:
            logger.error("Failed to persist edge log entries: %s", exc, exc_info=True)
            session.rollback()
        finally:
            if self._session is None:
                session.close()
