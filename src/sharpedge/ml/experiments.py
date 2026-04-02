"""A/B Testing (Experiment) Framework for SharpEdge.

Separates "I think this is better" from "I know this is better with 95% confidence."

Architecture
------------
- Experiments live in-memory (no extra DB table needed).
- Each experiment has a champion and a challenger configuration.
- Traffic split is deterministic: hash(match_id + experiment_id) so the
  same match always routes to the same variant — no data leakage across runs.
- Statistical tests are implemented from scratch (no scipy).
  * Two-proportion z-test for accuracy
  * Welch's t-test for ROI
- p-values use the Hart (1968) rational polynomial approximation to the
  normal CDF, which is accurate to ±7.5e-8.

Usage
-----
    fw = ExperimentFramework()
    exp = fw.create_experiment(
        name="kelly_0.30_vs_0.25",
        description="Test higher Kelly fraction",
        champion_config={"kelly_fraction": 0.25},
        challenger_config={"kelly_fraction": 0.30},
        traffic_split=0.50,
        min_samples=150,
    )
    variant = fw.assign_variant(match_id="eng_pl_2024_man_utd_chelsea", experiment_id=exp.id)
    # ... run match through variant config, get result ...
    fw.record_outcome(exp.id, match_id, variant, prediction, actual_outcome, odds)
    result = fw.evaluate_experiment(exp.id)
    should_promote, reason = fw.should_promote_challenger(exp.id)
    print(fw.generate_report(exp.id))
"""

from __future__ import annotations

import hashlib
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from sharpedge.config import settings
from sharpedge.db.engine import get_session
from sharpedge.db.models import DailyPick, Prediction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Statistical primitives (no scipy)
# ---------------------------------------------------------------------------

def _normal_cdf(x: float) -> float:
    """Cumulative distribution function of the standard normal.

    Uses the Hart (1968) rational polynomial approximation.
    Accuracy: ±7.5e-8 over the entire real line.
    """
    # Abramowitz & Stegun 26.2.17 approximation
    p = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    if x >= 0.0:
        t = 1.0 / (1.0 + p * x)
        poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
        return 1.0 - _standard_normal_pdf(x) * poly
    else:
        return 1.0 - _normal_cdf(-x)


def _standard_normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _z_test_proportions(
    p1: float, n1: int, p2: float, n2: int
) -> tuple[float, float]:
    """Two-proportion z-test (two-tailed).

    H0: p1 == p2
    Returns (z_stat, p_value).

    p1, p2 are proportions (accuracy rates, 0-1).
    n1, n2 are sample sizes.
    """
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    # Pooled proportion under H0
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    denom_sq = p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)
    if denom_sq <= 0.0:
        return 0.0, 1.0

    z_stat = (p1 - p2) / math.sqrt(denom_sq)
    # Two-tailed p-value
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z_stat)))
    return z_stat, p_value


def _t_test_means(
    mean1: float, std1: float, n1: int,
    mean2: float, std2: float, n2: int,
) -> tuple[float, float]:
    """Welch's two-sample t-test (two-tailed, unequal variances).

    Returns (t_stat, p_value).

    For p-value we use the normal approximation to the t distribution,
    which is conservative and adequate for n > 30 (our min_samples=100).
    """
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    var1 = std1 * std1 / n1
    var2 = std2 * std2 / n2
    se = math.sqrt(var1 + var2)
    if se == 0.0:
        return 0.0, 1.0

    t_stat = (mean1 - mean2) / se

    # Welch–Satterthwaite degrees of freedom (for reference; we use normal approx)
    if var1 == 0.0 and var2 == 0.0:
        df = float(n1 + n2 - 2)
    else:
        numerator = (var1 + var2) ** 2
        denominator = (var1 ** 2) / (n1 - 1) + (var2 ** 2) / (n2 - 1)
        df = numerator / denominator if denominator > 0.0 else float(n1 + n2 - 2)

    # Normal approximation (accurate when df > 30, which holds given min_samples=100)
    p_value = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
    return t_stat, p_value


def _confidence_interval_proportion(p: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    z = 1.96  # 95% CI (alpha=0.05)
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _confidence_interval_mean(mean: float, std: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% CI for a mean (normal approximation)."""
    if n < 2:
        return mean, mean
    z = 1.96
    se = std / math.sqrt(n)
    return mean - z * se, mean + z * se


def _brier_score(probs: list[float], outcomes: list[float]) -> float:
    """Mean squared error between predicted probabilities and binary outcomes."""
    if not probs:
        return 1.0
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return (mean, sample_std) for a list. Returns (0.0, 0.0) if empty."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    """Registered A/B experiment.

    Fields
    ------
    id : str
        UUID, generated automatically.
    name : str
        Short slug, e.g. "kelly_0.30_vs_0.25".
    description : str
        Human-readable purpose.
    champion_config : dict
        Config of the current production model.
    challenger_config : dict
        Config of the variant being tested.
    status : str
        "active" | "completed" | "cancelled"
    start_date : date
        Date the experiment was created.
    end_date : date | None
        Date the experiment was closed (completed/cancelled).
    min_samples : int
        Minimum observations per variant before drawing conclusions.
    traffic_split : float
        Fraction of traffic routed to challenger (0.0–1.0).
        0.5 = equal split.
    """
    id: str
    name: str
    description: str
    champion_config: dict[str, Any]
    challenger_config: dict[str, Any]
    status: str                          # "active" | "completed" | "cancelled"
    start_date: date
    end_date: date | None
    min_samples: int
    traffic_split: float                 # 0.0–1.0, fraction going to challenger


@dataclass
class ExperimentResult:
    """Evaluated outcome of an A/B experiment.

    Fields
    ------
    experiment_id : str
    champion_stats : dict
        accuracy, brier_score, roi, clv, n_samples, …
    challenger_stats : dict
        same keys as champion_stats
    winner : str
        "champion" | "challenger" | "inconclusive"
    p_value : float
        From ROI t-test (primary decision metric).
    is_significant : bool
        p_value < 0.05
    summary : str
        One-line human-readable verdict.
    """
    experiment_id: str
    champion_stats: dict[str, Any]
    challenger_stats: dict[str, Any]
    winner: str                          # "champion" | "challenger" | "inconclusive"
    p_value: float
    is_significant: bool
    summary: str


# Internal record per observation
@dataclass
class _ObservationRecord:
    match_id: str
    variant: str                         # "champion" | "challenger"
    predicted_prob: float                # probability assigned to the selected outcome
    actual_outcome: str                  # e.g. "H", "D", "A", "over", "under"
    predicted_selection: str            # what the model picked
    odds: float
    profit_loss: float                   # after-fee P/L in units (stake = 1 unit)
    clv: float                           # closing line value (positive = beat market)
    recorded_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# ExperimentFramework
# ---------------------------------------------------------------------------

class ExperimentFramework:
    """Model A/B testing framework.

    All state is in-memory.  Persist experiments externally if needed
    (e.g. serialise `self._experiments` to JSON before shutdown).

    Thread safety: not designed for concurrent writes.  Use a lock
    if calling from async context.
    """

    def __init__(self) -> None:
        self._experiments: dict[str, Experiment] = {}
        # experiment_id -> list of _ObservationRecord
        self._observations: dict[str, list[_ObservationRecord]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        description: str,
        champion_config: dict[str, Any],
        challenger_config: dict[str, Any],
        traffic_split: float = 0.5,
        min_samples: int = 100,
    ) -> Experiment:
        """Register a new A/B test and return the Experiment object.

        Parameters
        ----------
        name : str
            Short slug — used in reports and logs.
        description : str
            Why this experiment exists.
        champion_config : dict
            Config keys/values of the current production configuration.
        challenger_config : dict
            Config keys/values of the variant under test.
        traffic_split : float
            Fraction of picks assigned to challenger (default 0.5).
            Must be in (0.0, 1.0).
        min_samples : int
            Minimum observations per variant before allowing a promotion
            decision.  Defaults to 100.
        """
        if not 0.0 < traffic_split < 1.0:
            raise ValueError(f"traffic_split must be strictly between 0 and 1, got {traffic_split}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be at least 1, got {min_samples}")

        experiment_id = str(uuid.uuid4())
        exp = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            champion_config=dict(champion_config),
            challenger_config=dict(challenger_config),
            status="active",
            start_date=date.today(),
            end_date=None,
            min_samples=min_samples,
            traffic_split=traffic_split,
        )
        self._experiments[experiment_id] = exp
        self._observations[experiment_id] = []
        logger.info(
            "Experiment created: id=%s name=%s split=%.0f%% champion challenger",
            experiment_id, name, traffic_split * 100,
        )
        return exp

    def assign_variant(self, match_id: str, experiment_id: str) -> str:
        """Deterministically assign a match to champion or challenger.

        Uses SHA-256(match_id + experiment_id) so the same (match, experiment)
        pair always produces the same variant — essential for reproducibility
        and preventing data leakage if the pipeline is re-run.

        Parameters
        ----------
        match_id : str
            Unique identifier for the match/event.
        experiment_id : str
            Experiment to route for.

        Returns
        -------
        str
            "challenger" or "champion"
        """
        exp = self._get_experiment(experiment_id)
        digest = hashlib.sha256(f"{match_id}|{experiment_id}".encode()).hexdigest()
        # Take the last 8 hex digits → integer in [0, 2^32)
        bucket = int(digest[-8:], 16) % 100
        threshold = int(exp.traffic_split * 100)
        return "challenger" if bucket < threshold else "champion"

    def record_outcome(
        self,
        experiment_id: str,
        match_id: str,
        variant: str,
        prediction: dict[str, Any],
        actual_outcome: str,
        odds: float,
    ) -> None:
        """Record the result of a single prediction within an experiment.

        Parameters
        ----------
        experiment_id : str
        match_id : str
        variant : str
            "champion" or "challenger"
        prediction : dict
            Must contain:
              - "selection": str — what the model picked (e.g. "H", "over")
              - "prob": float    — model probability assigned to selection
              - "closing_odds": float (optional) — for CLV calculation
        actual_outcome : str
            The true result (e.g. "H", "D", "A", "over").
        odds : float
            Decimal odds at which the bet was placed.
        """
        exp = self._get_experiment(experiment_id)
        if exp.status != "active":
            raise ValueError(
                f"Experiment {experiment_id} is {exp.status!r}, not active. "
                "Cannot record outcomes into a closed experiment."
            )
        if variant not in ("champion", "challenger"):
            raise ValueError(f"variant must be 'champion' or 'challenger', got {variant!r}")

        selection: str = prediction.get("selection", "")
        prob: float = float(prediction.get("prob", 0.5))
        closing_odds: float = float(prediction.get("closing_odds", odds))

        won = (selection == actual_outcome)
        # Profit / loss in units (stake = 1 unit):
        #   win  → odds - 1 (net profit)
        #   lose → -1.0
        profit_loss = (odds - 1.0) if won else -1.0

        # Closing Line Value: model_prob vs implied prob from closing odds
        # CLV = model_prob - (1 / closing_odds)
        # Positive CLV means we bet at better-than-closing-line value.
        closing_implied = 1.0 / closing_odds if closing_odds > 0 else 0.0
        clv = prob - closing_implied

        record = _ObservationRecord(
            match_id=match_id,
            variant=variant,
            predicted_prob=prob,
            actual_outcome=actual_outcome,
            predicted_selection=selection,
            odds=odds,
            profit_loss=profit_loss,
            clv=clv,
        )
        self._observations[experiment_id].append(record)

    def evaluate_experiment(self, experiment_id: str) -> ExperimentResult:
        """Compare champion vs challenger on accuracy, Brier, ROI, and CLV.

        Statistical significance:
        - Two-proportion z-test on accuracy
        - Welch's t-test on per-observation profit/loss (ROI proxy)

        Winner is determined by ROI t-test if significant; otherwise
        "inconclusive".

        Returns
        -------
        ExperimentResult
        """
        exp = self._get_experiment(experiment_id)
        obs = self._observations[experiment_id]

        champ_obs = [o for o in obs if o.variant == "champion"]
        chall_obs = [o for o in obs if o.variant == "challenger"]

        champ_stats = self._compute_variant_stats(champ_obs, "champion")
        chall_stats = self._compute_variant_stats(chall_obs, "challenger")

        n_champ = champ_stats["n_samples"]
        n_chall = chall_stats["n_samples"]

        # --- Primary test: ROI (profit/loss per bet) ---
        t_stat, p_value_roi = _t_test_means(
            champ_stats["roi_mean"], champ_stats["roi_std"], n_champ,
            chall_stats["roi_mean"], chall_stats["roi_std"], n_chall,
        )

        # --- Secondary test: Accuracy (z-test on proportions) ---
        z_stat, p_value_acc = _z_test_proportions(
            champ_stats["accuracy"], n_champ,
            chall_stats["accuracy"], n_chall,
        )

        is_significant = p_value_roi < 0.05
        min_met = n_champ >= exp.min_samples and n_chall >= exp.min_samples

        # Determine winner
        if not min_met:
            winner = "inconclusive"
            summary = (
                f"Insufficient data — champion: {n_champ}/{exp.min_samples}, "
                f"challenger: {n_chall}/{exp.min_samples} samples."
            )
        elif not is_significant:
            winner = "inconclusive"
            summary = (
                f"No significant ROI difference (p={p_value_roi:.3f}, α=0.05). "
                f"Champion ROI={champ_stats['roi_pct']:.1f}%, "
                f"Challenger ROI={chall_stats['roi_pct']:.1f}%."
            )
        elif chall_stats["roi_mean"] > champ_stats["roi_mean"]:
            winner = "challenger"
            summary = (
                f"Challenger wins (p={p_value_roi:.3f}). "
                f"ROI: champion={champ_stats['roi_pct']:.1f}% vs "
                f"challenger={chall_stats['roi_pct']:.1f}%."
            )
        else:
            winner = "champion"
            summary = (
                f"Champion wins (p={p_value_roi:.3f}). "
                f"ROI: champion={champ_stats['roi_pct']:.1f}% vs "
                f"challenger={chall_stats['roi_pct']:.1f}%."
            )

        # Store secondary test stats inside the variant dicts for reporting
        champ_stats["_z_stat"] = z_stat
        champ_stats["_p_value_acc"] = p_value_acc
        champ_stats["_t_stat"] = t_stat
        champ_stats["_p_value_roi"] = p_value_roi

        return ExperimentResult(
            experiment_id=experiment_id,
            champion_stats=champ_stats,
            challenger_stats=chall_stats,
            winner=winner,
            p_value=p_value_roi,
            is_significant=is_significant,
            summary=summary,
        )

    def should_promote_challenger(self, experiment_id: str) -> tuple[bool, str]:
        """Decide whether to promote the challenger to production champion.

        Promotion criteria (ALL must hold):
        1. challenger ROI > champion ROI
        2. p_value < 0.05 (ROI difference is statistically significant)
        3. Both variants have >= min_samples observations

        Anti-promotion safeguard:
        - If challenger accuracy is lower than champion accuracy, refuse
          promotion even if ROI is higher.  Higher ROI with lower accuracy
          often indicates the challenger got lucky on a few high-odds bets —
          not a genuine edge.

        Returns
        -------
        (bool, str)
            (should_promote, human-readable reason)
        """
        result = self.evaluate_experiment(experiment_id)
        exp = self._get_experiment(experiment_id)

        champ = result.champion_stats
        chall = result.challenger_stats

        n_champ = champ["n_samples"]
        n_chall = chall["n_samples"]

        # Gate 1: sufficient data
        if n_champ < exp.min_samples or n_chall < exp.min_samples:
            return False, (
                f"Insufficient data. Need {exp.min_samples} samples per variant; "
                f"have champion={n_champ}, challenger={n_chall}."
            )

        # Gate 2: statistical significance
        if not result.is_significant:
            return False, (
                f"Not statistically significant (p={result.p_value:.3f}, α=0.05). "
                "Need more data or a bigger effect to be confident."
            )

        # Gate 3: challenger ROI must be higher
        if chall["roi_mean"] <= champ["roi_mean"]:
            return False, (
                f"Champion ROI ({champ['roi_pct']:.1f}%) is not worse than "
                f"challenger ROI ({chall['roi_pct']:.1f}%). No reason to promote."
            )

        # Anti-promotion safeguard: accuracy check
        acc_diff = chall["accuracy"] - champ["accuracy"]
        if acc_diff < -0.02:  # challenger accuracy more than 2 pp lower
            return False, (
                f"Challenger ROI is higher but accuracy is lower "
                f"(champion={champ['accuracy']:.1%}, challenger={chall['accuracy']:.1%}, "
                f"delta={acc_diff:+.1%}). "
                "This may be variance from high-odds bets. Refusing promotion."
            )

        return True, (
            f"Challenger outperforms champion with 95% confidence. "
            f"Challenger ROI={chall['roi_pct']:.1f}% vs Champion ROI={champ['roi_pct']:.1f}% "
            f"(p={result.p_value:.4f}). "
            f"Accuracy: champion={champ['accuracy']:.1%}, challenger={chall['accuracy']:.1%}. "
            f"Safe to promote."
        )

    def get_active_experiments(self) -> list[Experiment]:
        """Return all experiments with status == 'active'."""
        return [e for e in self._experiments.values() if e.status == "active"]

    def complete_experiment(self, experiment_id: str) -> Experiment:
        """Mark an experiment as completed."""
        exp = self._get_experiment(experiment_id)
        exp.status = "completed"
        exp.end_date = date.today()
        logger.info("Experiment %s (%s) marked as completed.", experiment_id, exp.name)
        return exp

    def cancel_experiment(self, experiment_id: str, reason: str = "") -> Experiment:
        """Cancel a running experiment."""
        exp = self._get_experiment(experiment_id)
        exp.status = "cancelled"
        exp.end_date = date.today()
        logger.info(
            "Experiment %s (%s) cancelled. Reason: %s",
            experiment_id, exp.name, reason or "none given",
        )
        return exp

    def generate_report(self, experiment_id: str) -> str:
        """Generate a human-readable experiment report.

        Includes:
        - Experiment metadata
        - Per-variant stats table (accuracy, Brier, ROI, CLV with 95% CIs)
        - Statistical test results
        - Recommendation

        Returns
        -------
        str
            Plain-text report (suitable for Telegram, logging, or Markdown).
        """
        exp = self._get_experiment(experiment_id)
        result = self.evaluate_experiment(experiment_id)
        should_promote, reason = self.should_promote_challenger(experiment_id)

        champ = result.champion_stats
        chall = result.challenger_stats

        # --- Accuracy confidence intervals ---
        champ_acc_lo, champ_acc_hi = _confidence_interval_proportion(
            champ["accuracy"], champ["n_samples"]
        )
        chall_acc_lo, chall_acc_hi = _confidence_interval_proportion(
            chall["accuracy"], chall["n_samples"]
        )

        # --- ROI confidence intervals ---
        champ_roi_lo, champ_roi_hi = _confidence_interval_mean(
            champ["roi_mean"], champ["roi_std"], champ["n_samples"]
        )
        chall_roi_lo, chall_roi_hi = _confidence_interval_mean(
            chall["roi_mean"], chall["roi_std"], chall["n_samples"]
        )

        # --- CLV confidence intervals ---
        champ_clv_lo, champ_clv_hi = _confidence_interval_mean(
            champ["clv_mean"], champ["clv_std"], champ["n_samples"]
        )
        chall_clv_lo, chall_clv_hi = _confidence_interval_mean(
            chall["clv_mean"], chall["clv_std"], chall["n_samples"]
        )

        col_w = 22

        def row(label: str, c_val: str, h_val: str) -> str:
            return f"  {label:<20}  {c_val:<{col_w}}  {h_val:<{col_w}}\n"

        lines: list[str] = []
        lines.append("=" * 70)
        lines.append(f"  EXPERIMENT REPORT: {exp.name}")
        lines.append("=" * 70)
        lines.append(f"  ID          : {exp.id}")
        lines.append(f"  Description : {exp.description}")
        lines.append(f"  Status      : {exp.status.upper()}")
        lines.append(f"  Started     : {exp.start_date}  "
                     f"Ended: {exp.end_date or 'ongoing'}")
        lines.append(f"  Traffic split: {exp.traffic_split:.0%} to challenger")
        lines.append(f"  Min samples : {exp.min_samples} per variant")
        lines.append("")
        lines.append(
            f"  {'Metric':<20}  {'Champion':<{col_w}}  {'Challenger':<{col_w}}"
        )
        lines.append("  " + "-" * 66)
        lines.append(row(
            "Samples",
            str(champ["n_samples"]),
            str(chall["n_samples"]),
        ))
        lines.append(row(
            "Accuracy",
            f"{champ['accuracy']:.1%} [{champ_acc_lo:.1%}–{champ_acc_hi:.1%}]",
            f"{chall['accuracy']:.1%} [{chall_acc_lo:.1%}–{chall_acc_hi:.1%}]",
        ))
        lines.append(row(
            "Brier Score",
            f"{champ['brier_score']:.4f}",
            f"{chall['brier_score']:.4f}",
        ))
        lines.append(row(
            "ROI (per bet)",
            f"{champ['roi_pct']:+.1f}% [{champ_roi_lo*100:+.1f}%–{champ_roi_hi*100:+.1f}%]",
            f"{chall['roi_pct']:+.1f}% [{chall_roi_lo*100:+.1f}%–{chall_roi_hi*100:+.1f}%]",
        ))
        lines.append(row(
            "CLV (mean)",
            f"{champ['clv_mean']:+.4f} [{champ_clv_lo:+.4f}–{champ_clv_hi:+.4f}]",
            f"{chall['clv_mean']:+.4f} [{chall_clv_lo:+.4f}–{chall_clv_hi:+.4f}]",
        ))
        lines.append("")
        lines.append("  STATISTICAL TESTS")
        lines.append("  " + "-" * 66)
        t_stat = champ.get("_t_stat", 0.0)
        p_roi = champ.get("_p_value_roi", 1.0)
        z_stat = champ.get("_z_stat", 0.0)
        p_acc = champ.get("_p_value_acc", 1.0)
        lines.append(
            f"  ROI t-test (Welch):   t={t_stat:+.3f},  p={p_roi:.4f}  "
            f"{'[SIGNIFICANT]' if p_roi < 0.05 else '[not significant]'}"
        )
        lines.append(
            f"  Accuracy z-test:      z={z_stat:+.3f},  p={p_acc:.4f}  "
            f"{'[SIGNIFICANT]' if p_acc < 0.05 else '[not significant]'}"
        )
        lines.append("")
        lines.append("  VERDICT")
        lines.append("  " + "-" * 66)
        lines.append(f"  Winner      : {result.winner.upper()}")
        lines.append(f"  Summary     : {result.summary}")
        lines.append("")
        lines.append("  RECOMMENDATION")
        lines.append("  " + "-" * 66)
        action = "PROMOTE CHALLENGER" if should_promote else "KEEP CHAMPION"
        lines.append(f"  Action      : {action}")
        lines.append(f"  Reason      : {reason}")
        lines.append("")
        lines.append("  CONFIGS")
        lines.append("  " + "-" * 66)
        lines.append(f"  Champion  : {exp.champion_config}")
        lines.append(f"  Challenger: {exp.challenger_config}")
        lines.append("=" * 70)

        return "\n".join(lines)

    def suggest_experiments(self, current_config: dict[str, Any]) -> list[dict[str, Any]]:
        """Given the current model config, suggest A/B experiments to run.

        Covers the four main levers that affect edge and profitability:
        - Kelly fraction (risk sizing)
        - Confidence threshold (selectivity)
        - Draw floor probability (draw suppression)
        - Ensemble weights (model mix)

        Each suggestion is a dict with the keys needed to call
        ``create_experiment`` directly.

        Parameters
        ----------
        current_config : dict
            Current production config.  Expected keys (all optional):
            ``kelly_fraction``, ``confidence_threshold``,
            ``draw_floor``, ``ensemble_weights``.

        Returns
        -------
        list[dict]
            Each dict: {name, description, champion_config,
                        challenger_config, rationale, priority}
        """
        suggestions: list[dict[str, Any]] = []

        # -- 1. Kelly fraction variations --------------------------------
        current_kelly = current_config.get("kelly_fraction", settings.kelly_fraction)
        kelly_candidates = [0.15, 0.20, 0.25, 0.30, 0.35]
        for k in kelly_candidates:
            if abs(k - current_kelly) < 0.001:
                continue  # skip current value
            direction = "more aggressive" if k > current_kelly else "more conservative"
            suggestions.append({
                "name": f"kelly_{str(k).replace('.', '_')}_vs_{str(current_kelly).replace('.', '_')}",
                "description": (
                    f"Test Kelly fraction {k} ({direction}) vs current {current_kelly}. "
                    "Higher fractions maximise growth but increase variance; "
                    "lower fractions reduce ruin risk."
                ),
                "champion_config": {**current_config, "kelly_fraction": current_kelly},
                "challenger_config": {**current_config, "kelly_fraction": k},
                "rationale": (
                    f"Kelly={k} vs current={current_kelly}. "
                    f"Expected effect: {'higher ROI with more variance' if k > current_kelly else 'lower ROI with less variance'}."
                ),
                "priority": "high" if abs(k - current_kelly) == 0.05 else "medium",
            })

        # -- 2. Confidence threshold variations --------------------------
        current_conf = current_config.get("confidence_threshold", 0.55)
        conf_candidates = [0.50, 0.55, 0.58, 0.60, 0.62, 0.65]
        for c in conf_candidates:
            if abs(c - current_conf) < 0.001:
                continue
            direction = "stricter" if c > current_conf else "looser"
            suggestions.append({
                "name": f"conf_{str(c).replace('.', '_')}_vs_{str(current_conf).replace('.', '_')}",
                "description": (
                    f"Test confidence threshold {c} ({direction}) vs current {current_conf}. "
                    "Stricter filters reduce volume but may improve precision."
                ),
                "champion_config": {**current_config, "confidence_threshold": current_conf},
                "challenger_config": {**current_config, "confidence_threshold": c},
                "rationale": (
                    f"Threshold {c} vs {current_conf}. "
                    f"{'Fewer but higher-conviction bets.' if c > current_conf else 'More bets but potentially noisier.'}"
                ),
                "priority": "high",
            })

        # -- 3. Draw floor variations ------------------------------------
        current_floor = current_config.get("draw_floor", 0.22)
        floor_candidates = [0.18, 0.20, 0.22, 0.24, 0.26, 0.28]
        for f_val in floor_candidates:
            if abs(f_val - current_floor) < 0.001:
                continue
            direction = "raises" if f_val > current_floor else "lowers"
            suggestions.append({
                "name": f"draw_floor_{str(f_val).replace('.', '_')}_vs_{str(current_floor).replace('.', '_')}",
                "description": (
                    f"Test draw floor {f_val} ({direction} minimum draw probability) vs {current_floor}. "
                    "Draw floors prevent model over-confidence in decisive outcomes."
                ),
                "champion_config": {**current_config, "draw_floor": current_floor},
                "challenger_config": {**current_config, "draw_floor": f_val},
                "rationale": (
                    f"Draw floor {f_val} vs {current_floor}. "
                    f"{'More conservative on non-draw bets.' if f_val > current_floor else 'More aggressive on non-draw bets.'}"
                ),
                "priority": "medium",
            })

        # -- 4. Ensemble weight variations -------------------------------
        current_weights = current_config.get(
            "ensemble_weights",
            {"xgboost": 0.35, "poisson": 0.25, "starlizard": 0.25, "tabpfn": 0.15},
        )
        # Suggest boosting the best-performing base model by 10pp
        for model_name, current_w in current_weights.items():
            if current_w >= 0.55:
                continue  # already dominant
            delta = 0.10
            new_w = min(current_w + delta, 0.60)
            new_weights = {k: v for k, v in current_weights.items()}
            # Redistribute the delta equally from others
            others = [k for k in new_weights if k != model_name]
            reduction = delta / len(others) if others else 0.0
            for k in others:
                new_weights[k] = max(0.05, new_weights[k] - reduction)
            new_weights[model_name] = new_w
            # Normalise
            total = sum(new_weights.values())
            new_weights = {k: round(v / total, 4) for k, v in new_weights.items()}

            suggestions.append({
                "name": f"boost_{model_name}_weight",
                "description": (
                    f"Increase {model_name} ensemble weight from {current_w:.0%} to {new_w:.0%}, "
                    "reducing other model contributions proportionally."
                ),
                "champion_config": {**current_config, "ensemble_weights": current_weights},
                "challenger_config": {**current_config, "ensemble_weights": new_weights},
                "rationale": (
                    f"Test whether giving {model_name} more influence improves ensemble accuracy and ROI. "
                    "Suitable if this model has recently outperformed others in CLV."
                ),
                "priority": "low",
            })

        # Sort: high priority first, then alphabetically by name
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: (priority_order.get(s.get("priority", "low"), 3), s["name"]))

        return suggestions

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_experiment(self, experiment_id: str) -> Experiment:
        try:
            return self._experiments[experiment_id]
        except KeyError:
            raise KeyError(f"Experiment {experiment_id!r} not found.") from None

    def _compute_variant_stats(
        self, observations: list[_ObservationRecord], variant_name: str
    ) -> dict[str, Any]:
        """Compute all statistics for one variant's observations."""
        n = len(observations)
        if n == 0:
            return {
                "variant": variant_name,
                "n_samples": 0,
                "accuracy": 0.0,
                "brier_score": 1.0,
                "roi_mean": 0.0,
                "roi_std": 0.0,
                "roi_pct": 0.0,
                "clv_mean": 0.0,
                "clv_std": 0.0,
            }

        correct = [
            1 if o.predicted_selection == o.actual_outcome else 0
            for o in observations
        ]
        accuracy = sum(correct) / n

        # Brier: predicted_prob vs binary outcome (1 = correct prediction)
        brier = _brier_score(
            [o.predicted_prob for o in observations],
            [float(c) for c in correct],
        )

        roi_values = [o.profit_loss for o in observations]
        roi_mean, roi_std = _mean_std(roi_values)
        roi_pct = roi_mean * 100.0

        clv_values = [o.clv for o in observations]
        clv_mean, clv_std = _mean_std(clv_values)

        return {
            "variant": variant_name,
            "n_samples": n,
            "accuracy": accuracy,
            "brier_score": brier,
            "roi_mean": roi_mean,
            "roi_std": roi_std,
            "roi_pct": roi_pct,
            "clv_mean": clv_mean,
            "clv_std": clv_std,
        }
