# src/sharpedge/ml/training/metrics.py
"""Prediction quality metrics for football models.

Primary metric: Ranked Probability Score (RPS) — penalises confident wrong predictions.
Secondary: log-loss, accuracy, ROI, CLV.
"""
import numpy as np
from numpy.typing import NDArray


def ranked_probability_score(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
) -> float:
    """Compute mean Ranked Probability Score for ordered categorical predictions.

    RPS measures the distance between predicted cumulative probabilities
    and observed cumulative outcomes. Lower is better. Range: [0, 1].

    Works for any number of outcome classes (2 for tennis, 3 for football, etc.).

    Parameters
    ----------
    y_true : array of shape (n_samples,)
        True outcomes as integer class indices (0, 1, ..., K-1).
    y_prob : array of shape (n_samples, K)
        Predicted probabilities for each of the K outcomes.

    Returns
    -------
    float
        Mean RPS across all samples.
    """
    n = len(y_true)
    k = y_prob.shape[1]  # number of outcome classes
    rps_sum = 0.0
    for i in range(n):
        outcome = np.zeros(k)
        outcome[y_true[i]] = 1.0
        cum_pred = np.cumsum(y_prob[i])
        cum_true = np.cumsum(outcome)
        rps_sum += np.sum((cum_pred - cum_true) ** 2) / (k - 1)
    return rps_sum / n


def accuracy(y_true: NDArray[np.int_], y_pred: NDArray[np.int_]) -> float:
    """Simple prediction accuracy (fraction correct)."""
    return float(np.mean(y_true == y_pred))


def log_loss_1x2(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    eps: float = 1e-15,
) -> float:
    """Multi-class log-loss for 1X2 predictions."""
    n = len(y_true)
    y_prob = np.clip(y_prob, eps, 1 - eps)
    loss = 0.0
    for i in range(n):
        loss -= np.log(y_prob[i, y_true[i]])
    return loss / n


def roi(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    odds: NDArray[np.float64],
    stake: float = 1.0,
) -> float:
    """Return on Investment for flat-stake betting.

    Parameters
    ----------
    y_true : outcomes (0=H, 1=D, 2=A)
    y_pred : predicted outcomes (0=H, 1=D, 2=A)
    odds : array of shape (n_samples, 3) with [home_odds, draw_odds, away_odds]
    stake : flat stake per bet
    """
    total_staked = 0.0
    total_return = 0.0
    for i in range(len(y_true)):
        total_staked += stake
        if y_pred[i] == y_true[i]:
            total_return += stake * odds[i, y_pred[i]]
    if total_staked == 0:
        return 0.0
    return (total_return - total_staked) / total_staked


def closing_line_value(
    pick_odds: NDArray[np.float64],
    closing_odds: NDArray[np.float64],
) -> float:
    """Mean Closing Line Value percentage.

    CLV > 0 means we consistently get better odds than the closing line.

    CLV% = mean((closing_implied / pick_implied) - 1) * 100
    """
    pick_implied = 1.0 / pick_odds
    closing_implied = 1.0 / closing_odds
    clv = np.mean((closing_implied / pick_implied) - 1) * 100
    return float(clv)


def calibration_error(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error for binary predictions.

    Groups predictions into bins and measures how well predicted
    probabilities match observed frequencies.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for j in range(n_bins):
        mask = (y_prob >= bin_edges[j]) & (y_prob < bin_edges[j + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / total * abs(bin_acc - bin_conf)
    return float(ece)
