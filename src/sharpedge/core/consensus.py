"""Consensus filter — generalized for N outcomes.

Computes per-match majority vote across models and the agreement count.
Works for any number of outcomes (2 for tennis/basketball, 3 for football, etc.).
"""

from __future__ import annotations

import numpy as np


def compute_consensus(
    model_probs: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-match consensus: how many models agree on the predicted outcome.

    Parameters
    ----------
    model_probs : dict mapping model_name -> (n_matches, n_outcomes) probability arrays.
        Works for any number of outcomes.

    Returns
    -------
    consensus_pred : (n_matches,) — the majority-voted outcome index
    consensus_count : (n_matches,) — number of models that agree on that outcome
    """
    model_names = list(model_probs.keys())
    n_matches = model_probs[model_names[0]].shape[0]
    n_outcomes = model_probs[model_names[0]].shape[1]
    n_models = len(model_names)

    # Each model's predicted outcome
    votes = np.zeros((n_models, n_matches), dtype=int)
    for i, name in enumerate(model_names):
        votes[i] = np.argmax(model_probs[name], axis=1)

    # Majority vote
    consensus_pred = np.zeros(n_matches, dtype=int)
    consensus_count = np.zeros(n_matches, dtype=int)

    for k in range(n_matches):
        match_votes = votes[:, k]
        counts = np.bincount(match_votes, minlength=n_outcomes)
        consensus_pred[k] = np.argmax(counts)
        consensus_count[k] = counts[consensus_pred[k]]

    return consensus_pred, consensus_count
