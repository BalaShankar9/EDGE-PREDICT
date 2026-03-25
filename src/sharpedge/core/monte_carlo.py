"""Monte Carlo match simulation engine — generalized for N outcomes.

Supports any number of outcomes per match (2 for tennis/basketball,
3 for football, N for any sport).
"""

from __future__ import annotations

import numpy as np


def monte_carlo_simulate(
    model_probs: dict[str, np.ndarray],
    n_sims: int = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Monte Carlo simulation across all models for a batch of matches.

    For each match and each simulation:
      1. Randomly pick one of the models (uniform)
      2. Use that model's probability vector to sample an outcome
      3. Aggregate across all sims to get robust probability estimates

    This captures model uncertainty — if models disagree, the MC simulation
    naturally produces wider confidence intervals and lower peak probabilities,
    which means fewer (but more reliable) bets.

    Parameters
    ----------
    model_probs : dict mapping model_name -> (n_matches, n_outcomes) probability arrays.
        Works for any number of outcomes (2, 3, or more).
    n_sims : number of simulations per match

    Returns
    -------
    mc_probs : (n_matches, n_outcomes) — MC-estimated outcome probabilities
    mc_confidence : (n_matches,) — confidence = max prob (higher = more certain)
    mc_std : (n_matches, n_outcomes) — standard deviation across bootstrap samples
    """
    model_names = list(model_probs.keys())
    n_models = len(model_names)
    n_matches = model_probs[model_names[0]].shape[0]
    n_outcomes = model_probs[model_names[0]].shape[1]

    # Stack all model predictions: (n_models, n_matches, n_outcomes)
    all_preds = np.stack([model_probs[name] for name in model_names], axis=0)

    # For each match, run MC simulation
    mc_counts = np.zeros((n_matches, n_outcomes), dtype=np.float64)

    # Vectorized: sample model indices and outcomes in bulk
    # model_choices: (n_sims, n_matches) — which model to use
    model_choices = np.random.randint(0, n_models, size=(n_sims, n_matches))
    match_indices = np.arange(n_matches)

    for sim in range(n_sims):
        # For each match, get the probabilities from the chosen model
        chosen_probs = all_preds[model_choices[sim], match_indices]  # (n_matches, n_outcomes)
        # Sample outcomes using cumulative probability + uniform random
        cumprobs = np.cumsum(chosen_probs, axis=1)
        rands = np.random.random(n_matches)
        # Generalized for N outcomes:
        # outcome = number of cumulative probability thresholds the random value exceeds
        outcomes = (rands[:, None] >= cumprobs[:, :-1]).sum(axis=1)
        # Count
        for outcome in range(n_outcomes):
            mc_counts[:, outcome] += (outcomes == outcome)

    mc_probs = mc_counts / n_sims

    # Confidence = peak probability
    mc_confidence = np.max(mc_probs, axis=1)

    # Bootstrap std: split sims into 10 blocks, compute std across blocks
    block_size = max(1, n_sims // 10)
    block_probs = []
    for b in range(10):
        start = b * block_size
        end = min(start + block_size, n_sims)
        if start >= n_sims:
            break
        block_counts = np.zeros((n_matches, n_outcomes))
        for sim in range(start, end):
            chosen_probs = all_preds[model_choices[sim], match_indices]
            cumprobs = np.cumsum(chosen_probs, axis=1)
            rands = np.random.random(n_matches)
            outcomes = (rands[:, None] >= cumprobs[:, :-1]).sum(axis=1)
            for outcome in range(n_outcomes):
                block_counts[:, outcome] += (outcomes == outcome)
        block_probs.append(block_counts / (end - start))

    mc_std = np.std(block_probs, axis=0)

    return mc_probs, mc_confidence, mc_std
