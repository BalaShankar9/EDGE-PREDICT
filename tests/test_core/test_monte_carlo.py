"""Tests for sharpedge.core.monte_carlo — Monte Carlo simulation engine."""

import numpy as np
import pytest

from sharpedge.core.monte_carlo import monte_carlo_simulate


class TestMonteCarloThreeOutcome:
    """Football-style 3-outcome tests (H/D/A)."""

    def test_output_shapes(self):
        probs = {
            "model_a": np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]]),
            "model_b": np.array([[0.6, 0.2, 0.2], [0.3, 0.3, 0.4]]),
        }
        mc_probs, mc_conf, mc_std = monte_carlo_simulate(probs, n_sims=1000)

        assert mc_probs.shape == (2, 3)
        assert mc_conf.shape == (2,)
        assert mc_std.shape == (2, 3)

    def test_probabilities_sum_to_one(self):
        probs = {
            "model_a": np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]]),
            "model_b": np.array([[0.6, 0.2, 0.2], [0.3, 0.3, 0.4]]),
        }
        mc_probs, _, _ = monte_carlo_simulate(probs, n_sims=5000)

        np.testing.assert_allclose(mc_probs.sum(axis=1), 1.0, atol=1e-10)

    def test_unanimous_models_high_confidence(self):
        """When all models strongly agree, confidence should be high."""
        np.random.seed(42)
        strong_home = np.array([[0.9, 0.05, 0.05]])
        probs = {
            "m1": strong_home,
            "m2": strong_home,
            "m3": strong_home,
        }
        mc_probs, mc_conf, mc_std = monte_carlo_simulate(probs, n_sims=5000)

        assert mc_conf[0] > 0.8
        assert mc_std[0, 0] < 0.05  # low std for the dominant outcome

    def test_disagreeing_models_lower_confidence(self):
        """When models disagree, confidence should be lower."""
        np.random.seed(42)
        probs = {
            "m1": np.array([[0.8, 0.1, 0.1]]),  # strongly home
            "m2": np.array([[0.1, 0.1, 0.8]]),  # strongly away
            "m3": np.array([[0.1, 0.8, 0.1]]),  # strongly draw
        }
        _, mc_conf_disagree, mc_std_disagree = monte_carlo_simulate(probs, n_sims=5000)

        # Compare with unanimous case
        strong = np.array([[0.9, 0.05, 0.05]])
        probs_agree = {"m1": strong, "m2": strong, "m3": strong}
        _, mc_conf_agree, mc_std_agree = monte_carlo_simulate(probs_agree, n_sims=5000)

        assert mc_conf_disagree[0] < mc_conf_agree[0]
        # Std should be higher when models disagree
        assert mc_std_disagree.max() > mc_std_agree.max()


class TestMonteCarloTwoOutcome:
    """Tennis/basketball-style 2-outcome tests."""

    def test_output_shapes(self):
        probs = {
            "model_a": np.array([[0.7, 0.3], [0.4, 0.6]]),
            "model_b": np.array([[0.6, 0.4], [0.5, 0.5]]),
        }
        mc_probs, mc_conf, mc_std = monte_carlo_simulate(probs, n_sims=1000)

        assert mc_probs.shape == (2, 2)
        assert mc_conf.shape == (2,)
        assert mc_std.shape == (2, 2)

    def test_probabilities_sum_to_one(self):
        probs = {
            "model_a": np.array([[0.7, 0.3], [0.4, 0.6]]),
            "model_b": np.array([[0.6, 0.4], [0.5, 0.5]]),
        }
        mc_probs, _, _ = monte_carlo_simulate(probs, n_sims=5000)

        np.testing.assert_allclose(mc_probs.sum(axis=1), 1.0, atol=1e-10)

    def test_strong_favorite(self):
        """Strong 2-outcome favorite should produce high confidence."""
        np.random.seed(42)
        probs = {
            "m1": np.array([[0.9, 0.1]]),
            "m2": np.array([[0.85, 0.15]]),
        }
        mc_probs, mc_conf, _ = monte_carlo_simulate(probs, n_sims=5000)

        assert mc_conf[0] > 0.8
        assert mc_probs[0, 0] > mc_probs[0, 1]


class TestMonteCarloSingleModel:
    """Edge case: single model should approximate that model's probs."""

    def test_single_model_3_outcome(self):
        np.random.seed(42)
        input_probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
        probs = {"only_model": input_probs}
        mc_probs, _, _ = monte_carlo_simulate(probs, n_sims=10000)

        # With enough sims, MC output should be close to input
        np.testing.assert_allclose(mc_probs, input_probs, atol=0.03)

    def test_single_model_2_outcome(self):
        np.random.seed(42)
        input_probs = np.array([[0.7, 0.3], [0.4, 0.6]])
        probs = {"only_model": input_probs}
        mc_probs, _, _ = monte_carlo_simulate(probs, n_sims=10000)

        np.testing.assert_allclose(mc_probs, input_probs, atol=0.03)
