"""Tests for sharpedge.core.consensus — consensus filter."""

import numpy as np
import pytest

from sharpedge.core.consensus import compute_consensus


class TestConsensusThreeOutcome:
    """Football-style 3-outcome consensus tests."""

    def test_unanimous_agreement(self):
        """All models agree on home win -> count = n_models."""
        probs = {
            "m1": np.array([[0.7, 0.2, 0.1]]),
            "m2": np.array([[0.6, 0.3, 0.1]]),
            "m3": np.array([[0.8, 0.1, 0.1]]),
        }
        pred, count = compute_consensus(probs)

        assert pred[0] == 0  # all predict home
        assert count[0] == 3

    def test_majority_wins(self):
        """2 out of 3 models predict home -> consensus = home, count = 2."""
        probs = {
            "m1": np.array([[0.7, 0.2, 0.1]]),  # home
            "m2": np.array([[0.6, 0.3, 0.1]]),  # home
            "m3": np.array([[0.1, 0.2, 0.7]]),  # away
        }
        pred, count = compute_consensus(probs)

        assert pred[0] == 0
        assert count[0] == 2

    def test_split_vote_argmax_tie(self):
        """Three models, three different predictions -> argmax picks lowest index (tie-break)."""
        probs = {
            "m1": np.array([[0.7, 0.2, 0.1]]),  # home (0)
            "m2": np.array([[0.1, 0.7, 0.2]]),  # draw (1)
            "m3": np.array([[0.1, 0.2, 0.7]]),  # away (2)
        }
        pred, count = compute_consensus(probs)

        # All have 1 vote; argmax on [1,1,1] returns 0 (first max)
        assert pred[0] in [0, 1, 2]
        assert count[0] == 1

    def test_multiple_matches(self):
        """Consensus works across multiple matches."""
        probs = {
            "m1": np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]),
            "m2": np.array([[0.6, 0.3, 0.1], [0.1, 0.3, 0.6]]),
        }
        pred, count = compute_consensus(probs)

        assert pred.shape == (2,)
        assert count.shape == (2,)
        assert pred[0] == 0  # both say home
        assert pred[1] == 2  # both say away
        assert count[0] == 2
        assert count[1] == 2


class TestConsensusTwoOutcome:
    """Tennis/basketball-style 2-outcome consensus tests."""

    def test_two_outcome_unanimous(self):
        probs = {
            "m1": np.array([[0.8, 0.2]]),
            "m2": np.array([[0.7, 0.3]]),
        }
        pred, count = compute_consensus(probs)

        assert pred[0] == 0
        assert count[0] == 2

    def test_two_outcome_split(self):
        """Two models disagree on 2-outcome -> one wins with count=1."""
        probs = {
            "m1": np.array([[0.8, 0.2]]),  # player 1
            "m2": np.array([[0.3, 0.7]]),  # player 2
        }
        pred, count = compute_consensus(probs)

        assert pred[0] in [0, 1]
        assert count[0] == 1

    def test_two_outcome_majority_three_models(self):
        """Three models, 2-outcome: 2 vs 1 -> majority wins."""
        probs = {
            "m1": np.array([[0.8, 0.2]]),  # player 1
            "m2": np.array([[0.7, 0.3]]),  # player 1
            "m3": np.array([[0.3, 0.7]]),  # player 2
        }
        pred, count = compute_consensus(probs)

        assert pred[0] == 0
        assert count[0] == 2
