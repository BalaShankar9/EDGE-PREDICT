"""Confidence tier assignment for banker picks.

Platinum: prob >= 0.85, edge >= 0.10, 0 risk flags
Gold: prob >= 0.78, edge >= 0.07, <= 1 minor flag
Silver: prob >= 0.70, edge >= 0.05
"""
from sharpedge.ml.banker.filter import Pick


class TierAssigner:
    """Assigns confidence tiers to picks."""

    def assign(self, picks: list[Pick]) -> list[Pick]:
        """Assign tier to each pick based on confidence criteria."""
        for pick in picks:
            minor_flags = [f for f in pick.risk_flags if not f.startswith("CRITICAL")]

            if (pick.model_prob >= 0.85
                    and pick.edge >= 0.10
                    and len(pick.risk_flags) == 0):
                pick.tier = "platinum"
            elif (pick.model_prob >= 0.78
                  and pick.edge >= 0.07
                  and len(minor_flags) <= 1):
                pick.tier = "gold"
            else:
                pick.tier = "silver"

        return picks
