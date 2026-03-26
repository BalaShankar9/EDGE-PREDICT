"""Injury Impact Modeler — models the IMPACT of injuries, not just presence.

Unlike the old InjuryFeatures (always zero during training), this module
models injury impact as a proportion of team quality that is missing.
"""


class InjuryImpactModeler:
    """Models injury impact across sports."""

    def compute_features(self, home_injuries: list[dict] = None,
                         away_injuries: list[dict] = None) -> dict[str, float]:
        """
        Parameters
        ----------
        home_injuries, away_injuries: list of dicts with:
            - player: str
            - minutes_share: float (0-1, fraction of team's minutes this player plays)
            - quality_rating: float (0-100, player quality)
            - status: "out" | "doubtful" | "questionable"
        """
        home_injuries = home_injuries or []
        away_injuries = away_injuries or []

        home_impact = self._compute_impact(home_injuries)
        away_impact = self._compute_impact(away_injuries)

        return {
            "inj_home_impact": home_impact,
            "inj_away_impact": away_impact,
            "inj_impact_diff": away_impact - home_impact,  # positive = home advantage
            "inj_home_key_player_out": 1.0 if any(
                i.get("minutes_share", 0) > 0.15 and i.get("status") == "out"
                for i in home_injuries
            ) else 0.0,
            "inj_away_key_player_out": 1.0 if any(
                i.get("minutes_share", 0) > 0.15 and i.get("status") == "out"
                for i in away_injuries
            ) else 0.0,
        }

    def _compute_impact(self, injuries: list[dict]) -> float:
        """Compute total injury impact as weighted sum of missing quality."""
        if not injuries:
            return 0.0

        total = 0.0
        for inj in injuries:
            minutes = inj.get("minutes_share", 0.0)
            quality = inj.get("quality_rating", 50) / 100.0
            status = inj.get("status", "out")

            # Weight by probability of missing
            miss_prob = {"out": 1.0, "doubtful": 0.7, "questionable": 0.3}.get(status, 0.5)
            total += minutes * quality * miss_prob

        return min(1.0, total)
