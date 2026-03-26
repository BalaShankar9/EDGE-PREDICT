"""Travel Fatigue Modeler — quantifies travel impact on performance."""


class TravelFatigueModeler:
    """Models travel fatigue across sports."""

    # Timezone impact (points/goals) from research
    NBA_TIMEZONE_IMPACT = 1.5  # points per timezone crossed
    NFL_EAST_EARLY_PENALTY = 2.0  # points for west->east early game

    def compute_features(self, sport: str, timezone_diff: int = 0,
                         travel_km: float = 0, back_to_back: bool = False,
                         days_rest: int = 3) -> dict[str, float]:
        features = {
            "tf_timezone_diff": abs(timezone_diff),
            "tf_travel_km": travel_km,
            "tf_back_to_back": 1.0 if back_to_back else 0.0,
            "tf_days_rest": days_rest,
            "tf_fatigue_index": 0.0,
        }

        # Compute composite fatigue index
        fatigue = 0.0
        fatigue += abs(timezone_diff) * 0.1
        fatigue += (1.0 if back_to_back else 0.0) * 0.3
        fatigue += max(0, 3 - days_rest) * 0.15
        fatigue += min(1.0, travel_km / 5000) * 0.2
        features["tf_fatigue_index"] = min(1.0, fatigue)

        return features
