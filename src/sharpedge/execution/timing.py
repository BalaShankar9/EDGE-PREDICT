"""Bet Timing Optimizer -- when to bet matters as much as what to bet."""


class BetTimingOptimizer:
    """Learns optimal bet placement timing per sport/league."""

    # Default optimal timing (hours before match start)
    DEFAULT_TIMING = {
        "football": 48,     # bet 2 days before (line opening)
        "tennis": 4,        # bet 4 hours before (injury/weather news)
        "basketball": 1,    # bet 1 hour before (lineup confirmations)
        "ice_hockey": 2,
        "american_football": 72,  # bet 3 days before (NFL lines move early)
        "baseball": 6,      # bet 6 hours before (starting pitcher confirmed)
    }

    def optimal_bet_time(self, sport: str, league: str = "") -> int:
        """Returns optimal hours before match to place bet."""
        return self.DEFAULT_TIMING.get(sport, 24)

    def compute_features(self, sport: str, hours_to_match: float) -> dict[str, float]:
        optimal = self.optimal_bet_time(sport)
        return {
            "bt_hours_to_match": hours_to_match,
            "bt_optimal_hours": float(optimal),
            "bt_timing_score": max(0, 1.0 - abs(hours_to_match - optimal) / optimal),
            "bt_early_bet": 1.0 if hours_to_match > optimal * 1.5 else 0.0,
            "bt_late_bet": 1.0 if hours_to_match < optimal * 0.3 else 0.0,
        }
