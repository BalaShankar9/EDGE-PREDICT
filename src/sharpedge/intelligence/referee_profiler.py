"""Referee/Umpire Profiler — build tendency profiles.

Referee assignment is known pre-match but rarely priced into odds.
"""
from collections import defaultdict

import numpy as np


class RefereeProfiler:
    def __init__(self):
        self._profiles: dict[str, dict] = defaultdict(lambda: {
            "matches": 0, "home_wins": 0, "total_cards": 0,
            "total_fouls": 0, "total_penalties": 0,
        })

    def update(self, referee: str, home_win: bool, cards: int = 0, fouls: int = 0, penalties: int = 0):
        p = self._profiles[referee]
        p["matches"] += 1
        p["home_wins"] += int(home_win)
        p["total_cards"] += cards
        p["total_fouls"] += fouls
        p["total_penalties"] += penalties

    def compute_features(self, referee: str | None) -> dict[str, float]:
        if referee is None or referee not in self._profiles:
            return {
                "ref_home_bias": 0.5, "ref_cards_per_match": 4.0,
                "ref_fouls_per_match": 22.0, "ref_penalty_rate": 0.0,
                "ref_strictness": 0.5,
            }

        p = self._profiles[referee]
        n = max(p["matches"], 1)
        cards_pm = p["total_cards"] / n
        fouls_pm = p["total_fouls"] / n

        return {
            "ref_home_bias": p["home_wins"] / n,
            "ref_cards_per_match": cards_pm,
            "ref_fouls_per_match": fouls_pm,
            "ref_penalty_rate": p["total_penalties"] / n,
            "ref_strictness": min(1.0, cards_pm / 6.0),  # normalized: 6 cards/match = max strict
        }
