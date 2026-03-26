"""Motivation Context Scorer — intangible motivation factors."""


class MotivationScorer:
    """Scores intangible motivation factors that affect performance."""

    def compute_features(self, sport: str, is_derby: bool = False,
                         is_relegation_battle: bool = False,
                         is_title_race: bool = False,
                         is_dead_rubber: bool = False,
                         is_revenge_game: bool = False,
                         is_playoff: bool = False,
                         league_position_home: int = 10,
                         league_position_away: int = 10) -> dict[str, float]:

        motivation_home = 0.5  # baseline
        motivation_away = 0.5

        if is_derby:
            motivation_home += 0.1
            motivation_away += 0.1
        if is_relegation_battle:
            # Team fighting relegation plays harder
            if league_position_home > 15:
                motivation_home += 0.15
            if league_position_away > 15:
                motivation_away += 0.15
        if is_title_race:
            if league_position_home <= 3:
                motivation_home += 0.1
            if league_position_away <= 3:
                motivation_away += 0.1
        if is_dead_rubber:
            motivation_home -= 0.15
            motivation_away -= 0.15
        if is_revenge_game:
            motivation_away += 0.05
        if is_playoff:
            motivation_home += 0.1
            motivation_away += 0.1

        motivation_home = max(0.1, min(1.0, motivation_home))
        motivation_away = max(0.1, min(1.0, motivation_away))

        return {
            "mot_home_motivation": motivation_home,
            "mot_away_motivation": motivation_away,
            "mot_motivation_diff": motivation_home - motivation_away,
            "mot_derby_flag": 1.0 if is_derby else 0.0,
            "mot_dead_rubber_flag": 1.0 if is_dead_rubber else 0.0,
            "mot_playoff_flag": 1.0 if is_playoff else 0.0,
        }
