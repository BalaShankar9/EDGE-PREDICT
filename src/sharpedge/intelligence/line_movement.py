"""Line Movement Tracker — tracks odds movements over time.

A steam move (sharp money) is when the line moves despite public betting
the other direction. This is the strongest signal in sports betting.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LineSnapshot:
    """A single odds snapshot at a point in time."""
    match_id: str
    timestamp: datetime
    bookmaker: str
    market: str
    odds: dict[str, float]  # {"home": 1.8, "draw": 3.5, "away": 4.2}


@dataclass
class LineMovement:
    """Computed line movement features for a match."""
    match_id: str
    opening_odds: dict[str, float]
    current_odds: dict[str, float]
    closing_odds: dict[str, float] | None  # filled post-match
    movement: dict[str, float]  # change in implied prob per outcome
    steam_move: bool  # True if line moved against public money
    reverse_line_move: bool  # True if line moved opposite to expected direction


class LineMovementTracker:
    """Tracks and analyzes line movements across matches."""

    def __init__(self):
        self._snapshots: dict[str, list[LineSnapshot]] = {}  # match_id -> snapshots

    def record_snapshot(self, snapshot: LineSnapshot) -> None:
        if snapshot.match_id not in self._snapshots:
            self._snapshots[snapshot.match_id] = []
        self._snapshots[snapshot.match_id].append(snapshot)

    def get_movement(self, match_id: str) -> LineMovement | None:
        snapshots = self._snapshots.get(match_id, [])
        if len(snapshots) < 2:
            return None

        sorted_snaps = sorted(snapshots, key=lambda s: s.timestamp)
        opening = sorted_snaps[0].odds
        current = sorted_snaps[-1].odds

        # Compute implied probability movement
        movement = {}
        for outcome in opening:
            open_impl = 1.0 / opening[outcome] if opening[outcome] > 0 else 0
            curr_impl = 1.0 / current[outcome] if current[outcome] > 0 else 0
            movement[outcome] = curr_impl - open_impl

        # Steam move detection (simplified): line shortened by >3% implied
        steam_move = any(abs(v) > 0.03 for v in movement.values())
        # Reverse line move: favorite became less favored
        fav = max(opening, key=lambda o: 1.0/opening[o] if opening[o] > 0 else 0)
        reverse_line_move = movement.get(fav, 0) < -0.02

        return LineMovement(
            match_id=match_id,
            opening_odds=opening,
            current_odds=current,
            closing_odds=None,
            movement=movement,
            steam_move=steam_move,
            reverse_line_move=reverse_line_move,
        )

    def compute_features(self, match_id: str) -> dict[str, float]:
        """Compute features from line movement for a match."""
        mv = self.get_movement(match_id)
        if mv is None:
            return {
                "lm_home_movement": 0.0, "lm_away_movement": 0.0,
                "lm_steam_move": 0.0, "lm_reverse_line_move": 0.0,
                "lm_total_movement": 0.0,
            }

        home_mv = mv.movement.get("home", 0.0)
        away_mv = mv.movement.get("away", mv.movement.get("player2", 0.0))

        return {
            "lm_home_movement": home_mv,
            "lm_away_movement": away_mv,
            "lm_steam_move": 1.0 if mv.steam_move else 0.0,
            "lm_reverse_line_move": 1.0 if mv.reverse_line_move else 0.0,
            "lm_total_movement": abs(home_mv) + abs(away_mv),
        }
