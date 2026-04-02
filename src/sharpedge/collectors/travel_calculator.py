"""
TravelCalculator — venue distance and travel fatigue modeling.

Pure computation module: no HTTP requests, no scraping. Uses the Haversine
formula to compute great-circle distances between stadium coordinates and
translates distances into calibrated fatigue multipliers.

Academic basis
--------------
Fatigue thresholds are informed by sports science literature on travel
load in professional football (Carling et al. 2012, Leatherwood & Dragoo
2013). The scoring bands reflect typical bus vs. flight travel boundaries
in European football.

Usage
-----
    from sharpedge.collectors.travel_calculator import TravelCalculator

    tc = TravelCalculator()

    # Basic distance
    km = tc.get_travel_distance("Arsenal", "Liverpool")   # -> ~320 km

    # Fatigue multiplier (0.0 = no fatigue, 1.0 = extreme)
    f  = tc.get_travel_fatigue_factor("PSG", "Napoli")    # -> 0.35

    # European midweek → domestic weekend analysis
    burden = tc.get_european_travel_fatigue(
        team="Chelsea",
        midweek_opponent="Real Madrid",
        weekend_opponent="Man City",
    )
"""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


class TravelCalculator:
    """Compute travel distances and fatigue factors between European venues.

    All coordinates represent the stadium location (lat, lon in decimal degrees).
    Distance calculations use the Haversine formula for great-circle distance.
    """

    # ------------------------------------------------------------------ #
    #  Venue coordinates — (latitude, longitude) in decimal degrees       #
    # ------------------------------------------------------------------ #

    VENUE_COORDINATES: dict[str, tuple[float, float]] = {
        # ---- Premier League ----
        "Arsenal":       (51.5549, -0.1084),   # Emirates Stadium
        "Aston Villa":   (52.5092, -1.8847),   # Villa Park
        "Bournemouth":   (50.7352, -1.8384),   # Vitality Stadium
        "Brentford":     (51.4907, -0.2886),   # GTech Community Stadium
        "Brighton":      (50.8616, -0.0837),   # Amex Stadium
        "Chelsea":       (51.4817, -0.1910),   # Stamford Bridge
        "Crystal Palace": (51.3983, -0.0855),  # Selhurst Park
        "Everton":       (53.4387, -2.9663),   # Goodison Park
        "Fulham":        (51.4749, -0.2217),   # Craven Cottage
        "Ipswich":       (52.0554,  1.1449),   # Portman Road
        "Leicester":     (52.6203, -1.1422),   # King Power Stadium
        "Liverpool":     (53.4308, -2.9608),   # Anfield
        "Man City":      (53.4831, -2.2004),   # Etihad Stadium
        "Man United":    (53.4631, -2.2913),   # Old Trafford
        "Newcastle":     (54.9756, -1.6217),   # St James' Park
        "Nottm Forest":  (52.9399, -1.1328),   # City Ground
        "Southampton":   (50.9058, -1.3910),   # St Mary's Stadium
        "Tottenham":     (51.6043, -0.0661),   # Tottenham Hotspur Stadium
        "West Ham":      (51.5387,  0.0166),   # London Stadium
        "Wolves":        (52.5901, -2.1306),   # Molineux Stadium
        # ---- Championship ----
        "Leeds":         (53.7774, -1.5721),   # Elland Road
        "Sheffield United": (53.3703, -1.4706), # Bramall Lane
        "Burnley":       (53.7892, -2.2294),   # Turf Moor
        "Middlesbrough": (54.5784, -1.2175),   # Riverside Stadium
        "Sunderland":    (54.9148, -1.3884),   # Stadium of Light
        "Norwich":       (52.6221,  1.3094),   # Carrow Road
        "Watford":       (51.6498, -0.4016),   # Vicarage Road
        # ---- La Liga ----
        "Barcelona":       (41.3809,  2.1228), # Camp Nou / Estadi Olimpic
        "Real Madrid":     (40.4531, -3.6883), # Santiago Bernabeu
        "Atletico Madrid": (40.4362, -3.5995), # Civitas Metropolitano
        "Sevilla":         (37.3840, -5.9706), # Ramon Sanchez-Pizjuan
        "Valencia":        (39.4745, -0.3583), # Mestalla
        "Real Betis":      (37.3564, -5.9817), # Estadio Benito Villamarin
        "Real Sociedad":   (43.3017, -1.9736), # Reale Arena
        "Athletic Bilbao": (43.2641, -2.9494), # San Mames
        "Villarreal":      (39.9442, -0.1036), # Estadio de la Ceramica
        "Osasuna":         (42.7966, -1.6367), # El Sadar
        "Getafe":          (40.3261, -3.7169), # Coliseum Alfonso Perez
        "Celta Vigo":      (42.2118, -8.7396), # Balaidos
        "Alaves":          (42.8464, -2.6771), # Mendizorroza
        "Girona":          (41.9631,  2.8248), # Montilivi
        "Mallorca":        (39.5900,  2.6455), # Visit Mallorca Estadi
        "Las Palmas":      (28.0997, -15.4503), # Gran Canaria
        "Rayo Vallecano":  (40.3920, -3.6607), # Estadio de Vallecas
        "Espanyol":        (41.3474,  2.0748), # RCDE Stadium
        # ---- Bundesliga ----
        "Bayern Munich":       (48.2188, 11.6247), # Allianz Arena
        "Dortmund":            (51.4926,  7.4518), # Signal Iduna Park
        "RB Leipzig":          (51.3459, 12.3487), # Red Bull Arena
        "Bayer Leverkusen":    (51.0382,  7.0020), # BayArena
        "Eintracht Frankfurt": (50.0687,  8.6454), # Deutsche Bank Park
        "VfB Stuttgart":       (48.7925,  9.2324), # MHPArena
        "Freiburg":            (48.0220,  7.8829), # Europa-Park Stadion
        "Wolfsburg":           (52.4327, 10.8029), # Volkswagen Arena
        "Mainz":               (49.9841,  8.2236), # MEWA ARENA
        "Augsburg":            (48.3234, 10.8863), # WWK Arena
        "Werder Bremen":       (53.0665,  8.8376), # Weserstadion
        "Hoffenheim":          (49.2386,  8.8889), # PreZero Arena
        "Union Berlin":        (52.4577, 13.5680), # An der Alten Forsterei
        "Borussia Monchengladbach": (51.1744,  6.3852), # Borussia Park
        "Cologne":             (50.9336,  6.8749), # RheinEnergieStadion
        "Hertha Berlin":       (52.5147, 13.2395), # Olympiastadion Berlin
        # ---- Serie A ----
        "Juventus":     (45.1096,  7.6413), # Allianz Stadium
        "AC Milan":     (45.4781,  9.1240), # San Siro
        "Inter Milan":  (45.4781,  9.1240), # San Siro (shared)
        "Napoli":       (40.8280, 14.1931), # Stadio Diego Armando Maradona
        "Roma":         (41.9341, 12.4547), # Stadio Olimpico
        "Lazio":        (41.9341, 12.4547), # Stadio Olimpico (shared)
        "Atalanta":     (45.7096,  9.6801), # Gewiss Stadium
        "Fiorentina":   (43.7808, 11.2825), # Stadio Artemio Franchi
        "Bologna":      (44.4958, 11.3137), # Stadio Renato Dall'Ara
        "Torino":       (45.0402,  7.6503), # Stadio Olimpico Grande Torino
        "Udinese":      (46.0764, 13.2003), # Bluenergy Stadium
        "Sassuolo":     (44.7009, 10.8921), # Mapei Stadium
        "Verona":       (45.4348, 10.9732), # Stadio Marc'Antonio Bentegodi
        "Empoli":       (43.7229, 10.9456), # Stadio Carlo Castellani
        "Salernitana":  (40.6726, 14.7853), # Stadio Arechi
        "Lecce":        (40.3592, 18.1800), # Via del Mare
        "Monza":        (45.5871,  9.2819), # U-Power Stadium
        "Venezia":      (45.4787, 12.2559), # Pier Luigi Penzo
        # ---- Ligue 1 ----
        "PSG":       (48.8414,  2.2530), # Parc des Princes
        "Marseille": (43.2698,  5.3958), # Orange Velodrome
        "Lyon":      (45.7652,  4.9822), # Groupama Stadium
        "Monaco":    (43.7277,  7.4154), # Stade Louis II
        "Lille":     (50.6120,  3.1303), # Stade Pierre-Mauroy
        "Rennes":    (48.1073, -1.7194), # Roazhon Park
        "Nice":      (43.7052,  7.1928), # Allianz Riviera
        "Lens":      (50.4338,  2.8151), # Stade Bollaert-Delelis
        "Strasbourg": (48.5600,  7.7532), # Stade de la Meinau
        "Nantes":    (47.2559, -1.5242), # Stade de la Beaujoire
        "Montpellier": (43.6221,  3.8137), # Stade de la Mosson
        "Brest":     (48.4180, -4.4694), # Stade Francis-Le Ble
        "Toulouse":  (43.5830,  1.4343), # Stadium de Toulouse
        "Reims":     (49.2539,  4.0264), # Stade Auguste-Delaune
        # ---- Eredivisie ----
        "Ajax":          (52.3143,  4.9415), # Johan Cruyff Arena
        "PSV Eindhoven": (51.4416,  5.4674), # Philips Stadion
        "Feyenoord":     (51.8939,  4.5225), # De Kuip
        "AZ Alkmaar":    (52.6069,  4.7384), # AFAS Stadion
        "Twente":        (52.2369,  6.8494), # De Grolsch Veste
        "Vitesse":       (51.9602,  5.9278), # GelreDome
        # ---- Portugal ----
        "Benfica":        (38.7524, -9.1847), # Estadio da Luz
        "Porto":          (41.1614, -8.5836), # Estadio do Dragao
        "Sporting CP":    (38.7613, -9.1604), # Estadio Jose Alvalade
        # ---- Scotland ----
        "Celtic":  (55.8495, -4.2057), # Celtic Park
        "Rangers": (55.8511, -4.3095), # Ibrox Stadium
        # ---- Turkey ----
        "Galatasaray":   (41.1049, 29.0215), # RAMS Park
        "Fenerbahce":    (40.9794, 29.0364), # Ulker Stadium
        "Besiktas":      (41.0431, 29.0085), # Vodafone Park
    }

    # ------------------------------------------------------------------ #
    #  Fatigue distance bands (km lower bound → fatigue multiplier)       #
    # ------------------------------------------------------------------ #

    _FATIGUE_BANDS: list[tuple[float, float]] = [
        (1000.0, 0.50),   # Major international travel
        (500.0,  0.35),   # Significant travel (flight needed)
        (300.0,  0.20),   # Moderate (possible overnight)
        (100.0,  0.10),   # Short domestic trip
        (0.0,    0.00),   # Local derby
    ]

    # ------------------------------------------------------------------ #
    #  Core calculations                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def haversine_distance(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate great-circle distance in kilometres using Haversine formula.

        Parameters
        ----------
        lat1, lon1 : float
            Latitude and longitude of point 1 (decimal degrees).
        lat2, lon2 : float
            Latitude and longitude of point 2 (decimal degrees).

        Returns
        -------
        float
            Distance in kilometres, rounded to 1 decimal place.
        """
        R = 6371.0  # Earth's mean radius in km

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(d_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return round(R * c, 1)

    def _resolve_team(self, team: str) -> Optional[tuple[float, float]]:
        """Look up venue coordinates, with case-insensitive fuzzy fallback."""
        # Exact match first
        if team in self.VENUE_COORDINATES:
            return self.VENUE_COORDINATES[team]

        # Case-insensitive match
        lower = team.lower()
        for key, coords in self.VENUE_COORDINATES.items():
            if key.lower() == lower:
                return coords

        # Partial match — e.g. "Man Utd" -> "Man United"
        for key, coords in self.VENUE_COORDINATES.items():
            if lower in key.lower() or key.lower() in lower:
                logger.debug(
                    f"[TravelCalculator] Fuzzy match: '{team}' -> '{key}'"
                )
                return coords

        logger.warning(
            f"[TravelCalculator] No venue coordinates found for '{team}'"
        )
        return None

    def get_travel_distance(
        self, home_team: str, away_team: str
    ) -> Optional[float]:
        """Return the straight-line distance in km between two team venues.

        Parameters
        ----------
        home_team : str
            Name of the home team (must match VENUE_COORDINATES key or fuzzy).
        away_team : str
            Name of the away team.

        Returns
        -------
        float or None
            Distance in km, or None if either team is not found.
        """
        home_coords = self._resolve_team(home_team)
        away_coords = self._resolve_team(away_team)

        if home_coords is None or away_coords is None:
            return None

        return self.haversine_distance(
            home_coords[0], home_coords[1],
            away_coords[0], away_coords[1],
        )

    def get_travel_fatigue_factor(
        self, home_team: str, away_team: str
    ) -> float:
        """Return a fatigue multiplier for the away team's travel burden.

        The multiplier is applied to match outcome probability adjustments.
        A value of 0.0 means no travel fatigue; 1.0 means extreme fatigue.

        Distance bands:
            0–100 km    → 0.00  (local derby, same city)
            100–300 km  → 0.10  (short domestic trip)
            300–500 km  → 0.20  (moderate, possible overnight)
            500–1000 km → 0.35  (significant, flight usually needed)
            1000+ km    → 0.50  (major international travel)

        Parameters
        ----------
        home_team : str
            Home team name.
        away_team : str
            Away team name (the team doing the travelling).

        Returns
        -------
        float
            Fatigue factor in [0.0, 0.5]. Returns 0.0 if either team
            is not found (conservative — no penalty applied on uncertainty).
        """
        distance = self.get_travel_distance(home_team, away_team)

        if distance is None:
            logger.warning(
                f"[TravelCalculator] Cannot compute fatigue factor: "
                f"unknown team(s) '{home_team}' / '{away_team}'"
            )
            return 0.0

        for threshold, factor in self._FATIGUE_BANDS:
            if distance >= threshold:
                return factor

        return 0.0

    # ------------------------------------------------------------------ #
    #  European midweek fatigue analysis                                  #
    # ------------------------------------------------------------------ #

    def get_european_travel_fatigue(
        self,
        team: str,
        midweek_opponent: str,
        weekend_opponent: str,
        turnaround_hours: float = 72.0,
    ) -> dict:
        """Assess travel fatigue burden for a team playing a midweek European
        fixture before a domestic weekend match.

        The academic literature (Lago-Peñas 2009, Dupont 2010) documents
        that teams with a Thursday Europa League away fixture have:
          - ~0.3 fewer goals scored in the following domestic match
          - ~0.2 more goals conceded
          - ~15% lower win probability

        This method quantifies the travel-related component of that burden.

        Parameters
        ----------
        team : str
            The team in question.
        midweek_opponent : str
            Their European opponent (used for travel distance calculation).
        weekend_opponent : str
            Their domestic opponent that weekend.
        turnaround_hours : float
            Hours between midweek and weekend kickoff (default: 72h ≈ Thu→Sun).

        Returns
        -------
        dict with keys:
            team, midweek_opponent, weekend_opponent,
            midweek_distance_km, return_distance_km, total_travel_km,
            turnaround_hours, fatigue_score (0–1),
            expected_goal_deficit, expected_concede_increase,
            win_probability_adjustment, recommendation
        """
        midweek_dist = self.get_travel_distance(team, midweek_opponent)
        weekend_dist = self.get_travel_distance(team, weekend_opponent)

        # Total travel: away trip + return journey
        total_travel_km: Optional[float] = None
        if midweek_dist is not None:
            total_travel_km = round(midweek_dist * 2, 1)  # out + back

        # Base fatigue from travel distance
        travel_fatigue = self.get_travel_fatigue_factor(team, midweek_opponent)

        # Time recovery factor — less rest = more fatigue
        if turnaround_hours < 72:
            time_penalty = max(0.0, (72 - turnaround_hours) / 72 * 0.3)
        else:
            time_penalty = 0.0

        fatigue_score = min(1.0, round(travel_fatigue + time_penalty, 3))

        # Research-derived impact estimates (scaled by fatigue score)
        scale = fatigue_score / 0.5  # normalise to [0, 1] relative to max
        expected_goal_deficit        = round(-0.30 * scale, 3)
        expected_concede_increase    = round(0.20 * scale, 3)
        win_probability_adjustment   = round(-0.15 * scale, 3)

        # Recommendation tier
        if fatigue_score >= 0.4:
            recommendation = "HIGH RISK — significant rotation recommended"
        elif fatigue_score >= 0.25:
            recommendation = "MODERATE RISK — consider rotating 2-3 key players"
        elif fatigue_score >= 0.1:
            recommendation = "LOW RISK — monitor key minutes but rotation optional"
        else:
            recommendation = "MINIMAL RISK — no rotation required"

        return {
            "team":                       team,
            "midweek_opponent":           midweek_opponent,
            "weekend_opponent":           weekend_opponent,
            "midweek_distance_km":        midweek_dist,
            "return_distance_km":         midweek_dist,   # same route back
            "total_travel_km":            total_travel_km,
            "weekend_match_distance_km":  weekend_dist,
            "turnaround_hours":           turnaround_hours,
            "travel_fatigue_base":        travel_fatigue,
            "time_penalty":               round(time_penalty, 3),
            "fatigue_score":              fatigue_score,
            "expected_goal_deficit":      expected_goal_deficit,
            "expected_concede_increase":  expected_concede_increase,
            "win_probability_adjustment": win_probability_adjustment,
            "recommendation":             recommendation,
        }

    # ------------------------------------------------------------------ #
    #  Utility: list available teams                                      #
    # ------------------------------------------------------------------ #

    def list_teams(self) -> list[str]:
        """Return all team names with known venue coordinates."""
        return sorted(self.VENUE_COORDINATES.keys())

    def get_all_distances(self, teams: list[str]) -> dict[tuple[str, str], float]:
        """Compute a distance matrix for a list of teams.

        Parameters
        ----------
        teams : list[str]
            List of team names.

        Returns
        -------
        dict
            Keys are (team_a, team_b) tuples; values are distances in km.
        """
        results: dict[tuple[str, str], float] = {}
        for i, t1 in enumerate(teams):
            for t2 in teams[i + 1:]:
                dist = self.get_travel_distance(t1, t2)
                if dist is not None:
                    results[(t1, t2)] = dist
        return results
