"""Tests for team name normalisation engine."""
import pytest
from sharpedge.normalisation.team_names import normalise, get_canonical_name, get_alias


# Reset the module-level cache before each test module run
@pytest.fixture(autouse=True)
def _reset_registry():
    """Ensure registry is loaded fresh."""
    import sharpedge.normalisation.team_names as mod
    mod._registry = None
    mod._source_lookup = {}
    yield


def test_exact_match():
    """normalise should find exact match for a known source alias."""
    assert normalise("Arsenal", "fbref") == "arsenal"


def test_case_insensitive():
    """normalise should be case-insensitive."""
    assert normalise("arsenal", "fbref") == "arsenal"
    assert normalise("ARSENAL", "fbref") == "arsenal"


def test_cross_source_fallback():
    """When source is unknown, normalise should still find the team across other sources."""
    assert normalise("Arsenal", "unknown_source") == "arsenal"


def test_fuzzy_match():
    """normalise should handle typos via fuzzy matching."""
    result = normalise("Arsneal FC", "unknown_source")
    assert result == "arsenal"


def test_get_canonical_name():
    """get_canonical_name should return the display name for a team ID."""
    assert get_canonical_name("arsenal") == "Arsenal"


def test_get_alias():
    """get_alias should return the source-specific alias."""
    alias = get_alias("arsenal", "fbref")
    assert alias is not None
    assert alias == "Arsenal"


def test_man_united_variants():
    """Manchester United should resolve from various source-specific aliases."""
    assert normalise("Man United", "football_data_uk") == "manchester_united"
    assert normalise("Manchester Utd", "fbref") == "manchester_united"
    assert normalise("Manchester United", "understat") == "manchester_united"
    assert normalise("Man. United", "forebet") == "manchester_united"


def test_psg_variants():
    """PSG should resolve from various aliases."""
    assert normalise("Paris Saint-Germain", "understat") == "psg"
    assert normalise("Paris SG", "football_data_uk") == "psg"
    assert normalise("PSG", "clubelo") == "psg"
    assert normalise("Paris Saint Germain", "understat") == "psg"


def test_inter_milan_variants():
    """Inter Milan variants should all resolve to the same ID."""
    id_fbref = normalise("Inter Milan", "fbref")
    id_understat = normalise("Internazionale", "understat")
    assert id_fbref is not None
    assert id_understat is not None
    assert id_fbref == id_understat == "inter"


def test_wolves_variants():
    """Wolverhampton Wanderers variants should resolve correctly."""
    assert normalise("Wolves", "football_data_uk") == "wolves"
    assert normalise("Wolverhampton Wanderers", "fbref") == "wolves"
    assert normalise("Wolverhampton", "forebet") == "wolves"


def test_nottingham_forest_variants():
    """Nottingham Forest variants should resolve correctly."""
    assert normalise("Nott'm Forest", "football_data_uk") == "nottingham_forest"
    assert normalise("Nottingham Forest", "fbref") == "nottingham_forest"
    assert normalise("Nott. Forest", "forebet") == "nottingham_forest"


def test_bundesliga_variants():
    """German team variants should resolve correctly."""
    assert normalise("Bayern Munich", "clubelo") == "bayern_munich"
    assert normalise("Dortmund", "football_data_uk") == "borussia_dortmund"
    assert normalise("B. Leverkusen", "forebet") == "bayer_leverkusen"
    assert normalise("E. Frankfurt", "forebet") == "eintracht_frankfurt"
    assert normalise("RB Leipzig", "fbref") == "rb_leipzig"


def test_la_liga_variants():
    """Spanish team variants should resolve correctly."""
    assert normalise("Atl. Madrid", "forebet") == "atletico_madrid"
    assert normalise("Athletic Club", "fbref") == "athletic_bilbao"
    assert normalise("Betis", "football_data_uk") == "real_betis"
    assert normalise("Celta de Vigo", "transfermarkt") == "celta_vigo"


def test_serie_a_variants():
    """Italian team variants should resolve correctly."""
    assert normalise("AC Milan", "fbref") == "ac_milan"
    assert normalise("Juventus", "fbref") == "juventus"
    assert normalise("SSC Napoli", "transfermarkt") == "napoli"
    assert normalise("AS Roma", "transfermarkt") == "roma"
    assert normalise("Verona", "football_data_uk") == "hellas_verona"


def test_unknown_team_returns_none():
    """normalise should return None for a completely unknown team."""
    result = normalise("Completely Fake FC 12345", "fbref")
    assert result is None


def test_get_canonical_name_unknown():
    """get_canonical_name should return None for unknown team ID."""
    assert get_canonical_name("nonexistent_team") is None


def test_get_alias_unknown_source():
    """get_alias should return None for unknown source."""
    assert get_alias("arsenal", "nonexistent_source") is None
