# SharpEdge AI — Phase 1: Data Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the complete data foundation — PostgreSQL schema, team name registry, bulletproof scraping engine with BaseCollector, 10 MVP collectors, 4-layer validation engine, and 5-season historical backfill.

**Architecture:** Python 3.11+ monorepo with SQLAlchemy 2.0 ORM, Alembic migrations, BaseCollector pattern for all scrapers, staging→validation→production data pipeline. Every collector inherits retry logic, caching, rate limiting, health reporting, and team name normalisation automatically.

**Tech Stack:** Python 3.11+, PostgreSQL (Supabase), SQLAlchemy 2.0, Alembic, soccerdata, BeautifulSoup4, requests, thefuzz, APScheduler, pytest

---

## Project Structure

```
sharpedge/
├── pyproject.toml
├── .env.example
├── .env
├── alembic.ini
├── alembic/
│   ├── env.py
│   └── versions/
├── src/
│   └── sharpedge/
│       ├── __init__.py
│       ├── config.py
│       ├── db/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   └── models.py
│       ├── collectors/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── football_data_uk.py
│       │   ├── club_elo.py
│       │   ├── understat.py
│       │   ├── fbref.py
│       │   ├── forebet.py
│       │   ├── predictz.py
│       │   ├── windrawwin.py
│       │   ├── open_meteo.py
│       │   ├── football_data_org.py
│       │   └── footystats.py
│       ├── normalisation/
│       │   ├── __init__.py
│       │   └── team_names.py
│       ├── validation/
│       │   ├── __init__.py
│       │   ├── schema.py
│       │   ├── statistical.py
│       │   ├── cross_source.py
│       │   └── freshness.py
│       ├── alerts/
│       │   ├── __init__.py
│       │   └── telegram.py
│       └── orchestrator.py
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_db/
│   │   └── test_models.py
│   ├── test_collectors/
│   │   ├── test_base.py
│   │   ├── test_football_data_uk.py
│   │   ├── test_club_elo.py
│   │   ├── test_understat.py
│   │   ├── test_fbref.py
│   │   ├── test_forebet.py
│   │   └── test_open_meteo.py
│   ├── test_normalisation/
│   │   └── test_team_names.py
│   └── test_validation/
│       ├── test_schema.py
│       └── test_statistical.py
├── data/
│   └── team_registry.json
└── scripts/
    ├── backfill.py
    └── health_check.py
```

---

## Task 1: Project Scaffolding + Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/sharpedge/__init__.py`
- Create: `src/sharpedge/config.py`
- Create: `tests/conftest.py`

**Step 1: Initialise git repo**

```bash
cd "/Users/balabollineni/Sharp Edge"
git init
```

**Step 2: Create pyproject.toml**

```toml
[project]
name = "sharpedge"
version = "0.1.0"
description = "AI-powered football prediction engine"
requires-python = ">=3.11"
dependencies = [
    "sqlalchemy[asyncio]>=2.0",
    "alembic>=1.13",
    "psycopg2-binary>=2.9",
    "pandas>=2.1",
    "requests>=2.31",
    "beautifulsoup4>=4.12",
    "lxml>=5.1",
    "soccerdata>=1.7",
    "thefuzz[speedup]>=0.22",
    "python-dotenv>=1.0",
    "pydantic>=2.5",
    "pydantic-settings>=2.1",
    "apscheduler>=3.10",
    "tenacity>=8.2",
    "httpx>=0.27",
    "python-telegram-bot>=21.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.1",
    "pytest-mock>=3.12",
    "ruff>=0.3",
    "mypy>=1.8",
]
ml = [
    "xgboost>=2.0",
    "scikit-learn>=1.4",
    "scipy>=1.12",
    "optuna>=3.5",
    "shap>=0.44",
    "mlflow>=2.10",
]

[build-system]
requires = ["setuptools>=69.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.ruff]
target-version = "py311"
line-length = 100
```

**Step 3: Create .env.example**

```env
DATABASE_URL=postgresql://user:password@host:5432/sharpedge
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_ALERT_CHAT_ID=your_chat_id_here
ENVIRONMENT=development
```

**Step 4: Create .gitignore**

```gitignore
__pycache__/
*.pyc
.env
*.egg-info/
dist/
build/
.venv/
venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
htmlcov/
*.db
data/cache/
mlruns/
```

**Step 5: Create config.py**

```python
# src/sharpedge/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://localhost:5432/sharpedge"
    telegram_bot_token: str = ""
    telegram_alert_chat_id: str = ""
    environment: str = "development"

    # Scraping settings
    default_request_delay: float = 3.0
    max_retries: int = 5
    request_timeout: int = 30

    # Data directories
    cache_dir: str = "data/cache"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
```

**Step 6: Create __init__.py and conftest.py**

```python
# src/sharpedge/__init__.py
"""SharpEdge AI — Football prediction engine."""
```

```python
# tests/conftest.py
import pytest
from sharpedge.config import Settings


@pytest.fixture
def test_settings():
    return Settings(
        database_url="sqlite:///test.db",
        environment="test",
    )
```

**Step 7: Create test for config, run it**

```python
# tests/test_config.py
from sharpedge.config import Settings


def test_settings_defaults():
    s = Settings(database_url="sqlite:///test.db")
    assert s.environment == "development"
    assert s.default_request_delay == 3.0
    assert s.max_retries == 5
```

Run: `cd "/Users/balabollineni/Sharp Edge" && pip install -e ".[dev]" && pytest tests/test_config.py -v`
Expected: PASS

**Step 8: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with dependencies and config"
```

---

## Task 2: Database Models + Alembic Migrations

**Files:**
- Create: `src/sharpedge/db/__init__.py`
- Create: `src/sharpedge/db/engine.py`
- Create: `src/sharpedge/db/models.py`
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `tests/test_db/test_models.py`

**Step 1: Create database engine**

```python
# src/sharpedge/db/__init__.py
```

```python
# src/sharpedge/db/engine.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sharpedge.config import settings

engine = create_engine(settings.database_url, echo=False)
SessionLocal = sessionmaker(bind=engine)


def get_session() -> Session:
    return SessionLocal()
```

**Step 2: Create all database models**

This is the core schema. Every table maps to the design doc.

```python
# src/sharpedge/db/models.py
from datetime import datetime, date
from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime, Boolean,
    ForeignKey, UniqueConstraint, Index, Text, JSON,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ── Master Registries ──────────────────────────────────────

class Team(Base):
    """Master team registry with per-source aliases."""
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    canonical_name = Column(String(100), unique=True, nullable=False)
    country = Column(String(50))
    # Per-source name aliases stored as JSON: {"fbref": "Man Utd", "understat": "Manchester United"}
    aliases = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)


class League(Base):
    """League/competition registry."""
    __tablename__ = "leagues"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    country = Column(String(50), nullable=False)
    # Source-specific league identifiers
    fd_uk_code = Column(String(10))       # Football-Data.co.uk code (E0, SP1, etc.)
    fbref_id = Column(String(50))
    understat_name = Column(String(100))
    season_format = Column(String(20), default="autumn-spring")  # or "spring-autumn"

    __table_args__ = (UniqueConstraint("name", "country"),)


class Season(Base):
    """Season registry (e.g. 2024-25)."""
    __tablename__ = "seasons"

    id = Column(Integer, primary_key=True)
    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    label = Column(String(10), nullable=False)  # "2024-25"
    start_date = Column(Date)
    end_date = Column(Date)

    league = relationship("League")
    __table_args__ = (UniqueConstraint("league_id", "label"),)


# ── Match Data ──────────────────────────────────────────────

class Match(Base):
    """Core match record — single source of truth."""
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    season_id = Column(Integer, ForeignKey("seasons.id"), nullable=False)
    match_date = Column(Date, nullable=False)
    kick_off_time = Column(String(5))  # "15:00"
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    home_goals_ht = Column(Integer)
    away_goals_ht = Column(Integer)
    result = Column(String(1))  # H, D, A
    status = Column(String(20), default="scheduled")  # scheduled, completed, postponed
    referee = Column(String(100))
    venue = Column(String(200))

    season = relationship("Season")
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])

    __table_args__ = (
        UniqueConstraint("season_id", "match_date", "home_team_id", "away_team_id"),
        Index("ix_matches_date", "match_date"),
        Index("ix_matches_status", "status"),
    )


class MatchStats(Base):
    """Detailed match statistics from FBref/SofaScore."""
    __tablename__ = "match_stats"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False, unique=True)
    source = Column(String(30), nullable=False)  # fbref, sofascore, etc.
    # Home stats
    home_shots = Column(Integer)
    home_shots_on_target = Column(Integer)
    home_possession = Column(Float)
    home_passes = Column(Integer)
    home_pass_accuracy = Column(Float)
    home_fouls = Column(Integer)
    home_corners = Column(Integer)
    home_yellow_cards = Column(Integer)
    home_red_cards = Column(Integer)
    # Away stats
    away_shots = Column(Integer)
    away_shots_on_target = Column(Integer)
    away_possession = Column(Float)
    away_passes = Column(Integer)
    away_pass_accuracy = Column(Float)
    away_fouls = Column(Integer)
    away_corners = Column(Integer)
    away_yellow_cards = Column(Integer)
    away_red_cards = Column(Integer)
    # Raw JSON for any extra stats
    extra_stats = Column(JSON)

    match = relationship("Match")


class MatchXG(Base):
    """Expected goals data from Understat/FBref."""
    __tablename__ = "match_xg"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    source = Column(String(30), nullable=False)  # understat, fbref
    home_xg = Column(Float)
    away_xg = Column(Float)
    home_npxg = Column(Float)  # non-penalty xG
    away_npxg = Column(Float)

    match = relationship("Match")
    __table_args__ = (UniqueConstraint("match_id", "source"),)


# ── Odds Data ───────────────────────────────────────────────

class MatchOdds(Base):
    """Bookmaker odds for a match."""
    __tablename__ = "match_odds"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    bookmaker = Column(String(50), nullable=False)
    market = Column(String(30), nullable=False)  # 1x2, over_under_25, btts, etc.
    odds_type = Column(String(10), default="closing")  # opening, closing
    # 1X2 odds
    odds_home = Column(Float)
    odds_draw = Column(Float)
    odds_away = Column(Float)
    # Over/Under
    odds_over = Column(Float)
    odds_under = Column(Float)
    line = Column(Float)  # 2.5, 1.5, etc.
    # Timestamps
    captured_at = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match")
    __table_args__ = (
        UniqueConstraint("match_id", "bookmaker", "market", "odds_type"),
        Index("ix_odds_match", "match_id"),
    )


# ── ELO Ratings ─────────────────────────────────────────────

class EloRating(Base):
    """Daily ELO ratings from ClubELO."""
    __tablename__ = "elo_ratings"

    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    rating_date = Column(Date, nullable=False)
    elo = Column(Float, nullable=False)
    rank = Column(Integer)
    source = Column(String(30), default="clubelo")

    team = relationship("Team")
    __table_args__ = (
        UniqueConstraint("team_id", "rating_date", "source"),
        Index("ix_elo_date", "rating_date"),
    )


# ── Competitor Predictions ──────────────────────────────────

class CompetitorPrediction(Base):
    """Predictions scraped from Forebet, PredictZ, WinDrawWin, etc."""
    __tablename__ = "competitor_predictions"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    source = Column(String(30), nullable=False)  # forebet, predictz, windrawwin, betclan
    predicted_result = Column(String(5))  # H, D, A, or "1X", "X2", etc.
    prob_home = Column(Float)
    prob_draw = Column(Float)
    prob_away = Column(Float)
    predicted_score_home = Column(Integer)
    predicted_score_away = Column(Integer)
    btts_prediction = Column(Boolean)
    over_under_prediction = Column(String(10))  # "over", "under"
    over_under_line = Column(Float)  # 2.5, 1.5
    confidence = Column(Float)
    scraped_at = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match")
    __table_args__ = (UniqueConstraint("match_id", "source"),)


# ── Weather ─────────────────────────────────────────────────

class MatchWeather(Base):
    """Weather conditions at match time from Open-Meteo."""
    __tablename__ = "match_weather"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False, unique=True)
    temperature_c = Column(Float)
    precipitation_mm = Column(Float)
    wind_speed_kmh = Column(Float)
    humidity_pct = Column(Float)
    weather_code = Column(Integer)  # WMO weather code

    match = relationship("Match")


# ── Source Health Tracking ──────────────────────────────────

class SourceHealth(Base):
    """Tracks health status of each data source."""
    __tablename__ = "source_health"

    id = Column(Integer, primary_key=True)
    source_name = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False)  # healthy, degraded, down
    last_success = Column(DateTime)
    last_failure = Column(DateTime)
    consecutive_failures = Column(Integer, default=0)
    rows_collected = Column(Integer, default=0)
    avg_collection_time_ms = Column(Integer)
    error_message = Column(Text)
    checked_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_health_source", "source_name"),)


# ── Raw Staging Tables ──────────────────────────────────────

class RawStagingRecord(Base):
    """Staging area for raw scraped data before validation."""
    __tablename__ = "raw_staging"

    id = Column(Integer, primary_key=True)
    source = Column(String(50), nullable=False)
    record_type = Column(String(30), nullable=False)  # match, odds, xg, prediction, etc.
    raw_data = Column(JSON, nullable=False)
    status = Column(String(20), default="pending")  # pending, validated, rejected, promoted
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    validated_at = Column(DateTime)

    __table_args__ = (Index("ix_staging_status", "status"),)
```

**Step 3: Write model tests**

```python
# tests/test_db/test_models.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sharpedge.db.models import Base, Team, League, Season, Match, MatchOdds, EloRating


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_create_team(db_session):
    team = Team(canonical_name="Manchester United", country="England", aliases={"fbref": "Man Utd"})
    db_session.add(team)
    db_session.commit()
    assert team.id is not None
    assert team.aliases["fbref"] == "Man Utd"


def test_create_match_with_teams(db_session):
    home = Team(canonical_name="Arsenal", country="England")
    away = Team(canonical_name="Chelsea", country="England")
    league = League(name="Premier League", country="England", fd_uk_code="E0")
    db_session.add_all([home, away, league])
    db_session.flush()
    season = Season(league_id=league.id, label="2024-25")
    db_session.add(season)
    db_session.flush()
    from datetime import date
    match = Match(
        season_id=season.id, match_date=date(2025, 1, 15),
        home_team_id=home.id, away_team_id=away.id,
        home_goals=2, away_goals=1, result="H", status="completed",
    )
    db_session.add(match)
    db_session.commit()
    assert match.id is not None
    assert match.result == "H"


def test_match_odds(db_session):
    team1 = Team(canonical_name="Team A", country="X")
    team2 = Team(canonical_name="Team B", country="X")
    league = League(name="Test League", country="X")
    db_session.add_all([team1, team2, league])
    db_session.flush()
    season = Season(league_id=league.id, label="2024-25")
    db_session.add(season)
    db_session.flush()
    from datetime import date
    match = Match(
        season_id=season.id, match_date=date(2025, 1, 1),
        home_team_id=team1.id, away_team_id=team2.id,
    )
    db_session.add(match)
    db_session.flush()
    odds = MatchOdds(
        match_id=match.id, bookmaker="Bet365", market="1x2",
        odds_home=1.5, odds_draw=4.0, odds_away=6.5,
    )
    db_session.add(odds)
    db_session.commit()
    assert odds.odds_home == 1.5
```

Run: `pytest tests/test_db/test_models.py -v`
Expected: 3 tests PASS

**Step 4: Set up Alembic**

```bash
cd "/Users/balabollineni/Sharp Edge"
alembic init alembic
```

Then update `alembic/env.py` to import our models and use our database URL.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: database models — teams, matches, odds, xG, ELO, predictions, staging"
```

---

## Task 3: Team Name Normalisation Engine

**Files:**
- Create: `src/sharpedge/normalisation/__init__.py`
- Create: `src/sharpedge/normalisation/team_names.py`
- Create: `data/team_registry.json`
- Create: `tests/test_normalisation/test_team_names.py`

**Step 1: Create the team registry JSON (Big 5 leagues — all teams)**

The registry maps a canonical ID to all known aliases per source. This is the single most important data file in the project — without it, data fusion fails. Start with 2024-25 season teams. File will be ~500 entries.

Structure per team:
```json
{
  "arsenal": {
    "canonical": "Arsenal",
    "country": "England",
    "league": "Premier League",
    "aliases": {
      "football_data_uk": "Arsenal",
      "fbref": "Arsenal",
      "understat": "Arsenal",
      "forebet": "Arsenal",
      "predictz": "Arsenal",
      "windrawwin": "Arsenal",
      "oddsportal": "Arsenal",
      "transfermarkt": "Arsenal FC",
      "clubelo": "Arsenal",
      "sofascore": "Arsenal"
    }
  }
}
```

Generate the full registry covering all ~100 teams across Big 5 leagues.

**Step 2: Create the normalisation module**

```python
# src/sharpedge/normalisation/team_names.py
import json
from pathlib import Path
from thefuzz import fuzz, process
from typing import Optional

_REGISTRY_PATH = Path(__file__).parent.parent.parent.parent / "data" / "team_registry.json"
_registry: dict | None = None
_source_lookup: dict[str, dict[str, str]] = {}


def _load_registry() -> dict:
    global _registry, _source_lookup
    if _registry is None:
        with open(_REGISTRY_PATH) as f:
            _registry = json.load(f)
        # Build reverse lookup: source → {alias: canonical_id}
        for team_id, info in _registry.items():
            for source, alias in info.get("aliases", {}).items():
                if source not in _source_lookup:
                    _source_lookup[source] = {}
                _source_lookup[source][alias.lower()] = team_id
    return _registry


def normalise(name: str, source: str) -> Optional[str]:
    """Resolve a team name from a specific source to canonical team ID.

    Returns the canonical team ID (e.g. "arsenal") or None if not found.
    Uses exact match first, then fuzzy matching as fallback.
    """
    _load_registry()

    # Exact match (case-insensitive)
    lookup = _source_lookup.get(source, {})
    if name.lower() in lookup:
        return lookup[name.lower()]

    # Try across all sources (team might use same name)
    for src, names in _source_lookup.items():
        if name.lower() in names:
            return names[name.lower()]

    # Fuzzy match as last resort
    all_names = {}
    for team_id, info in _registry.items():
        all_names[info["canonical"]] = team_id
        for alias in info.get("aliases", {}).values():
            all_names[alias] = team_id

    match, score = process.extractOne(name, all_names.keys(), scorer=fuzz.token_sort_ratio)
    if score >= 85:
        return all_names[match]

    return None


def get_canonical_name(team_id: str) -> Optional[str]:
    """Get the display name for a canonical team ID."""
    _load_registry()
    if team_id in _registry:
        return _registry[team_id]["canonical"]
    return None


def get_alias(team_id: str, source: str) -> Optional[str]:
    """Get the source-specific name for a team."""
    _load_registry()
    if team_id in _registry:
        return _registry[team_id].get("aliases", {}).get(source)
    return None
```

**Step 3: Write tests**

```python
# tests/test_normalisation/test_team_names.py
from sharpedge.normalisation.team_names import normalise, get_canonical_name, get_alias


def test_exact_match():
    result = normalise("Arsenal", "fbref")
    assert result == "arsenal"


def test_case_insensitive():
    result = normalise("arsenal", "fbref")
    assert result == "arsenal"


def test_cross_source_fallback():
    # Even if the source is wrong, should find via other sources
    result = normalise("Arsenal", "unknown_source")
    assert result == "arsenal"


def test_fuzzy_match():
    result = normalise("Arsneal FC", "unknown_source")  # typo
    assert result == "arsenal"


def test_get_canonical_name():
    assert get_canonical_name("arsenal") == "Arsenal"


def test_get_alias():
    alias = get_alias("arsenal", "fbref")
    assert alias is not None
```

Run: `pytest tests/test_normalisation/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: team name normalisation with fuzzy matching + Big 5 registry"
```

---

## Task 4: BaseCollector — The Bulletproof Scraper Foundation

**Files:**
- Create: `src/sharpedge/collectors/__init__.py`
- Create: `src/sharpedge/collectors/base.py`
- Create: `src/sharpedge/alerts/__init__.py`
- Create: `src/sharpedge/alerts/telegram.py`
- Create: `tests/test_collectors/test_base.py`

**Step 1: Create Telegram alerting module**

```python
# src/sharpedge/alerts/telegram.py
import logging
from sharpedge.config import settings

logger = logging.getLogger(__name__)


async def send_alert(message: str) -> None:
    """Send alert to Telegram. Falls back to logging if not configured."""
    if not settings.telegram_bot_token or not settings.telegram_alert_chat_id:
        logger.warning(f"ALERT (Telegram not configured): {message}")
        return
    try:
        import httpx
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        async with httpx.AsyncClient() as client:
            await client.post(url, json={
                "chat_id": settings.telegram_alert_chat_id,
                "text": f"🚨 SharpEdge Alert\n\n{message}",
                "parse_mode": "HTML",
            })
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")


def send_alert_sync(message: str) -> None:
    """Synchronous version for use in collectors."""
    if not settings.telegram_bot_token or not settings.telegram_alert_chat_id:
        logger.warning(f"ALERT: {message}")
        return
    try:
        import httpx
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        with httpx.Client() as client:
            client.post(url, json={
                "chat_id": settings.telegram_alert_chat_id,
                "text": f"🚨 SharpEdge Alert\n\n{message}",
            })
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")
```

**Step 2: Create BaseCollector**

```python
# src/sharpedge/collectors/base.py
import logging
import random
import time
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from sharpedge.config import settings
from sharpedge.alerts.telegram import send_alert_sync
from sharpedge.normalisation.team_names import normalise

logger = logging.getLogger(__name__)

# Rotating User-Agent pool
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
]


class CollectorError(Exception):
    """Base exception for collector errors."""
    pass


class StructureChangedError(CollectorError):
    """Raised when the source HTML structure has changed unexpectedly."""
    pass


class BaseCollector(ABC):
    """Base class for all data collectors.

    Provides: rate limiting, retry logic, caching, health reporting,
    team name normalisation, structural fingerprinting, alerting, metrics.

    Every collector inherits this. Zero boilerplate per new scraper.
    """

    # Subclasses must set these
    source_name: str = ""
    base_url: str = ""
    request_delay: float = 3.0  # seconds between requests

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
        })
        self._last_request_time = 0.0
        self._request_count = 0
        self._error_count = 0
        self._start_time = None
        self._cache_dir = Path(settings.cache_dir) / self.source_name
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Rate Limiting ───────────────────────────────────────

    def _rate_limit(self) -> None:
        """Enforce minimum delay between requests with jitter."""
        elapsed = time.time() - self._last_request_time
        delay = self.request_delay + random.uniform(0, 1.5)  # jitter
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_time = time.time()

    # ── HTTP with retry ─────────────────────────────────────

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=300),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry {retry_state.attempt_number}/5 for {retry_state.args[0] if retry_state.args else 'unknown'}"
        ),
    )
    def _fetch(self, url: str, **kwargs) -> requests.Response:
        """Fetch URL with rate limiting, retry, and rotating UA."""
        self._rate_limit()
        self._session.headers["User-Agent"] = random.choice(USER_AGENTS)
        response = self._session.get(url, timeout=settings.request_timeout, **kwargs)
        response.raise_for_status()
        self._request_count += 1
        return response

    def _fetch_json(self, url: str, **kwargs) -> Any:
        """Fetch and parse JSON response."""
        resp = self._fetch(url, **kwargs)
        return resp.json()

    # ── Caching ─────────────────────────────────────────────

    def _cache_key(self, identifier: str) -> Path:
        """Generate cache file path for an identifier."""
        safe_name = hashlib.md5(identifier.encode()).hexdigest()
        return self._cache_dir / f"{safe_name}.json"

    def _get_cached(self, identifier: str) -> Any | None:
        """Return cached data if exists and is not stale."""
        cache_file = self._cache_key(identifier)
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            return data.get("payload")
        return None

    def _set_cache(self, identifier: str, payload: Any) -> None:
        """Cache data to disk."""
        cache_file = self._cache_key(identifier)
        cache_file.write_text(json.dumps({
            "cached_at": datetime.utcnow().isoformat(),
            "identifier": identifier,
            "payload": payload,
        }, default=str))

    # ── Team Name Normalisation ─────────────────────────────

    def normalise_team(self, name: str) -> str | None:
        """Resolve team name to canonical ID using this collector's source."""
        return normalise(name, self.source_name)

    # ── Structural Fingerprinting ───────────────────────────

    def _check_structure(self, html: str, expected_markers: list[str]) -> bool:
        """Verify the page still has expected structural markers.

        Each collector defines markers (CSS selectors, text patterns)
        that should be present if the page structure hasn't changed.
        """
        for marker in expected_markers:
            if marker not in html:
                logger.warning(f"[{self.source_name}] Structure marker missing: {marker}")
                return False
        return True

    # ── Health Reporting ────────────────────────────────────

    def _report_health(self, status: str, rows: int = 0, error: str = "") -> dict:
        """Generate health report for this collection run."""
        duration = (time.time() - self._start_time) * 1000 if self._start_time else 0
        report = {
            "source_name": self.source_name,
            "status": status,
            "rows_collected": rows,
            "requests_made": self._request_count,
            "errors": self._error_count,
            "duration_ms": int(duration),
            "error_message": error,
            "checked_at": datetime.utcnow().isoformat(),
        }
        if status == "down":
            send_alert_sync(
                f"Source DOWN: {self.source_name}\n"
                f"Error: {error}\n"
                f"Consecutive failures: {self._error_count}"
            )
        return report

    # ── Main Collection Interface ───────────────────────────

    def collect(self, **kwargs) -> pd.DataFrame:
        """Run the collector with full error handling and health reporting.

        This is the public API. Subclasses implement _collect().
        """
        self._start_time = time.time()
        self._request_count = 0
        self._error_count = 0

        try:
            logger.info(f"[{self.source_name}] Starting collection...")
            df = self._collect(**kwargs)
            rows = len(df) if df is not None else 0
            logger.info(f"[{self.source_name}] Collected {rows} rows")
            self._report_health("healthy", rows=rows)
            return df
        except StructureChangedError as e:
            self._error_count += 1
            logger.error(f"[{self.source_name}] Structure changed: {e}")
            self._report_health("degraded", error=str(e))
            return pd.DataFrame()
        except Exception as e:
            self._error_count += 1
            logger.error(f"[{self.source_name}] Collection failed: {e}")
            self._report_health("down", error=str(e))
            return pd.DataFrame()

    @abstractmethod
    def _collect(self, **kwargs) -> pd.DataFrame:
        """Implement the actual collection logic. Subclasses override this."""
        ...
```

**Step 3: Write BaseCollector tests**

```python
# tests/test_collectors/test_base.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from sharpedge.collectors.base import BaseCollector, CollectorError


class MockCollector(BaseCollector):
    source_name = "test_source"
    base_url = "https://example.com"
    request_delay = 0.0  # no delay in tests

    def _collect(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({"team": ["Arsenal", "Chelsea"], "goals": [2, 1]})


class FailingCollector(BaseCollector):
    source_name = "failing_source"
    base_url = "https://example.com"
    request_delay = 0.0

    def _collect(self, **kwargs) -> pd.DataFrame:
        raise ValueError("Simulated failure")


def test_collect_returns_dataframe():
    collector = MockCollector()
    df = collector.collect()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_collect_handles_failure_gracefully():
    collector = FailingCollector()
    df = collector.collect()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0  # empty df on failure, not an exception


def test_cache_roundtrip():
    collector = MockCollector()
    collector._set_cache("test_key", {"data": [1, 2, 3]})
    result = collector._get_cached("test_key")
    assert result == {"data": [1, 2, 3]}


def test_cache_miss():
    collector = MockCollector()
    result = collector._get_cached("nonexistent_key")
    assert result is None


def test_structure_check_passes():
    collector = MockCollector()
    html = '<div class="table-stats">Some content</div>'
    assert collector._check_structure(html, ["table-stats"]) is True


def test_structure_check_fails():
    collector = MockCollector()
    html = '<div class="new-layout">Different content</div>'
    assert collector._check_structure(html, ["table-stats"]) is False


def test_normalise_team():
    collector = MockCollector()
    result = collector.normalise_team("Arsenal")
    # This will hit the real registry — should resolve
    assert result == "arsenal"
```

Run: `pytest tests/test_collectors/test_base.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: BaseCollector with retry, caching, rate limiting, health reporting"
```

---

## Task 5: Football-Data.co.uk Collector (CSV Downloads)

**Files:**
- Create: `src/sharpedge/collectors/football_data_uk.py`
- Create: `tests/test_collectors/test_football_data_uk.py`

**Step 1: Implement the collector**

This is the most important source — direct CSV downloads, no scraping needed. Provides match results + odds from multiple bookmakers going back to the 1990s.

```python
# src/sharpedge/collectors/football_data_uk.py
import io
import pandas as pd
from sharpedge.collectors.base import BaseCollector

# Football-Data.co.uk league codes
LEAGUE_CODES = {
    "Premier League": "E0",
    "La Liga": "SP1",
    "Bundesliga": "D1",
    "Serie A": "I1",
    "Ligue 1": "F1",
}

# Season URL patterns
# Current season: https://www.football-data.co.uk/mmz4281/2425/E0.csv
# Historical: same pattern
SEASON_LABELS = {
    "2024-25": "2425",
    "2023-24": "2324",
    "2022-23": "2223",
    "2021-22": "2122",
    "2020-21": "2021",
}

# Key columns we extract
RESULT_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR", "Referee",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
]

ODDS_COLUMNS = [
    "B365H", "B365D", "B365A",  # Bet365
    "BWH", "BWD", "BWA",        # Betway
    "PSH", "PSD", "PSA",        # Pinnacle
    "WHH", "WHD", "WHA",        # William Hill
    "VCH", "VCD", "VCA",        # VC Bet
    "MaxH", "MaxD", "MaxA",     # Max odds across bookmakers
    "AvgH", "AvgD", "AvgA",    # Average odds
]

OU_ODDS_COLUMNS = [
    "BbAv>2.5", "BbAv<2.5",    # Average Over/Under 2.5
    "P>2.5", "P<2.5",          # Pinnacle O/U (newer seasons)
]


class FootballDataUKCollector(BaseCollector):
    source_name = "football_data_uk"
    base_url = "https://www.football-data.co.uk/mmz4281"
    request_delay = 1.0  # CSVs are lightweight, gentle delay is enough

    def _collect(self, league: str = None, season: str = None, **kwargs) -> pd.DataFrame:
        """Collect match results + odds CSVs.

        Args:
            league: Single league name, or None for all Big 5
            season: Single season label (e.g. "2024-25"), or None for all 5 seasons
        """
        leagues = {league: LEAGUE_CODES[league]} if league else LEAGUE_CODES
        seasons = {season: SEASON_LABELS[season]} if season else SEASON_LABELS

        all_data = []
        for league_name, league_code in leagues.items():
            for season_label, season_code in seasons.items():
                cache_key = f"{league_code}_{season_code}"
                cached = self._get_cached(cache_key)
                if cached is not None:
                    df = pd.DataFrame(cached)
                    all_data.append(df)
                    continue

                url = f"{self.base_url}/{season_code}/{league_code}.csv"
                try:
                    response = self._fetch(url)
                    df = pd.read_csv(io.StringIO(response.text))

                    # Keep only columns that exist in this CSV
                    available_cols = [c for c in RESULT_COLUMNS + ODDS_COLUMNS + OU_ODDS_COLUMNS if c in df.columns]
                    df = df[available_cols].copy()
                    df["league"] = league_name
                    df["season"] = season_label

                    # Normalise team names
                    df["home_team_id"] = df["HomeTeam"].apply(lambda x: self.normalise_team(str(x)))
                    df["away_team_id"] = df["AwayTeam"].apply(lambda x: self.normalise_team(str(x)))

                    self._set_cache(cache_key, df.to_dict(orient="records"))
                    all_data.append(df)
                except Exception as e:
                    self._error_count += 1
                    logger.warning(f"Failed to fetch {league_name} {season_label}: {e}")

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)


import logging
logger = logging.getLogger(__name__)
```

**Step 2: Write tests (using real CSV download for integration test, mock for unit)**

```python
# tests/test_collectors/test_football_data_uk.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from sharpedge.collectors.football_data_uk import FootballDataUKCollector, LEAGUE_CODES, SEASON_LABELS


def test_league_codes_cover_big_5():
    assert "Premier League" in LEAGUE_CODES
    assert "La Liga" in LEAGUE_CODES
    assert "Bundesliga" in LEAGUE_CODES
    assert "Serie A" in LEAGUE_CODES
    assert "Ligue 1" in LEAGUE_CODES


def test_season_labels_cover_5_seasons():
    assert len(SEASON_LABELS) == 5
    assert "2024-25" in SEASON_LABELS
    assert "2020-21" in SEASON_LABELS


@pytest.mark.integration
def test_collect_single_season_single_league():
    """Integration test — actually downloads one CSV."""
    collector = FootballDataUKCollector()
    df = collector.collect(league="Premier League", season="2023-24")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 300  # Full PL season = 380 matches
    assert "HomeTeam" in df.columns
    assert "B365H" in df.columns or "PSH" in df.columns
```

Run: `pytest tests/test_collectors/test_football_data_uk.py -v -m "not integration"` (unit tests only)
Run: `pytest tests/test_collectors/test_football_data_uk.py -v -m integration` (integration test)

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: Football-Data.co.uk collector — CSV match results + bookmaker odds"
```

---

## Task 6: ClubELO Collector

**Files:**
- Create: `src/sharpedge/collectors/club_elo.py`
- Create: `tests/test_collectors/test_club_elo.py`

**Step 1: Implement collector**

```python
# src/sharpedge/collectors/club_elo.py
import io
import logging
from datetime import date, timedelta
import pandas as pd
from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)


class ClubEloCollector(BaseCollector):
    source_name = "clubelo"
    base_url = "http://api.clubelo.com"
    request_delay = 1.0

    def _collect(self, team: str = None, date_str: str = None,
                 date_range: tuple[str, str] = None, **kwargs) -> pd.DataFrame:
        """Collect ELO ratings.

        Args:
            team: Single team name (e.g. "Arsenal") for full history
            date_str: Single date (YYYY-MM-DD) for all teams on that date
            date_range: Tuple (start, end) for weekly snapshots across range
        """
        if team:
            return self._collect_team(team)
        elif date_str:
            return self._collect_date(date_str)
        elif date_range:
            return self._collect_range(*date_range)
        else:
            # Default: today's ratings
            return self._collect_date(date.today().isoformat())

    def _collect_date(self, date_str: str) -> pd.DataFrame:
        """Get all club ratings for a specific date."""
        cache_key = f"date_{date_str}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return pd.DataFrame(cached)

        url = f"{self.base_url}/{date_str}"
        response = self._fetch(url)
        df = pd.read_csv(io.StringIO(response.text), names=[
            "rank", "club", "country", "level", "elo", "from_date", "to_date"
        ])
        df["rating_date"] = date_str
        df["team_id"] = df["club"].apply(lambda x: self.normalise_team(str(x)))

        self._set_cache(cache_key, df.to_dict(orient="records"))
        return df

    def _collect_team(self, team: str) -> pd.DataFrame:
        """Get full ELO history for a single team."""
        cache_key = f"team_{team}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return pd.DataFrame(cached)

        url = f"{self.base_url}/{team}"
        response = self._fetch(url)
        df = pd.read_csv(io.StringIO(response.text), names=[
            "rank", "club", "country", "level", "elo", "from_date", "to_date"
        ])
        df["team_id"] = self.normalise_team(team)

        self._set_cache(cache_key, df.to_dict(orient="records"))
        return df

    def _collect_range(self, start: str, end: str) -> pd.DataFrame:
        """Get weekly snapshots between start and end dates."""
        all_data = []
        current = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
        while current <= end_date:
            df = self._collect_date(current.isoformat())
            all_data.append(df)
            current += timedelta(days=7)

        if not all_data:
            return pd.DataFrame()
        return pd.concat(all_data, ignore_index=True)
```

**Step 2: Write tests**

```python
# tests/test_collectors/test_club_elo.py
import pytest
import pandas as pd
from sharpedge.collectors.club_elo import ClubEloCollector


@pytest.mark.integration
def test_collect_single_date():
    collector = ClubEloCollector()
    df = collector.collect(date_str="2025-01-01")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 400  # Hundreds of clubs rated
    assert "elo" in df.columns
    assert "club" in df.columns


@pytest.mark.integration
def test_collect_single_team():
    collector = ClubEloCollector()
    df = collector.collect(team="Arsenal")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 50  # Many historical entries
```

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: ClubELO collector — daily ELO ratings via CSV API"
```

---

## Task 7: Understat Collector (xG Data)

**Files:**
- Create: `src/sharpedge/collectors/understat.py`
- Create: `tests/test_collectors/test_understat.py`

**Step 1: Implement collector**

Understat embeds JSON data in `<script>` tags. No browser automation needed — just parse the JavaScript variables.

```python
# src/sharpedge/collectors/understat.py
import json
import logging
import re
import pandas as pd
from sharpedge.collectors.base import BaseCollector, StructureChangedError

logger = logging.getLogger(__name__)

UNDERSTAT_LEAGUES = {
    "Premier League": "EPL",
    "La Liga": "La_liga",
    "Bundesliga": "Bundesliga",
    "Serie A": "Serie_A",
    "Ligue 1": "Ligue_1",
}


class UnderstatCollector(BaseCollector):
    source_name = "understat"
    base_url = "https://understat.com"
    request_delay = 3.0

    def _collect(self, league: str = None, season: int = None, **kwargs) -> pd.DataFrame:
        """Collect xG data from Understat.

        Args:
            league: League name, or None for all Big 5
            season: Year the season starts (e.g. 2024 for 2024-25), or None for all
        """
        leagues = {league: UNDERSTAT_LEAGUES[league]} if league else UNDERSTAT_LEAGUES
        seasons = [season] if season else list(range(2020, 2026))

        all_data = []
        for league_name, league_code in leagues.items():
            for yr in seasons:
                cache_key = f"{league_code}_{yr}"
                cached = self._get_cached(cache_key)
                if cached is not None:
                    all_data.append(pd.DataFrame(cached))
                    continue

                try:
                    df = self._scrape_league_season(league_code, yr, league_name)
                    if len(df) > 0:
                        self._set_cache(cache_key, df.to_dict(orient="records"))
                        all_data.append(df)
                except Exception as e:
                    self._error_count += 1
                    logger.warning(f"Failed: {league_name} {yr}: {e}")

        if not all_data:
            return pd.DataFrame()
        return pd.concat(all_data, ignore_index=True)

    def _scrape_league_season(self, league_code: str, season: int, league_name: str) -> pd.DataFrame:
        """Scrape a single league-season page."""
        url = f"{self.base_url}/league/{league_code}/{season}"
        response = self._fetch(url)
        html = response.text

        # Structural check
        if not self._check_structure(html, ["datesData", "teamsData"]):
            raise StructureChangedError(f"Understat page structure changed for {league_code}/{season}")

        # Extract JSON from script tags
        matches = self._extract_json_var(html, "datesData")
        if matches is None:
            raise StructureChangedError("Could not find datesData in page")

        rows = []
        for match in matches:
            rows.append({
                "match_id_understat": match.get("id"),
                "date": match.get("datetime", "")[:10],
                "home_team": match.get("h", {}).get("title"),
                "away_team": match.get("a", {}).get("title"),
                "home_goals": int(match.get("goals", {}).get("h", 0)),
                "away_goals": int(match.get("goals", {}).get("a", 0)),
                "home_xg": float(match.get("xG", {}).get("h", 0)),
                "away_xg": float(match.get("xG", {}).get("a", 0)),
                "league": league_name,
                "season": f"{season}-{str(season+1)[-2:]}",
                "is_result": match.get("isResult", False),
            })

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df["home_team_id"] = df["home_team"].apply(lambda x: self.normalise_team(str(x)))
            df["away_team_id"] = df["away_team"].apply(lambda x: self.normalise_team(str(x)))
        return df

    @staticmethod
    def _extract_json_var(html: str, var_name: str):
        """Extract a JSON variable from Understat's inline JavaScript."""
        pattern = rf"var\s+{var_name}\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, html)
        if not match:
            return None
        # Understat escapes hex characters
        raw = match.group(1)
        decoded = raw.encode().decode("unicode_escape")
        return json.loads(decoded)
```

**Step 2: Write tests**

```python
# tests/test_collectors/test_understat.py
import pytest
from sharpedge.collectors.understat import UnderstatCollector, UNDERSTAT_LEAGUES


def test_league_codes():
    assert len(UNDERSTAT_LEAGUES) == 5
    assert UNDERSTAT_LEAGUES["Premier League"] == "EPL"


@pytest.mark.integration
def test_collect_single_league_season():
    collector = UnderstatCollector()
    df = collector.collect(league="Premier League", season=2023)
    assert len(df) > 300  # Full season ~380 matches
    assert "home_xg" in df.columns
    assert "away_xg" in df.columns
    assert df["home_xg"].dtype == float
```

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: Understat collector — shot-level xG data from inline JSON"
```

---

## Task 8: FBref Collector (via soccerdata)

**Files:**
- Create: `src/sharpedge/collectors/fbref.py`
- Create: `tests/test_collectors/test_fbref.py`

**Step 1: Implement collector (wraps soccerdata library)**

```python
# src/sharpedge/collectors/fbref.py
import logging
import pandas as pd
import soccerdata as sd
from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

FBREF_LEAGUES = {
    "Premier League": "ENG-Premier League",
    "La Liga": "ESP-La Liga",
    "Bundesliga": "GER-Bundesliga",
    "Serie A": "ITA-Serie A",
    "Ligue 1": "FRA-Ligue 1",
}


class FBrefCollector(BaseCollector):
    source_name = "fbref"
    base_url = "https://fbref.com"
    request_delay = 4.0  # FBref is strict: max 10 req/min

    def _collect(self, league: str = None, season: str = None,
                 stat_type: str = "schedule", **kwargs) -> pd.DataFrame:
        """Collect stats from FBref via soccerdata.

        Args:
            league: League name, or None for all Big 5
            season: Season (e.g. "2324"), or None for latest
            stat_type: "schedule" (match list), "shooting", "passing", "defense", etc.
        """
        leagues = [FBREF_LEAGUES[league]] if league else list(FBREF_LEAGUES.values())
        seasons = [season] if season else None

        try:
            fbref = sd.FBref(leagues=leagues, seasons=seasons)

            if stat_type == "schedule":
                df = fbref.read_schedule()
            elif stat_type == "shooting":
                df = fbref.read_team_season_stats(stat_type="shooting")
            elif stat_type == "passing":
                df = fbref.read_team_season_stats(stat_type="passing")
            elif stat_type == "defense":
                df = fbref.read_team_season_stats(stat_type="defense")
            else:
                df = fbref.read_team_season_stats(stat_type=stat_type)

            # Reset multi-index if present
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()

            return df
        except Exception as e:
            logger.error(f"FBref collection failed: {e}")
            raise
```

**Step 2: Write tests**

```python
# tests/test_collectors/test_fbref.py
import pytest
from sharpedge.collectors.fbref import FBrefCollector, FBREF_LEAGUES


def test_fbref_leagues():
    assert len(FBREF_LEAGUES) == 5


@pytest.mark.integration
def test_collect_schedule():
    collector = FBrefCollector()
    df = collector.collect(league="Premier League", season="2324", stat_type="schedule")
    assert len(df) > 0
```

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: FBref collector — comprehensive match stats via soccerdata"
```

---

## Task 9: Forebet Collector (Competitor Predictions)

**Files:**
- Create: `src/sharpedge/collectors/forebet.py`
- Create: `tests/test_collectors/test_forebet.py`

**Step 1: Implement collector**

```python
# src/sharpedge/collectors/forebet.py
import logging
import re
from datetime import date
import pandas as pd
from bs4 import BeautifulSoup
from sharpedge.collectors.base import BaseCollector, StructureChangedError

logger = logging.getLogger(__name__)

FOREBET_LEAGUES = {
    "Premier League": "england/premier-league",
    "La Liga": "spain/la-liga",
    "Bundesliga": "germany/bundesliga",
    "Serie A": "italy/serie-a",
    "Ligue 1": "france/ligue-1",
}


class ForebetCollector(BaseCollector):
    source_name = "forebet"
    base_url = "https://www.forebet.com/en/football-tips-and-predictions-for"
    request_delay = 3.0

    def _collect(self, league: str = None, **kwargs) -> pd.DataFrame:
        """Collect today's predictions from Forebet."""
        leagues = {league: FOREBET_LEAGUES[league]} if league else FOREBET_LEAGUES

        all_data = []
        for league_name, league_path in leagues.items():
            try:
                df = self._scrape_league(league_path, league_name)
                all_data.append(df)
            except Exception as e:
                self._error_count += 1
                logger.warning(f"Forebet failed for {league_name}: {e}")

        if not all_data:
            return pd.DataFrame()
        return pd.concat(all_data, ignore_index=True)

    def _scrape_league(self, league_path: str, league_name: str) -> pd.DataFrame:
        """Scrape predictions for a single league."""
        url = f"{self.base_url}-{league_path}"
        response = self._fetch(url)
        soup = BeautifulSoup(response.text, "lxml")

        # Structure check
        rows_container = soup.select(".rcnt")
        if not rows_container:
            raise StructureChangedError("Forebet: .rcnt container not found")

        rows = []
        for row in rows_container:
            try:
                teams_el = row.select_one(".homemark, .tnms")
                if not teams_el:
                    continue

                teams_text = teams_el.get_text(separator="|").strip()
                parts = [t.strip() for t in teams_text.split("|") if t.strip()]
                if len(parts) < 2:
                    continue
                home_team, away_team = parts[0], parts[1]

                # Probabilities (1X2)
                probs = row.select(".fprc span")
                prob_home = float(probs[0].text) / 100 if len(probs) > 0 and probs[0].text else None
                prob_draw = float(probs[1].text) / 100 if len(probs) > 1 and probs[1].text else None
                prob_away = float(probs[2].text) / 100 if len(probs) > 2 and probs[2].text else None

                # Predicted score
                score_el = row.select_one(".predict_score, .ex_sc")
                score_home, score_away = None, None
                if score_el:
                    score_text = score_el.get_text().strip()
                    score_match = re.match(r"(\d+)\s*[-–]\s*(\d+)", score_text)
                    if score_match:
                        score_home = int(score_match.group(1))
                        score_away = int(score_match.group(2))

                rows.append({
                    "home_team": home_team,
                    "away_team": away_team,
                    "prob_home": prob_home,
                    "prob_draw": prob_draw,
                    "prob_away": prob_away,
                    "predicted_score_home": score_home,
                    "predicted_score_away": score_away,
                    "league": league_name,
                    "source": "forebet",
                    "scraped_date": date.today().isoformat(),
                })
            except Exception as e:
                logger.debug(f"Skipped row: {e}")
                continue

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df["home_team_id"] = df["home_team"].apply(lambda x: self.normalise_team(str(x)))
            df["away_team_id"] = df["away_team"].apply(lambda x: self.normalise_team(str(x)))
        return df
```

**Step 2: Write tests + commit**

```bash
git add -A
git commit -m "feat: Forebet collector — competitor ML predictions"
```

---

## Task 10: Open-Meteo Weather Collector

**Files:**
- Create: `src/sharpedge/collectors/open_meteo.py`
- Create: `data/stadiums.json` (stadium → lat/lon mapping)

**Step 1: Implement collector**

```python
# src/sharpedge/collectors/open_meteo.py
import json
import logging
from pathlib import Path
import pandas as pd
from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

STADIUMS_PATH = Path(__file__).parent.parent.parent.parent / "data" / "stadiums.json"


class OpenMeteoCollector(BaseCollector):
    source_name = "open_meteo"
    base_url = "https://api.open-meteo.com/v1"
    request_delay = 0.5  # Very generous free API

    def __init__(self):
        super().__init__()
        with open(STADIUMS_PATH) as f:
            self._stadiums = json.load(f)

    def _collect(self, venue: str = None, date_str: str = None,
                 hour: int = 15, **kwargs) -> pd.DataFrame:
        """Get weather for a match venue at a specific date/time.

        Args:
            venue: Stadium name (must be in stadiums.json)
            date_str: Match date (YYYY-MM-DD)
            hour: Kick-off hour (0-23)
        """
        if not venue or not date_str:
            return pd.DataFrame()

        coords = self._stadiums.get(venue)
        if not coords:
            logger.warning(f"Unknown venue: {venue}")
            return pd.DataFrame()

        lat, lon = coords["lat"], coords["lon"]
        url = (
            f"{self.base_url}/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m,weather_code"
            f"&start_date={date_str}&end_date={date_str}"
            f"&timezone=Europe/London"
        )

        data = self._fetch_json(url)
        hourly = data.get("hourly", {})

        if not hourly or hour >= len(hourly.get("time", [])):
            return pd.DataFrame()

        return pd.DataFrame([{
            "venue": venue,
            "date": date_str,
            "hour": hour,
            "temperature_c": hourly["temperature_2m"][hour],
            "precipitation_mm": hourly["precipitation"][hour],
            "wind_speed_kmh": hourly["wind_speed_10m"][hour],
            "humidity_pct": hourly["relative_humidity_2m"][hour],
            "weather_code": hourly["weather_code"][hour],
        }])
```

**Step 2: Create stadiums.json (Big 5 league venues) and tests**

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: Open-Meteo weather collector — match-time conditions"
```

---

## Task 11: Validation Engine (4 Layers)

**Files:**
- Create: `src/sharpedge/validation/__init__.py`
- Create: `src/sharpedge/validation/schema.py`
- Create: `src/sharpedge/validation/statistical.py`
- Create: `src/sharpedge/validation/cross_source.py`
- Create: `src/sharpedge/validation/freshness.py`
- Create: `tests/test_validation/test_schema.py`
- Create: `tests/test_validation/test_statistical.py`

**Step 1: Schema validator**

```python
# src/sharpedge/validation/schema.py
import logging
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str]
    warnings: list[str]


# Expected schemas per record type
SCHEMAS = {
    "match": {
        "required": ["match_date", "home_team_id", "away_team_id", "league", "season"],
        "types": {"home_goals": "numeric", "away_goals": "numeric"},
    },
    "xg": {
        "required": ["match_date", "home_team_id", "away_team_id", "home_xg", "away_xg"],
        "types": {"home_xg": "numeric", "away_xg": "numeric"},
    },
    "odds": {
        "required": ["match_date", "home_team_id", "away_team_id"],
        "types": {},
    },
    "prediction": {
        "required": ["home_team", "away_team", "source"],
        "types": {"prob_home": "numeric", "prob_draw": "numeric", "prob_away": "numeric"},
    },
    "elo": {
        "required": ["club", "elo"],
        "types": {"elo": "numeric"},
    },
}


def validate_schema(df: pd.DataFrame, record_type: str) -> ValidationResult:
    """Validate DataFrame matches expected schema."""
    errors = []
    warnings = []
    schema = SCHEMAS.get(record_type)

    if schema is None:
        return ValidationResult(True, [], [f"No schema defined for '{record_type}'"])

    # Check required columns
    for col in schema["required"]:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    # Check for nulls in required columns
    for col in schema["required"]:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                null_pct = null_count / len(df) * 100
                if null_pct > 10:
                    errors.append(f"Column '{col}' has {null_pct:.1f}% nulls (>{10}% threshold)")
                elif null_pct > 0:
                    warnings.append(f"Column '{col}' has {null_count} nulls ({null_pct:.1f}%)")

    # Check types
    for col, expected_type in schema.get("types", {}).items():
        if col in df.columns and expected_type == "numeric":
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' should be numeric, got {df[col].dtype}")

    return ValidationResult(passed=len(errors) == 0, errors=errors, warnings=warnings)
```

**Step 2: Statistical validator**

```python
# src/sharpedge/validation/statistical.py
import logging
import pandas as pd
from sharpedge.validation.schema import ValidationResult

logger = logging.getLogger(__name__)

# Expected value ranges
RANGES = {
    "home_xg": (0.0, 8.0),
    "away_xg": (0.0, 8.0),
    "home_goals": (0, 15),
    "away_goals": (0, 15),
    "elo": (800, 2200),
    "prob_home": (0.0, 1.0),
    "prob_draw": (0.0, 1.0),
    "prob_away": (0.0, 1.0),
    "odds_home": (1.01, 100.0),
    "odds_draw": (1.01, 100.0),
    "odds_away": (1.01, 100.0),
    "temperature_c": (-20.0, 50.0),
    "wind_speed_kmh": (0.0, 200.0),
}


def validate_statistical(df: pd.DataFrame) -> ValidationResult:
    """Check values fall within expected statistical ranges."""
    errors = []
    warnings = []

    for col, (low, high) in RANGES.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        out_of_range = ((series < low) | (series > high)).sum()
        if out_of_range > 0:
            pct = out_of_range / len(df) * 100
            msg = f"'{col}': {out_of_range} values ({pct:.1f}%) outside [{low}, {high}]"
            if pct > 5:
                errors.append(msg)
            else:
                warnings.append(msg)

    # Check for duplicates
    if "home_team_id" in df.columns and "away_team_id" in df.columns and "match_date" in df.columns:
        dupes = df.duplicated(subset=["home_team_id", "away_team_id", "match_date"]).sum()
        if dupes > 0:
            errors.append(f"{dupes} duplicate matches detected")

    return ValidationResult(passed=len(errors) == 0, errors=errors, warnings=warnings)
```

**Step 3: Write tests and commit**

```bash
git add -A
git commit -m "feat: validation engine — schema + statistical + cross-source validators"
```

---

## Task 12: Orchestrator + Backfill Script

**Files:**
- Create: `src/sharpedge/orchestrator.py`
- Create: `scripts/backfill.py`
- Create: `scripts/health_check.py`

**Step 1: Create orchestrator**

The orchestrator runs all collectors in dependency order, validates output, and promotes to production tables.

```python
# src/sharpedge/orchestrator.py
import logging
from dataclasses import dataclass

from sharpedge.collectors.football_data_uk import FootballDataUKCollector
from sharpedge.collectors.club_elo import ClubEloCollector
from sharpedge.collectors.understat import UnderstatCollector
from sharpedge.collectors.fbref import FBrefCollector
from sharpedge.collectors.forebet import ForebetCollector
from sharpedge.collectors.open_meteo import OpenMeteoCollector
from sharpedge.validation.schema import validate_schema
from sharpedge.validation.statistical import validate_statistical
from sharpedge.alerts.telegram import send_alert_sync

logger = logging.getLogger(__name__)


@dataclass
class CollectionResult:
    source: str
    rows: int
    status: str  # healthy, degraded, down
    errors: list[str]


def run_daily_collection() -> list[CollectionResult]:
    """Run all collectors for daily update."""
    results = []

    collectors = [
        ("football_data_uk", FootballDataUKCollector(), {"season": "2024-25"}),
        ("clubelo", ClubEloCollector(), {}),
        ("understat", UnderstatCollector(), {"season": 2024}),
        ("forebet", ForebetCollector(), {}),
    ]

    for name, collector, kwargs in collectors:
        logger.info(f"Running collector: {name}")
        df = collector.collect(**kwargs)

        if len(df) == 0:
            results.append(CollectionResult(name, 0, "down", ["No data returned"]))
            continue

        # Validate
        schema_result = validate_schema(df, _infer_type(name))
        stat_result = validate_statistical(df)

        all_errors = schema_result.errors + stat_result.errors
        status = "healthy" if not all_errors else "degraded"

        results.append(CollectionResult(name, len(df), status, all_errors))

        if all_errors:
            send_alert_sync(f"Validation errors for {name}:\n" + "\n".join(all_errors))

    return results


def run_backfill(seasons: list[str] = None) -> list[CollectionResult]:
    """Backfill historical data for specified seasons."""
    if seasons is None:
        seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

    results = []

    # Football-Data.co.uk — all seasons, all leagues
    logger.info("Backfilling Football-Data.co.uk...")
    fd_collector = FootballDataUKCollector()
    df = fd_collector.collect()
    results.append(CollectionResult("football_data_uk", len(df), "healthy" if len(df) > 0 else "down", []))

    # ClubELO — weekly snapshots across all seasons
    logger.info("Backfilling ClubELO ratings...")
    elo_collector = ClubEloCollector()
    df = elo_collector.collect(date_range=("2020-08-01", "2025-06-30"))
    results.append(CollectionResult("clubelo", len(df), "healthy" if len(df) > 0 else "down", []))

    # Understat — all seasons
    logger.info("Backfilling Understat xG data...")
    us_collector = UnderstatCollector()
    df = us_collector.collect()
    results.append(CollectionResult("understat", len(df), "healthy" if len(df) > 0 else "down", []))

    return results


def _infer_type(source_name: str) -> str:
    """Map source name to record type for validation."""
    mapping = {
        "football_data_uk": "match",
        "clubelo": "elo",
        "understat": "xg",
        "fbref": "match",
        "forebet": "prediction",
    }
    return mapping.get(source_name, "match")
```

**Step 2: Create backfill script**

```python
# scripts/backfill.py
"""Run historical data backfill for all sources.

Usage: python scripts/backfill.py [--seasons 2020-21 2021-22 ...]
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sharpedge.orchestrator import run_backfill

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical data")
    parser.add_argument("--seasons", nargs="+", default=None, help="Seasons to backfill")
    args = parser.parse_args()

    print("=" * 60)
    print("SharpEdge AI — Historical Data Backfill")
    print("=" * 60)

    results = run_backfill(args.seasons)

    print("\n" + "=" * 60)
    print("BACKFILL RESULTS")
    print("=" * 60)
    for r in results:
        icon = "✅" if r.status == "healthy" else "⚠️" if r.status == "degraded" else "❌"
        print(f"  {icon} {r.source}: {r.rows} rows ({r.status})")
        for err in r.errors:
            print(f"      ⚠ {err}")
    print("=" * 60)
```

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: orchestrator + backfill script — runs all collectors with validation"
```

---

## Task 13: Remaining Collectors (PredictZ, WinDrawWin, FootyStats, football-data.org)

Implement 4 more collectors following the exact same BaseCollector pattern. Each one:
- Inherits BaseCollector
- Implements `_collect()`
- Uses structural fingerprinting
- Has rate limiting configured per source
- Normalises team names
- Has integration tests

**Files to create:**
- `src/sharpedge/collectors/predictz.py`
- `src/sharpedge/collectors/windrawwin.py`
- `src/sharpedge/collectors/footystats.py`
- `src/sharpedge/collectors/football_data_org.py`

Each follows the Forebet collector as a template: fetch HTML, parse with BeautifulSoup, extract predictions/stats, normalise teams, return DataFrame.

**Commit after each collector.**

---

## Task 14: Integration Test — Full Pipeline

**Files:**
- Create: `tests/test_integration/test_full_pipeline.py`

**Step 1: Write end-to-end test**

```python
# tests/test_integration/test_full_pipeline.py
import pytest
from sharpedge.collectors.football_data_uk import FootballDataUKCollector
from sharpedge.collectors.club_elo import ClubEloCollector
from sharpedge.collectors.understat import UnderstatCollector
from sharpedge.validation.schema import validate_schema
from sharpedge.validation.statistical import validate_statistical


@pytest.mark.integration
class TestFullPipeline:
    def test_football_data_uk_pipeline(self):
        collector = FootballDataUKCollector()
        df = collector.collect(league="Premier League", season="2023-24")
        assert len(df) > 300

        schema = validate_schema(df, "match")
        assert schema.passed, f"Schema errors: {schema.errors}"

        stats = validate_statistical(df)
        assert stats.passed, f"Statistical errors: {stats.errors}"

    def test_elo_pipeline(self):
        collector = ClubEloCollector()
        df = collector.collect(date_str="2025-01-01")
        assert len(df) > 400

        schema = validate_schema(df, "elo")
        assert schema.passed

    def test_understat_pipeline(self):
        collector = UnderstatCollector()
        df = collector.collect(league="Premier League", season=2023)
        assert len(df) > 300
        assert "home_xg" in df.columns

        schema = validate_schema(df, "xg")
        assert schema.passed
```

Run: `pytest tests/test_integration/ -v -m integration`

**Step 2: Commit**

```bash
git add -A
git commit -m "test: integration tests for full data pipeline"
```

---

## Summary: Phase 1 Deliverables

After completing all 14 tasks, you will have:

| Component | What It Does |
|-----------|-------------|
| **PostgreSQL Schema** | 12 tables covering matches, stats, xG, odds, ELO, predictions, weather, health, staging |
| **Team Name Registry** | ~100 teams across Big 5 leagues with per-source aliases + fuzzy matching |
| **BaseCollector** | Bulletproof foundation with retry, caching, rate limiting, health, validation, alerting |
| **10 Collectors** | Football-Data.co.uk, ClubELO, Understat, FBref, Forebet, PredictZ, WinDrawWin, FootyStats, Open-Meteo, football-data.org |
| **4-Layer Validation** | Schema, statistical, cross-source, freshness validation |
| **Orchestrator** | Runs all collectors in order with dependency management |
| **Backfill Script** | One command to populate 5 seasons of historical data |
| **Health Monitoring** | Source health tracking + Telegram alerts |

**Next Phase:** Phase 2 — ML Engine (Feature Engineering, XGBoost, Poisson, Ensemble, Banker Filter)
