from datetime import datetime, date
from typing import Optional, List

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Date,
    DateTime,
    Boolean,
    ForeignKey,
    UniqueConstraint,
    Index,
    Text,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    country: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    aliases: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    home_matches: Mapped[List["Match"]] = relationship(
        "Match", foreign_keys="Match.home_team_id", back_populates="home_team"
    )
    away_matches: Mapped[List["Match"]] = relationship(
        "Match", foreign_keys="Match.away_team_id", back_populates="away_team"
    )
    elo_ratings: Mapped[List["EloRating"]] = relationship("EloRating", back_populates="team")


class League(Base):
    __tablename__ = "leagues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    country: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    fd_uk_code: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    fbref_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    understat_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    season_format: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    seasons: Mapped[List["Season"]] = relationship("Season", back_populates="league")


class Season(Base):
    __tablename__ = "seasons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    league_id: Mapped[int] = mapped_column(Integer, ForeignKey("leagues.id"), nullable=False)
    label: Mapped[str] = mapped_column(String(20), nullable=False)
    start_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    end_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    league: Mapped["League"] = relationship("League", back_populates="seasons")
    matches: Mapped[List["Match"]] = relationship("Match", back_populates="season")


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint("season_id", "match_date", "home_team_id", "away_team_id"),
        Index("ix_matches_match_date", "match_date"),
        Index("ix_matches_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    season_id: Mapped[int] = mapped_column(Integer, ForeignKey("seasons.id"), nullable=False)
    match_date: Mapped[date] = mapped_column(Date, nullable=False)
    kick_off_time: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    home_team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    home_goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_goals_ht: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_goals_ht: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    result: Mapped[Optional[str]] = mapped_column(String(1), nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    referee: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    venue: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    season: Mapped["Season"] = relationship("Season", back_populates="matches")
    home_team: Mapped["Team"] = relationship(
        "Team", foreign_keys=[home_team_id], back_populates="home_matches"
    )
    away_team: Mapped["Team"] = relationship(
        "Team", foreign_keys=[away_team_id], back_populates="away_matches"
    )
    stats: Mapped[Optional["MatchStats"]] = relationship("MatchStats", back_populates="match")
    xg: Mapped[List["MatchXG"]] = relationship("MatchXG", back_populates="match")
    odds: Mapped[List["MatchOdds"]] = relationship("MatchOdds", back_populates="match")
    predictions: Mapped[List["CompetitorPrediction"]] = relationship(
        "CompetitorPrediction", back_populates="match"
    )
    weather: Mapped[Optional["MatchWeather"]] = relationship(
        "MatchWeather", back_populates="match"
    )


class MatchStats(Base):
    __tablename__ = "match_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), unique=True, nullable=False
    )
    source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    home_shots: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_shots_on_target: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_possession: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    home_passes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_pass_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    home_fouls: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_corners: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_yellow_cards: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_red_cards: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_shots: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_shots_on_target: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_possession: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    away_passes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_pass_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    away_fouls: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_corners: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_yellow_cards: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_red_cards: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    extra_stats: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    match: Mapped["Match"] = relationship("Match", back_populates="stats")


class MatchXG(Base):
    __tablename__ = "match_xg"
    __table_args__ = (UniqueConstraint("match_id", "source"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    home_xg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    away_xg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    home_npxg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    away_npxg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    match: Mapped["Match"] = relationship("Match", back_populates="xg")


class MatchOdds(Base):
    __tablename__ = "match_odds"
    __table_args__ = (
        UniqueConstraint("match_id", "bookmaker", "market", "odds_type"),
        Index("ix_match_odds_match_id", "match_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    bookmaker: Mapped[str] = mapped_column(String(100), nullable=False)
    market: Mapped[str] = mapped_column(String(50), nullable=False)
    odds_type: Mapped[str] = mapped_column(String(20), nullable=False)
    odds_home: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_draw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_away: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_over: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_under: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    captured_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    match: Mapped["Match"] = relationship("Match", back_populates="odds")


class EloRating(Base):
    __tablename__ = "elo_ratings"
    __table_args__ = (
        UniqueConstraint("team_id", "rating_date", "source"),
        Index("ix_elo_ratings_rating_date", "rating_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    rating_date: Mapped[date] = mapped_column(Date, nullable=False)
    elo: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)

    team: Mapped["Team"] = relationship("Team", back_populates="elo_ratings")


class CompetitorPrediction(Base):
    __tablename__ = "competitor_predictions"
    __table_args__ = (UniqueConstraint("match_id", "source"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    predicted_result: Mapped[Optional[str]] = mapped_column(String(1), nullable=True)
    prob_home: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prob_draw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prob_away: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    predicted_score_home: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    predicted_score_away: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    btts_prediction: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    over_under_prediction: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    over_under_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    scraped_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    match: Mapped["Match"] = relationship("Match", back_populates="predictions")


class MatchWeather(Base):
    __tablename__ = "match_weather"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), unique=True, nullable=False
    )
    temperature_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precipitation_mm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_speed_kmh: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    weather_code: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    match: Mapped["Match"] = relationship("Match", back_populates="weather")


class SourceHealth(Base):
    __tablename__ = "source_health"
    __table_args__ = (Index("ix_source_health_source_name", "source_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    last_success: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_failure: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    rows_collected: Mapped[int] = mapped_column(Integer, default=0)
    avg_collection_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    checked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class RawStagingRecord(Base):
    __tablename__ = "raw_staging_records"
    __table_args__ = (Index("ix_raw_staging_records_status", "status"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    record_type: Mapped[str] = mapped_column(String(50), nullable=False)
    raw_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    validated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
