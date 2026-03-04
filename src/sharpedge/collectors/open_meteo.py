"""
Open-Meteo Weather Collector — match-day weather data.

Fetches weather forecasts/history from the free Open-Meteo API for
stadium locations. No API key required.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

# Resolve stadiums.json relative to project root
_STADIUMS_PATH = Path(__file__).resolve().parents[3] / "data" / "stadiums.json"


def _load_stadiums() -> dict[str, dict[str, float]]:
    """Load stadium coordinates from the JSON registry."""
    try:
        with open(_STADIUMS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load stadiums.json: {e}")
        return {}


class OpenMeteoCollector(BaseCollector):
    """Collector for Open-Meteo weather data at stadium locations."""

    source_name = "open_meteo"
    base_url = "https://api.open-meteo.com"
    request_delay = 0.5

    def __init__(self) -> None:
        super().__init__()
        self._stadiums = _load_stadiums()

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect weather data for a specific venue and date.

        Parameters
        ----------
        venue : str, optional
            Stadium name (must exist in stadiums.json).
            If None, returns empty DataFrame.
        date_str : str, optional
            Date in YYYY-MM-DD format. Defaults to today.
        hour : int, optional
            Hour of day (0-23) for the weather reading. Defaults to 15.
        """
        venue: Optional[str] = kwargs.get("venue")
        date_str: Optional[str] = kwargs.get("date_str")
        hour: int = kwargs.get("hour", 15)

        if not venue:
            logger.warning("No venue specified for weather collection")
            return pd.DataFrame()

        if venue not in self._stadiums:
            logger.warning(f"Unknown venue: {venue}")
            return pd.DataFrame()

        if not date_str:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        coords = self._stadiums[venue]
        lat = coords["lat"]
        lon = coords["lon"]

        url = (
            f"{self.base_url}/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,precipitation,wind_speed_10m,"
            f"relative_humidity_2m,weather_code"
            f"&start_date={date_str}&end_date={date_str}"
            f"&timezone=Europe/London"
        )

        try:
            data = self._fetch_json(url)
            hourly = data.get("hourly", {})

            times = hourly.get("time", [])
            # Find the index for the requested hour
            hour_index = min(hour, len(times) - 1) if times else 0

            if not times:
                logger.warning(f"No hourly data returned for {venue}")
                return pd.DataFrame()

            return pd.DataFrame([{
                "venue": venue,
                "date": date_str,
                "hour": hour,
                "latitude": lat,
                "longitude": lon,
                "temperature_c": hourly.get("temperature_2m", [None])[hour_index],
                "precipitation_mm": hourly.get("precipitation", [None])[hour_index],
                "wind_speed_kmh": hourly.get("wind_speed_10m", [None])[hour_index],
                "humidity_pct": hourly.get("relative_humidity_2m", [None])[hour_index],
                "weather_code": hourly.get("weather_code", [None])[hour_index],
            }])

        except Exception as e:
            logger.error(f"Open-Meteo collection failed for {venue}: {e}")
            raise

    def collect_batch(self, matches: list[dict]) -> pd.DataFrame:
        """Collect weather data for multiple matches.

        Parameters
        ----------
        matches : list of dict
            Each dict must have keys: "venue" (str), "date" (str),
            and optionally "hour" (int, default 15).

        Returns
        -------
        pd.DataFrame
            Combined weather data for all valid venues.
        """
        frames: list[pd.DataFrame] = []

        for match in matches:
            venue = match.get("venue")
            date_str = match.get("date")
            hour = match.get("hour", 15)

            df = self.collect(venue=venue, date_str=date_str, hour=hour)
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)
