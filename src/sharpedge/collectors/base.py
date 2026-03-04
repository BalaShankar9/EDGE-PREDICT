"""
BaseCollector — abstract foundation for all SharpEdge data collectors.

Provides retry logic, rate limiting, caching, health reporting,
team name normalisation, structural fingerprinting, and Telegram alerting.
"""

import hashlib
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from sharpedge.alerts.telegram import send_alert_sync
from sharpedge.config import settings
from sharpedge.normalisation.team_names import normalise

logger = logging.getLogger(__name__)

# Rotating User-Agent pool — real browser UAs
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
]


class BaseCollector(ABC):
    """Abstract base class for all data collectors.

    Subclasses must set:
        source_name: str — unique identifier for this data source
        base_url: str — root URL of the data source
        request_delay: float — minimum seconds between requests (default 3.0)

    Subclasses must implement:
        _collect(**kwargs) -> pd.DataFrame
    """

    source_name: str = ""
    base_url: str = ""
    request_delay: float = 3.0

    def __init__(self) -> None:
        self._session = requests.Session()
        self._last_request_time: float = 0.0
        self._cache_dir = Path(settings.cache_dir) / self.source_name
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  HTTP fetching with rate limiting and retry                         #
    # ------------------------------------------------------------------ #

    def _get_headers(self) -> dict[str, str]:
        """Return request headers with a random User-Agent."""
        return {
            "User-Agent": random.choice(_USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    def _rate_limit(self) -> None:
        """Enforce per-collector rate limiting with random jitter."""
        elapsed = time.time() - self._last_request_time
        delay = self.request_delay + random.uniform(0, 1.5)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=300),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError)),
        reraise=True,
    )
    def _fetch(self, url: str, **kwargs: Any) -> requests.Response:
        """Fetch a URL with rate limiting, rotating UAs, and retry logic."""
        self._rate_limit()
        headers = self._get_headers()
        headers.update(kwargs.pop("headers", {}))
        logger.debug(f"[{self.source_name}] GET {url}")
        response = self._session.get(
            url,
            headers=headers,
            timeout=settings.request_timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response

    def _fetch_json(self, url: str, **kwargs: Any) -> Any:
        """Fetch URL and parse JSON response."""
        response = self._fetch(url, **kwargs)
        return response.json()

    # ------------------------------------------------------------------ #
    #  Caching — MD5-based file cache                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cache_key(key: str) -> str:
        """Generate an MD5 hex digest for a cache key string."""
        return hashlib.md5(key.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[Any]:
        """Retrieve a cached value by key. Returns None on miss."""
        path = self._cache_dir / f"{self._cache_key(key)}.json"
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"[{self.source_name}] Cache read error for '{key}': {e}")
        return None

    def _set_cache(self, key: str, data: Any) -> None:
        """Write a value to the file cache."""
        path = self._cache_dir / f"{self._cache_key(key)}.json"
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except OSError as e:
            logger.warning(f"[{self.source_name}] Cache write error for '{key}': {e}")

    # ------------------------------------------------------------------ #
    #  Team name normalisation                                            #
    # ------------------------------------------------------------------ #

    def normalise_team(self, name: str) -> Optional[str]:
        """Normalise a team name using this collector's source context."""
        return normalise(name, source=self.source_name)

    # ------------------------------------------------------------------ #
    #  Structural fingerprinting                                          #
    # ------------------------------------------------------------------ #

    def _check_structure(self, html: str, expected_markers: list[str]) -> bool:
        """Check if expected string markers are present in the HTML.

        Returns True if ALL markers are found, False otherwise.
        Logs a warning for each missing marker.
        """
        missing = [m for m in expected_markers if m not in html]
        if missing:
            logger.warning(
                f"[{self.source_name}] Structure check failed — "
                f"missing markers: {missing}"
            )
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Health reporting                                                   #
    # ------------------------------------------------------------------ #

    def _report_health(
        self,
        status: str,
        rows: int = 0,
        error: Optional[str] = None,
    ) -> dict[str, Any]:
        """Generate a health-check dict and alert on failures."""
        report = {
            "source": self.source_name,
            "status": status,
            "rows": rows,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if status == "down":
            msg = (
                f"Source <b>{self.source_name}</b> is DOWN\n"
                f"Error: {error or 'unknown'}"
            )
            send_alert_sync(msg)
        return report

    # ------------------------------------------------------------------ #
    #  Main collection interface                                          #
    # ------------------------------------------------------------------ #

    def collect(self, **kwargs: Any) -> pd.DataFrame:
        """Public collection API. Wraps _collect() with error handling.

        Returns the collected DataFrame on success, or an empty DataFrame
        on failure. Never raises exceptions to the caller.
        """
        start = time.time()
        try:
            logger.info(f"[{self.source_name}] Starting collection...")
            df = self._collect(**kwargs)
            elapsed = time.time() - start
            self._report_health("ok", rows=len(df))
            logger.info(
                f"[{self.source_name}] Collection complete — "
                f"{len(df)} rows in {elapsed:.1f}s"
            )
            return df
        except Exception as e:
            elapsed = time.time() - start
            logger.error(
                f"[{self.source_name}] Collection FAILED after {elapsed:.1f}s: {e}"
            )
            self._report_health("down", error=str(e))
            return pd.DataFrame()

    @abstractmethod
    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Subclasses implement their actual scraping logic here."""
        ...
