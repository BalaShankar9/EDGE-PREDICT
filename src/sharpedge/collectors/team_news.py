"""
TeamNewsCollector — team news, press conferences, and injury updates.

Aggregates football news from BBC Sport, Sky Sports, The Guardian, and
ESPN FC.  Each article is classified by type (injury / suspension /
return / tactical / general) and sentiment (positive / negative /
neutral), plus an impact player is extracted when identifiable.

Sources:
  1. BBC Sport Football  — https://www.bbc.com/sport/football
  2. Sky Sports Football — https://www.skysports.com/football/news
  3. The Guardian        — https://www.theguardian.com/football
  4. ESPN FC             — https://www.espn.com/soccer/

_collect() row schema
---------------------
  match_id      str or None    if the article relates to a specific match
  team          str or None    team the article concerns
  news_date     str            publication date (ISO-8601 or raw text)
  headline      str
  summary       str            first 200 characters of the article body
  news_type     str            "injury" | "suspension" | "return" |
                               "transfer" | "tactical" | "general"
  impact_player str or None    player named in article
  sentiment     str            "positive" | "negative" | "neutral"
  source_url    str
  source_name   str
  scraped_at    str
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Source configurations                                              #
# ------------------------------------------------------------------ #

_SOURCES: list[dict] = [
    {
        "name": "bbc_sport",
        "url": "https://www.bbc.com/sport/football",
        "article_selector": "a[href*='/sport/football/'], a[class*='media__link']",
        "headline_selector": "h3, h2, [class*='media__title']",
        "summary_selector": "[class*='media__summary'], p",
        "date_selector": "time, [class*='date'], [datetime]",
    },
    {
        "name": "sky_sports",
        "url": "https://www.skysports.com/football/news",
        "article_selector": "a[href*='/football/news/']",
        "headline_selector": "h4, h3, [class*='news-list__headline']",
        "summary_selector": "[class*='news-list__summary'], p",
        "date_selector": "time, [class*='news-list__time']",
    },
    {
        "name": "guardian",
        "url": "https://www.theguardian.com/football",
        "article_selector": "a[href*='/football/']",
        "headline_selector": "[class*='fc-item__title'], h3, h2",
        "summary_selector": "[class*='fc-item__standfirst'], p",
        "date_selector": "time, [data-timestamp]",
    },
    {
        "name": "espn",
        "url": "https://www.espn.com/soccer/",
        "article_selector": "a[href*='/soccer/story/']",
        "headline_selector": "[class*='contentItem__title'], h2, h3",
        "summary_selector": "[class*='contentItem__subheadline'], p",
        "date_selector": "[class*='contentMeta__info'], time",
    },
]

# ------------------------------------------------------------------ #
#  Classification keyword sets                                        #
# ------------------------------------------------------------------ #

_INJURY_KEYWORDS: frozenset[str] = frozenset([
    "injured", "injury", "ruled out", "sidelined", "doubt", "doubtful",
    "miss", "misses", "missing", "hamstring", "knee", "ankle", "muscle",
    "strain", "tear", "broken", "fracture", "surgery", "operation",
    "concussion", "illness", "ill", "sick", "fitness concern", "out for",
    "weeks out", "months out",
])

_RETURN_KEYWORDS: frozenset[str] = frozenset([
    "return", "returns", "returning", "back in training", "fit",
    "available", "recovery", "recovered", "fitness", "passed fit",
    "back in contention", "back in squad",
])

_SUSPENSION_KEYWORDS: frozenset[str] = frozenset([
    "suspended", "ban", "banned", "red card", "suspension",
    "yellow card accumulation", "serving ban", "UEFA ban",
])

_TACTICAL_KEYWORDS: frozenset[str] = frozenset([
    "formation", "system", "tactics", "tactical", "lineup change",
    "line-up", "setup", "shape", "4-3-3", "4-2-3-1", "pressing",
    "high press", "build-up", "role", "position",
])

_TRANSFER_KEYWORDS: frozenset[str] = frozenset([
    "transfer", "signing", "signed", "loan", "deal", "fee",
    "bid", "offers", "contract", "released", "free agent",
])

# Regex to extract a plausible player name from headline text
# Matches two or three capitalised words (typical football player name)
_PLAYER_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})\b")

# Max articles to process per source (prevents runaway scraping)
_MAX_ARTICLES_PER_SOURCE = 30


class TeamNewsCollector(BaseCollector):
    """Collector for football team news, injury reports, and press conference updates."""

    source_name = "team_news"
    base_url = ""
    request_delay = 3.0

    # ------------------------------------------------------------------ #
    #  Main entry-point                                                   #
    # ------------------------------------------------------------------ #

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect team news from all configured sources.

        Parameters
        ----------
        team : str, optional
            Filter articles to those mentioning a specific team name.
        news_types : list[str], optional
            Only return rows matching these types (e.g. ["injury", "return"]).
        source : str, optional
            Restrict to a single source name (e.g. "bbc_sport").

        Returns
        -------
        pd.DataFrame
            One row per article; schema described in module docstring.
        """
        filter_team: Optional[str] = kwargs.get("team")
        filter_types: Optional[list[str]] = kwargs.get("news_types")
        restrict_source: Optional[str] = kwargs.get("source")

        frames: list[pd.DataFrame] = []

        for source_cfg in _SOURCES:
            if restrict_source and source_cfg["name"] != restrict_source:
                continue
            try:
                df = self._collect_source(source_cfg)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                logger.error(
                    f"[{self.source_name}] Source {source_cfg['name']} failed: {exc}"
                )

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)

        # Apply optional filters
        if filter_team:
            team_lower = filter_team.lower()
            mask = df["headline"].str.lower().str.contains(team_lower, na=False) | \
                   df["summary"].str.lower().str.contains(team_lower, na=False) | \
                   df["team"].str.lower().str.contains(team_lower, na=False)
            df = df[mask].copy()

        if filter_types:
            df = df[df["news_type"].isin(filter_types)].copy()

        logger.info(f"[{self.source_name}] Total articles collected: {len(df)}")
        return df

    # ------------------------------------------------------------------ #
    #  Per-source collection                                              #
    # ------------------------------------------------------------------ #

    def _collect_source(self, source_cfg: dict) -> pd.DataFrame:
        """Fetch and parse the news index page for one source.

        Parameters
        ----------
        source_cfg : dict
            Entry from the _SOURCES list with name, url, and selectors.

        Returns
        -------
        pd.DataFrame
            Parsed articles from this source.
        """
        source_name = source_cfg["name"]
        url = source_cfg["url"]

        cache_key = f"team_news_{source_name}_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H')}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[{self.source_name}] Cache hit for {source_name}")
            return pd.DataFrame(cached)

        response = self._fetch(url)
        html = response.text

        articles = self._extract_articles(html, source_cfg, base_url=url)

        if not articles:
            logger.warning(f"[{self.source_name}] {source_name}: 0 articles extracted")
            return pd.DataFrame()

        logger.info(f"[{self.source_name}] {source_name}: {len(articles)} articles")
        self._set_cache(cache_key, articles)
        return pd.DataFrame(articles)

    # ------------------------------------------------------------------ #
    #  HTML parsing                                                       #
    # ------------------------------------------------------------------ #

    def _extract_articles(
        self, html: str, source_cfg: dict, base_url: str
    ) -> list[dict]:
        """Extract and classify articles from a news index HTML page.

        Parameters
        ----------
        html : str
            Raw HTML from the news index page.
        source_cfg : dict
            Source configuration dict with CSS selectors.
        base_url : str
            Base URL for resolving relative href values.

        Returns
        -------
        list[dict]
            Classified article rows.
        """
        soup = BeautifulSoup(html, "html.parser")
        scraped_at = datetime.now(timezone.utc).isoformat()
        articles: list[dict] = []
        seen_urls: set[str] = set()

        link_els = soup.select(source_cfg["article_selector"])

        for link_el in link_els[:_MAX_ARTICLES_PER_SOURCE]:
            try:
                href = link_el.get("href", "")
                if not href:
                    continue
                article_url = href if href.startswith("http") else urljoin(base_url, href)

                if article_url in seen_urls:
                    continue
                seen_urls.add(article_url)

                # Headline: text of the link itself or a child heading element
                headline_el = link_el.select_one(source_cfg["headline_selector"])
                if headline_el:
                    headline = headline_el.get_text(strip=True)
                else:
                    headline = link_el.get_text(strip=True)

                if not headline or len(headline) < 5:
                    continue

                # Summary: sibling / child summary element, trimmed to 200 chars
                parent = link_el.parent or link_el
                summary_el = parent.select_one(source_cfg["summary_selector"])
                summary = ""
                if summary_el:
                    summary = summary_el.get_text(strip=True)[:200]

                # Date
                date_el = parent.select_one(source_cfg["date_selector"])
                news_date = ""
                if date_el:
                    news_date = (
                        date_el.get("datetime")
                        or date_el.get("data-timestamp")
                        or date_el.get_text(strip=True)
                        or ""
                    )

                # Classification
                combined_text = f"{headline} {summary}".lower()
                news_type = _classify_news_type(combined_text)
                sentiment = _classify_sentiment(combined_text, news_type)
                impact_player = _extract_player_name(headline)
                team = _extract_team_name(headline, summary)

                articles.append({
                    "match_id": None,
                    "team": team,
                    "news_date": str(news_date),
                    "headline": headline,
                    "summary": summary,
                    "news_type": news_type,
                    "impact_player": impact_player,
                    "sentiment": sentiment,
                    "source_url": article_url,
                    "source_name": source_cfg["name"],
                    "scraped_at": scraped_at,
                })

            except Exception as exc:
                logger.debug(f"[{self.source_name}] Article parse error: {exc}")
                continue

        return articles


# ------------------------------------------------------------------ #
#  Module-level classification helpers                               #
# ------------------------------------------------------------------ #

def _classify_news_type(text: str) -> str:
    """Classify article text into a news type category.

    Uses ordered priority: injury > suspension > return > transfer >
    tactical > general.

    Parameters
    ----------
    text : str
        Lower-cased combined headline + summary text.

    Returns
    -------
    str
        One of: "injury", "suspension", "return", "transfer",
        "tactical", "general".
    """
    if any(kw in text for kw in _INJURY_KEYWORDS):
        return "injury"
    if any(kw in text for kw in _SUSPENSION_KEYWORDS):
        return "suspension"
    if any(kw in text for kw in _RETURN_KEYWORDS):
        return "return"
    if any(kw in text for kw in _TRANSFER_KEYWORDS):
        return "transfer"
    if any(kw in text for kw in _TACTICAL_KEYWORDS):
        return "tactical"
    return "general"


def _classify_sentiment(text: str, news_type: str) -> str:
    """Classify the sentiment of a news item.

    Parameters
    ----------
    text : str
        Lower-cased combined headline + summary text.
    news_type : str
        Pre-classified news type from _classify_news_type().

    Returns
    -------
    str
        "positive" — player returning to fitness / available.
        "negative" — player injured / suspended / ruled out.
        "neutral"  — general tactical or transfer news.
    """
    if news_type == "return":
        return "positive"
    if news_type in ("injury", "suspension"):
        return "negative"
    # Some injury articles mention a return in the same breath — check
    if any(kw in text for kw in _RETURN_KEYWORDS):
        return "positive"
    if any(kw in text for kw in _INJURY_KEYWORDS) or any(kw in text for kw in _SUSPENSION_KEYWORDS):
        return "negative"
    return "neutral"


def _extract_player_name(headline: str) -> Optional[str]:
    """Attempt to extract a player name from the headline.

    Uses a simple heuristic: the first capitalised two-to-three word
    sequence that looks like a person name.

    Parameters
    ----------
    headline : str
        Raw headline string.

    Returns
    -------
    str or None
        Best candidate player name, or None.
    """
    # Common words to skip that match the capitalised pattern
    _SKIP_WORDS: frozenset[str] = frozenset([
        "Premier League", "Champions League", "Europa League",
        "FA Cup", "World Cup", "Man City", "Man United",
        "Real Madrid", "Transfer News", "Team News", "Press Conference",
    ])
    matches = _PLAYER_NAME_RE.findall(headline)
    for candidate in matches:
        if candidate not in _SKIP_WORDS and len(candidate) > 3:
            return candidate
    return None


def _extract_team_name(headline: str, summary: str) -> Optional[str]:
    """Try to extract a team name from headline or summary text.

    Uses a naive heuristic: looks for a capitalised word/phrase
    before common football keywords.

    Parameters
    ----------
    headline : str
        Article headline.
    summary : str
        Article summary.

    Returns
    -------
    str or None
        Extracted team name, or None.
    """
    team_triggers = re.compile(
        r"([A-Z][a-zA-Z\s]+?)\s+"
        r"(?:manager|boss|coach|striker|midfielder|defender|keeper|captain|star|ace|winger)",
        re.IGNORECASE,
    )
    text = f"{headline} {summary}"
    match = team_triggers.search(text)
    if match:
        candidate = match.group(1).strip()
        # Drop overly short or generic candidates
        if len(candidate) >= 3 and candidate not in ("The", "A", "An"):
            return candidate
    return None
