"""
CrowdSentimentCollector — public crowd wisdom from Reddit betting communities.

Scrapes r/SoccerBetting and r/soccer using Reddit's public JSON API (no
authentication required). Analyses posts and comments for team mentions,
betting pick directions, and sentiment polarity.

The "inverse crowd" signal: when Reddit consensus strongly favours one
outcome, the contrarian position is flagged. Sharp-money fades public
steam; this collector surfaces the raw public lean so the model can decide
whether to follow or fade it.
"""

import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Sentiment keyword dictionaries                                     #
# ------------------------------------------------------------------ #

_POSITIVE_WORDS = frozenset(
    {
        "confident",
        "confidence",
        "lock",
        "banker",
        "easy",
        "strong",
        "value",
        "win",
        "winning",
        "sure",
        "safe",
        "solid",
        "clear",
        "obvious",
        "smash",
        "hammer",
        "guaranteed",
        "free money",
        "no brainer",
    }
)

_NEGATIVE_WORDS = frozenset(
    {
        "risky",
        "risk",
        "avoid",
        "trap",
        "fade",
        "skip",
        "dangerous",
        "danger",
        "worry",
        "worried",
        "unsure",
        "uncertain",
        "doubt",
        "doubting",
        "nervous",
        "scared",
        "tough",
        "difficult",
        "nightmare",
        "stay away",
        "no bet",
        "not sure",
    }
)

# Outcome pick keywords mapped to canonical labels
_PICK_KEYWORDS: dict[str, str] = {
    # Home win
    "home win": "home",
    "home": "home",
    "1x2: 1": "home",
    "backing home": "home",
    "back home": "home",
    "home ml": "home",
    # Draw
    "draw": "draw",
    "x": "draw",
    "1x2: x": "draw",
    "backing draw": "draw",
    # Away win
    "away win": "away",
    "away": "away",
    "1x2: 2": "away",
    "backing away": "away",
    "back away": "away",
    "away ml": "away",
    # Goals markets
    "over 2.5": "over",
    "over 1.5": "over",
    "over 3.5": "over",
    "o2.5": "over",
    "o1.5": "over",
    "o3.5": "over",
    "under 2.5": "under",
    "under 1.5": "under",
    "under 3.5": "under",
    "u2.5": "under",
    "u1.5": "under",
    "u3.5": "under",
    "btts yes": "btts_yes",
    "btts no": "btts_no",
    "both teams to score": "btts_yes",
}

# Reddit public JSON endpoints — no API key required
_REDDIT_SOCCERBETTING_URL = (
    "https://www.reddit.com/r/SoccerBetting/.json?limit=100&sort=new"
)
_REDDIT_MATCH_THREADS_URL = (
    "https://www.reddit.com/r/soccer/search.json"
    "?q=match+thread&restrict_sr=1&sort=new&limit=25"
)

# Headers Reddit expects for non-browser JSON clients
_REDDIT_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


class CrowdSentimentCollector(BaseCollector):
    """Collector for crowd sentiment signals from Reddit betting communities."""

    source_name = "crowd_sentiment"
    base_url = ""
    request_delay = 5.0

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect and score crowd sentiment from Reddit.

        Parameters
        ----------
        home_team : str, optional
            If provided, filter results to posts mentioning this team.
        away_team : str, optional
            If provided, filter results to posts mentioning this team.
        include_match_threads : bool, optional
            Also fetch r/soccer match threads (default False).

        Returns
        -------
        pd.DataFrame
            One row per detected match pairing.  Columns:
              match_id, home_team, away_team,
              reddit_home_mentions, reddit_away_mentions,
              reddit_home_sentiment, reddit_away_sentiment,
              reddit_consensus_pick, reddit_confidence,
              reddit_contrarian_pick,
              post_count, avg_upvote_ratio
        """
        home_filter: Optional[str] = kwargs.get("home_team")
        away_filter: Optional[str] = kwargs.get("away_team")
        include_threads: bool = bool(kwargs.get("include_match_threads", False))

        posts = self._fetch_soccerbetting_posts()

        if include_threads:
            thread_posts = self._fetch_match_thread_posts()
            posts.extend(thread_posts)

        if not posts:
            logger.warning(
                f"[{self.source_name}] No Reddit posts retrieved — "
                "returning empty DataFrame"
            )
            return pd.DataFrame()

        # Aggregate sentiment across all posts
        match_data = self._aggregate_posts(posts)

        if not match_data:
            return pd.DataFrame()

        rows = list(match_data.values())
        df = pd.DataFrame(rows)

        # Filter by requested teams if provided
        if home_filter or away_filter:
            mask = pd.Series([True] * len(df))
            if home_filter:
                ht = home_filter.lower()
                mask &= (
                    df["home_team"].str.lower().str.contains(ht, na=False)
                    | df["away_team"].str.lower().str.contains(ht, na=False)
                )
            if away_filter:
                at = away_filter.lower()
                mask &= (
                    df["home_team"].str.lower().str.contains(at, na=False)
                    | df["away_team"].str.lower().str.contains(at, na=False)
                )
            df = df[mask].reset_index(drop=True)

        # Normalise team names using BaseCollector utility
        df["home_team_id"] = df["home_team"].apply(
            lambda x: self.normalise_team(str(x)) if x else None
        )
        df["away_team_id"] = df["away_team"].apply(
            lambda x: self.normalise_team(str(x)) if x else None
        )

        df["scraped_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        df["source"] = self.source_name

        logger.info(
            f"[{self.source_name}] Processed {len(posts)} posts → "
            f"{len(df)} match sentiment rows"
        )
        return df

    # ------------------------------------------------------------------ #
    #  Reddit fetching                                                     #
    # ------------------------------------------------------------------ #

    def _fetch_soccerbetting_posts(self) -> list[dict]:
        """Fetch recent posts from r/SoccerBetting via Reddit public API."""
        try:
            data = self._fetch_json(
                _REDDIT_SOCCERBETTING_URL, headers=_REDDIT_HEADERS
            )
            return self._extract_posts_from_listing(data)
        except Exception as e:
            logger.error(
                f"[{self.source_name}] Failed to fetch r/SoccerBetting: {e}"
            )
            return []

    def _fetch_match_thread_posts(self) -> list[dict]:
        """Fetch recent match thread posts from r/soccer."""
        try:
            data = self._fetch_json(
                _REDDIT_MATCH_THREADS_URL, headers=_REDDIT_HEADERS
            )
            return self._extract_posts_from_listing(data)
        except Exception as e:
            logger.error(
                f"[{self.source_name}] Failed to fetch r/soccer match threads: {e}"
            )
            return []

    @staticmethod
    def _extract_posts_from_listing(data: Any) -> list[dict]:
        """Extract normalised post dicts from a Reddit listing response.

        Parameters
        ----------
        data : dict
            Parsed JSON from a Reddit listing endpoint.

        Returns
        -------
        list[dict]
            Each dict contains: title, selftext, upvote_ratio, score,
            num_comments, permalink, subreddit.
        """
        posts: list[dict] = []
        try:
            children = data.get("data", {}).get("children", [])
            for child in children:
                post_data = child.get("data", {})
                posts.append(
                    {
                        "title": post_data.get("title", ""),
                        "selftext": post_data.get("selftext", ""),
                        "upvote_ratio": float(
                            post_data.get("upvote_ratio", 0.5)
                        ),
                        "score": int(post_data.get("score", 0)),
                        "num_comments": int(
                            post_data.get("num_comments", 0)
                        ),
                        "permalink": post_data.get("permalink", ""),
                        "subreddit": post_data.get("subreddit", ""),
                    }
                )
        except (AttributeError, TypeError, KeyError) as e:
            logger.debug(
                f"CrowdSentimentCollector: Error extracting posts: {e}"
            )
        return posts

    # ------------------------------------------------------------------ #
    #  Aggregation across posts                                           #
    # ------------------------------------------------------------------ #

    def _aggregate_posts(
        self, posts: list[dict]
    ) -> dict[str, dict]:
        """Walk every post and accumulate sentiment statistics per match pairing.

        Match pairings are identified by finding two team names in the same
        post title or body (e.g. "Arsenal vs Chelsea pick thread").

        Parameters
        ----------
        posts : list[dict]
            Normalised post dicts from _extract_posts_from_listing().

        Returns
        -------
        dict[str, dict]
            Keyed by canonical match_id ("TeamA_vs_TeamB"), values are
            partially-computed sentiment rows ready for pd.DataFrame.
        """
        # Accumulator keyed by match_id
        accum: dict[str, dict] = {}
        upvote_ratios: dict[str, list[float]] = defaultdict(list)
        post_counts: dict[str, int] = defaultdict(int)

        for post in posts:
            full_text = f"{post['title']} {post['selftext']}".lower()
            match_pairs = self._detect_match_pairs(full_text)

            if not match_pairs:
                continue

            picks = self._extract_picks(full_text)
            pos, neg = self._score_sentiment_keywords(full_text)

            for home_team, away_team in match_pairs:
                match_id = f"{home_team}_vs_{away_team}"

                if match_id not in accum:
                    accum[match_id] = {
                        "match_id": match_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "reddit_home_mentions": 0,
                        "reddit_away_mentions": 0,
                        "_home_pos": 0,
                        "_home_neg": 0,
                        "_away_pos": 0,
                        "_away_neg": 0,
                        "_pick_counts": defaultdict(int),
                    }

                row = accum[match_id]

                # Count team mentions in this post
                home_mentions = full_text.count(home_team.lower())
                away_mentions = full_text.count(away_team.lower())
                row["reddit_home_mentions"] += home_mentions
                row["reddit_away_mentions"] += away_mentions

                # Attribute sentiment keywords proportionally to mention counts
                total_mentions = home_mentions + away_mentions
                if total_mentions > 0:
                    home_share = home_mentions / total_mentions
                    away_share = away_mentions / total_mentions
                    row["_home_pos"] += pos * home_share
                    row["_home_neg"] += neg * home_share
                    row["_away_pos"] += pos * away_share
                    row["_away_neg"] += neg * away_share

                for pick in picks:
                    row["_pick_counts"][pick] += 1

                upvote_ratios[match_id].append(post["upvote_ratio"])
                post_counts[match_id] += 1

        # Finalise each match row
        result: dict[str, dict] = {}
        for match_id, row in accum.items():
            result[match_id] = self._finalise_match_row(
                row,
                upvote_ratios[match_id],
                post_counts[match_id],
            )

        return result

    @staticmethod
    def _finalise_match_row(
        row: dict, upvote_ratios: list[float], post_count: int
    ) -> dict:
        """Compute derived fields and strip accumulator-only keys.

        Parameters
        ----------
        row : dict
            Raw accumulator dict for one match_id.
        upvote_ratios : list[float]
            Upvote ratios from all posts referencing this match.
        post_count : int
            Total posts referencing this match.

        Returns
        -------
        dict
            Clean output dict with all sentinel keys removed.
        """
        home_pos = row["_home_pos"]
        home_neg = row["_home_neg"]
        away_pos = row["_away_pos"]
        away_neg = row["_away_neg"]

        home_mentions = row["reddit_home_mentions"] or 1  # avoid /0
        away_mentions = row["reddit_away_mentions"] or 1

        # Sentiment in [-1, 1]; shift to [0, 1] for output
        home_raw_sentiment = (home_pos - home_neg) / home_mentions
        away_raw_sentiment = (away_pos - away_neg) / away_mentions
        reddit_home_sentiment = max(0.0, min(1.0, (home_raw_sentiment + 1) / 2))
        reddit_away_sentiment = max(0.0, min(1.0, (away_raw_sentiment + 1) / 2))

        # Consensus pick — most-mentioned outcome
        pick_counts: dict[str, int] = row["_pick_counts"]
        reddit_consensus_pick: Optional[str] = None
        reddit_confidence: Optional[float] = None
        if pick_counts:
            total_picks = sum(pick_counts.values())
            reddit_consensus_pick = max(pick_counts, key=pick_counts.__getitem__)
            reddit_confidence = pick_counts[reddit_consensus_pick] / total_picks

        # Contrarian signal — fade the consensus
        _CONTRARIAN_MAP = {
            "home": "away",
            "away": "home",
            "draw": "home",  # fade draw → backing a decisive result
            "over": "under",
            "under": "over",
            "btts_yes": "btts_no",
            "btts_no": "btts_yes",
        }
        reddit_contrarian_pick: Optional[str] = (
            _CONTRARIAN_MAP.get(reddit_consensus_pick, None)
            if reddit_consensus_pick
            else None
        )

        avg_upvote_ratio = (
            sum(upvote_ratios) / len(upvote_ratios) if upvote_ratios else None
        )

        return {
            "match_id": row["match_id"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "reddit_home_mentions": row["reddit_home_mentions"],
            "reddit_away_mentions": row["reddit_away_mentions"],
            "reddit_home_sentiment": round(reddit_home_sentiment, 4),
            "reddit_away_sentiment": round(reddit_away_sentiment, 4),
            "reddit_consensus_pick": reddit_consensus_pick,
            "reddit_confidence": (
                round(reddit_confidence, 4) if reddit_confidence is not None else None
            ),
            "reddit_contrarian_pick": reddit_contrarian_pick,
            "post_count": post_count,
            "avg_upvote_ratio": (
                round(avg_upvote_ratio, 4) if avg_upvote_ratio is not None else None
            ),
        }

    # ------------------------------------------------------------------ #
    #  Text analysis helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_match_pairs(text: str) -> list[tuple[str, str]]:
        """Detect home/away team pairs from a post's text.

        Looks for patterns like:
          "Arsenal vs Chelsea"
          "Arsenal - Chelsea"
          "Arsenal v Chelsea"

        Parameters
        ----------
        text : str
            Lowercased combined post title + selftext.

        Returns
        -------
        list[tuple[str, str]]
            List of (home_team, away_team) string pairs found.
        """
        pattern = re.compile(
            r"([a-z][a-z\s\.\-']+?)"     # home team (1+ words)
            r"\s+(?:vs\.?|v\.?|-)\s+"    # separator: vs / v / -
            r"([a-z][a-z\s\.\-']+)",     # away team
            re.IGNORECASE,
        )
        pairs: list[tuple[str, str]] = []
        for m in pattern.finditer(text):
            home = m.group(1).strip().title()
            away = m.group(2).strip().title()
            # Filter out very short or clearly non-team strings
            if len(home) >= 3 and len(away) >= 3:
                # Truncate at 30 chars to avoid sentence fragments
                home = home[:30].strip()
                away = away[:30].strip()
                pairs.append((home, away))
        return pairs

    @staticmethod
    def _extract_picks(text: str) -> list[str]:
        """Extract canonical betting pick labels from text.

        Matches multi-word and single-word pick keywords defined in
        _PICK_KEYWORDS (longest match first to avoid partial overlaps).

        Parameters
        ----------
        text : str
            Lowercased post text.

        Returns
        -------
        list[str]
            List of canonical pick labels found (may contain duplicates
            if the same pick is mentioned multiple times).
        """
        picks: list[str] = []
        # Sort by length descending so longer phrases match before sub-words
        for keyword, label in sorted(
            _PICK_KEYWORDS.items(), key=lambda kv: -len(kv[0])
        ):
            count = text.count(keyword.lower())
            picks.extend([label] * count)
        return picks

    @staticmethod
    def _score_sentiment_keywords(text: str) -> tuple[int, int]:
        """Count positive and negative sentiment keyword hits in text.

        Parameters
        ----------
        text : str
            Lowercased post text.

        Returns
        -------
        tuple[int, int]
            (positive_count, negative_count)
        """
        tokens = re.findall(r"\b\w+\b", text)
        positive = sum(1 for t in tokens if t in _POSITIVE_WORDS)
        negative = sum(1 for t in tokens if t in _NEGATIVE_WORDS)

        # Also check bigrams for multi-word phrases
        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
        positive += sum(1 for bg in bigrams if bg in _POSITIVE_WORDS)
        negative += sum(1 for bg in bigrams if bg in _NEGATIVE_WORDS)

        return positive, negative
