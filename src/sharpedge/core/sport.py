"""Sport-agnostic configuration and registry."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Market:
    """A betting market type."""

    name: str
    outcomes: tuple[str, ...]  # tuple not list — immutable
    description: str = ""


@dataclass
class SportConfig:
    """Configuration for a sport module."""

    name: str  # "Football", "Tennis", "Basketball"
    slug: str  # "football", "tennis", "basketball"
    markets: list[Market] = field(default_factory=list)
    min_train_seasons: int = 3
    default_mc_sims: int = 5000
    default_min_consensus: int = 3


class SportRegistry:
    """Global registry of configured sports."""

    def __init__(self) -> None:
        self._sports: dict[str, SportConfig] = {}

    def register(self, config: SportConfig) -> None:
        if config.slug in self._sports:
            raise ValueError(f"Sport '{config.slug}' already registered")
        self._sports[config.slug] = config

    def get(self, slug: str) -> SportConfig:
        if slug not in self._sports:
            raise KeyError(f"Sport '{slug}' not registered")
        return self._sports[slug]

    def list_sports(self) -> list[str]:
        return list(self._sports.keys())

    def is_registered(self, slug: str) -> bool:
        return slug in self._sports


# Singleton
sport_registry = SportRegistry()
