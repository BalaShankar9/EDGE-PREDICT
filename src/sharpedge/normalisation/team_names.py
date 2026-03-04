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
        # Build reverse lookup: source -> {alias_lowercase: canonical_id}
        for team_id, info in _registry.items():
            for source, alias in info.get("aliases", {}).items():
                if source not in _source_lookup:
                    _source_lookup[source] = {}
                _source_lookup[source][alias.lower()] = team_id
    return _registry


def normalise(name: str, source: str) -> Optional[str]:
    """Resolve a team name from a specific source to canonical team ID.
    Returns the canonical team ID (e.g. 'arsenal') or None if not found.
    """
    _load_registry()

    # Exact match (case-insensitive) for the given source
    lookup = _source_lookup.get(source, {})
    if name.lower() in lookup:
        return lookup[name.lower()]

    # Try across all sources
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
    """Get the canonical display name for a team ID."""
    _load_registry()
    if team_id in _registry:
        return _registry[team_id]["canonical"]
    return None


def get_alias(team_id: str, source: str) -> Optional[str]:
    """Get the alias used by a specific source for a team."""
    _load_registry()
    if team_id in _registry:
        return _registry[team_id].get("aliases", {}).get(source)
    return None
