"""
football/team_names.py — Team-name matching shared by every data source.

football-data.org, ESPN and The Odds API all spell clubs differently
("FC Internazionale Milano" / "Internazionale" / "Inter Milan"). Every client
used to carry its own copy of a keyword-intersection heuristic; this module is
the single implementation.
"""

from __future__ import annotations

import re

_STOPWORDS = {
    "fc", "ac", "as", "ss", "sc", "cf", "afc", "us", "usc", "ssc", "acf",
    "united", "city", "sport", "sporting", "club", "calcio", "the", "de",
    "1.", "1907", "1909", "1913",
}

# Same club, different spelling depending on the provider.
_ALIASES = {
    "internazionale": "inter",
    "juve": "juventus",
    "hellas": "verona",
}

# Words that identify one club so strongly that their presence on one side
# and absence on the other rules the match out, even if other words overlap
# (the Milan derby: "Inter Milan" vs "AC Milan" share "milan").
_EXCLUSIVE = {"inter"}


def keywords(name: str) -> set[str]:
    """Normalised identifying words of a team name (aliases applied)."""
    words = re.sub(r"[^a-z0-9\s]", " ", name.lower()).split()
    return {_ALIASES.get(w, w) for w in words if w not in _STOPWORDS and len(w) > 2}


def match_score(a: str, b: str) -> int:
    """Number of shared identifying keywords; 0 means no match."""
    if not a or not b:
        return 0
    if a.strip().lower() == b.strip().lower():
        return 99
    ka, kb = keywords(a), keywords(b)
    for word in _EXCLUSIVE:
        if (word in ka) != (word in kb):
            return 0
    return len(ka & kb)


def teams_match(a: str, b: str) -> bool:
    """True when the two names share at least one identifying keyword."""
    return match_score(a, b) > 0
