"""
predictor/fifa_ranking.py — FIFA World Ranking lookup and lambda strength factor.

Static ranking table (June 2026 approximate).
Used as a strength prior for national-team competitions (WC, WCQX)
when historical xG/statistical data is limited or based on qualifiers
against weak opposition.
"""

from __future__ import annotations

# FIFA World Rankings — June 2026 (approximate)
_RANKINGS: dict[str, int] = {
    "Argentina":             1,
    "France":                2,
    "Spain":                 3,
    "England":               4,
    "Brazil":                5,
    "Portugal":              6,
    "Netherlands":           7,
    "Belgium":               8,
    "Germany":               9,
    "Italy":                10,
    "Colombia":             11,
    "Croatia":              12,
    "Morocco":              13,
    "Uruguay":              14,
    "United States":        15,
    "USA":                  15,
    "Mexico":               16,
    "Japan":                17,
    "Senegal":              18,
    "Denmark":              19,
    "Switzerland":          20,
    "Ecuador":              21,
    "Austria":              22,
    "Turkey":               23,
    "Ukraine":              24,
    "Australia":            25,
    "South Korea":          26,
    "Korea Republic":       26,
    "Republic of Korea":    26,
    "Serbia":               27,
    "Hungary":              28,
    "Iran":                 29,
    "Poland":               30,
    "Peru":                 31,
    "Chile":                32,
    "Czech Republic":       33,
    "Czechia":              33,
    "Venezuela":            34,
    "Saudi Arabia":         35,
    "Romania":              36,
    "Scotland":             37,
    "Egypt":                38,
    "Norway":               39,
    "Wales":                40,
    "Algeria":              41,
    "Paraguay":             42,
    "Slovakia":             43,
    "Greece":               44,
    "Sweden":               45,
    "Bolivia":              46,
    "Mali":                 47,
    "South Africa":         48,
    "Tunisia":              49,
    "Ivory Coast":          50,
    "Côte d'Ivoire":        50,
    "Cote d'Ivoire":        50,
    "Nigeria":              51,
    "Ghana":                52,
    "Cameroon":             53,
    "Canada":               54,
    "Costa Rica":           55,
    "Jamaica":              56,
    "Panama":               57,
    "Honduras":             58,
    "El Salvador":          59,
    "Qatar":                60,
    "Iraq":                 61,
    "United Arab Emirates": 62,
    "UAE":                  62,
    "Uzbekistan":           63,
    "New Zealand":          64,
    "Albania":              65,
    "Israel":               66,
    "Bahrain":              67,
    "DR Congo":             68,
    "Congo DR":             68,
    "Burkina Faso":         69,
    "Zambia":               70,
    "Angola":               71,
    "Rwanda":               72,
    "Kenya":                73,
    "Oman":                 74,
    "Guatemala":            75,
    "Trinidad and Tobago":  76,
    "Cuba":                 77,
    "Haiti":                78,
    "Libya":                79,
    "Mozambique":           80,
}

# Competition codes where FIFA ranking is meaningful
_WC_CODES = {"WC", "WCQE", "WCQA", "WCQC", "WCQAS", "WCQAF"}


def get_ranking(team_name: str) -> int | None:
    """Return the FIFA ranking for a national team name, or None if unknown."""
    rank = _RANKINGS.get(team_name)
    if rank is not None:
        return rank

    lower = team_name.lower().strip()
    for name, r in _RANKINGS.items():
        if name.lower() == lower:
            return r

    # Keyword overlap (e.g. "Korea Republic" matches "South Korea" via "korea")
    keywords = {w.lower() for w in team_name.split() if len(w) > 3}
    for name, r in _RANKINGS.items():
        if keywords & {w.lower() for w in name.split()}:
            return r

    return None


def ranking_lambda_factors(home_rank: int, away_rank: int) -> tuple[float, float]:
    """
    Returns (home_mult, away_mult) to adjust Poisson lambdas based on FIFA ranking gap.

    Each 20 rank positions = 1% adjustment, capped at ±12%.
    A better-ranked (lower number) home team gets a small lambda boost.
    """
    if home_rank <= 0 or away_rank <= 0:
        return 1.0, 1.0

    rank_gap = away_rank - home_rank   # positive when home team is better ranked
    raw_adj = rank_gap * 0.0005        # 1% per 20 positions
    adj = max(-0.12, min(raw_adj, 0.12))
    return round(1.0 + adj, 4), round(1.0 - adj, 4)


def is_wc_competition(competition_code: str) -> bool:
    """True for World Cup and all qualifying competitions."""
    return competition_code in _WC_CODES
