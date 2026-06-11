"""
predictor/wc_team_ratings.py

Static FIFA-ranking-based strength ratings for WC 2026 participants.
Used as a guaranteed last-resort fallback when all external data sources
(football-data.org, FBref, ESPN results) are unavailable.

Format: team_name -> (goals_scored_pg, goals_conceded_pg, virtual_games)
virtual_games drives Bayesian shrinkage weight.
30 = high confidence in the rating (equivalent to ~30 games of evidence),
so top-tier teams retain their advantage over bottom-tier teams.
"""

WC_STRENGTH: dict[str, tuple[float, float, int]] = {
    # ── Tier 1: FIFA top 5 ────────────────────────────────────────────────
    "Argentina":              (1.95, 0.80, 30),
    "France":                 (1.85, 0.82, 30),
    "England":                (1.80, 0.85, 30),
    "Brazil":                 (1.78, 0.82, 30),
    "Spain":                  (1.75, 0.80, 30),

    # ── Tier 2: FIFA 6-15 ─────────────────────────────────────────────────
    "Portugal":               (1.80, 0.88, 30),
    "Netherlands":            (1.75, 0.88, 30),
    "Belgium":                (1.72, 0.90, 30),
    "Germany":                (1.75, 0.92, 30),
    "Italy":                  (1.65, 0.88, 30),
    "Croatia":                (1.55, 0.95, 30),
    "Denmark":                (1.55, 0.95, 30),
    "Switzerland":            (1.50, 0.95, 30),
    "United States":          (1.55, 1.00, 30),
    "USA":                    (1.55, 1.00, 30),
    "Morocco":                (1.40, 0.90, 30),

    # ── Tier 3: FIFA 16-30 ────────────────────────────────────────────────
    "Mexico":                 (1.55, 1.05, 30),
    "Uruguay":                (1.52, 1.02, 30),
    "Colombia":               (1.50, 1.05, 30),
    "Senegal":                (1.40, 1.00, 30),
    "Japan":                  (1.45, 1.00, 30),
    "Korea Republic":         (1.40, 1.05, 30),
    "South Korea":            (1.40, 1.05, 30),
    "Republic of Korea":      (1.40, 1.05, 30),
    "Canada":                 (1.45, 1.05, 30),
    "Turkey":                 (1.50, 1.08, 30),
    "Austria":                (1.48, 1.05, 30),
    "Ecuador":                (1.38, 1.08, 30),
    "Australia":              (1.32, 1.10, 30),
    "Ukraine":                (1.45, 1.05, 30),
    "Sweden":                 (1.42, 1.05, 30),
    "Chile":                  (1.35, 1.15, 30),
    "Norway":                 (1.45, 1.05, 30),

    # ── Tier 4: FIFA 31-50 ────────────────────────────────────────────────
    "Serbia":                 (1.45, 1.12, 30),
    "Scotland":               (1.40, 1.10, 30),
    "Poland":                 (1.38, 1.12, 30),
    "Iran":                   (1.28, 1.12, 30),
    "IR Iran":                (1.28, 1.12, 30),
    "Saudi Arabia":           (1.30, 1.12, 30),
    "Nigeria":                (1.32, 1.12, 30),
    "Egypt":                  (1.30, 1.10, 30),
    "Tunisia":                (1.25, 1.12, 30),
    "Algeria":                (1.28, 1.12, 30),
    "Ivory Coast":            (1.30, 1.12, 30),
    "Côte d'Ivoire":          (1.30, 1.12, 30),
    "Hungary":                (1.35, 1.12, 30),
    "Slovakia":               (1.32, 1.12, 30),
    "Romania":                (1.30, 1.15, 30),
    "Iraq":                   (1.28, 1.12, 30),
    "Jordan":                 (1.22, 1.12, 30),
    "Uzbekistan":             (1.25, 1.12, 30),
    "Georgia":                (1.28, 1.15, 30),
    "Slovenia":               (1.30, 1.15, 30),
    "Albania":                (1.25, 1.18, 30),
    "Peru":                   (1.28, 1.18, 30),
    "Paraguay":               (1.22, 1.18, 30),
    "Venezuela":              (1.22, 1.20, 30),

    # ── Tier 5: FIFA 51-80 ────────────────────────────────────────────────
    "Ghana":                  (1.25, 1.18, 30),
    "Cameroon":               (1.25, 1.18, 30),
    "Panama":                 (1.18, 1.22, 30),
    "Jamaica":                (1.15, 1.22, 30),
    "Honduras":               (1.15, 1.22, 30),
    "Costa Rica":             (1.20, 1.20, 30),
    "Qatar":                  (1.18, 1.22, 30),
    "Mali":                   (1.22, 1.18, 30),
    "DR Congo":               (1.22, 1.18, 30),
    "Cape Verde":             (1.20, 1.18, 30),
    "New Zealand":            (1.12, 1.28, 30),
    "Czech Republic":         (1.38, 1.12, 30),

    # ── Tier 6: FIFA 80+ ──────────────────────────────────────────────────
    "Bolivia":                (1.10, 1.32, 30),
    "El Salvador":            (1.08, 1.30, 30),
    "Cuba":                   (1.05, 1.35, 30),
    "Haiti":                  (1.08, 1.30, 30),
    "Trinidad and Tobago":    (1.10, 1.28, 30),
    "Guatemala":              (1.08, 1.30, 30),
    "Suriname":               (1.05, 1.32, 30),
    "Guyana":                 (1.05, 1.35, 30),
}
