"""
predictor/wc_team_ratings.py

Static FIFA-ranking-based strength ratings for WC 2026 participants.
Used as a guaranteed last-resort fallback when all external data sources
(football-data.org, FBref, ESPN results) are unavailable.

Format: team_name -> (goals_scored_pg, goals_conceded_pg, virtual_games)
virtual_games drives Bayesian shrinkage weight (8 = moderate confidence).
"""

WC_STRENGTH: dict[str, tuple[float, float, int]] = {
    # ── Tier 1: FIFA top 5 ────────────────────────────────────────────────
    "Argentina":              (1.95, 0.80, 8),
    "France":                 (1.85, 0.82, 8),
    "England":                (1.80, 0.85, 8),
    "Brazil":                 (1.78, 0.82, 8),
    "Spain":                  (1.75, 0.80, 8),

    # ── Tier 2: FIFA 6-15 ─────────────────────────────────────────────────
    "Portugal":               (1.80, 0.88, 8),
    "Netherlands":            (1.75, 0.88, 8),
    "Belgium":                (1.72, 0.90, 8),
    "Germany":                (1.75, 0.92, 8),
    "Italy":                  (1.65, 0.88, 8),
    "Croatia":                (1.55, 0.95, 8),
    "Denmark":                (1.55, 0.95, 8),
    "Switzerland":            (1.50, 0.95, 8),
    "United States":          (1.55, 1.00, 8),
    "USA":                    (1.55, 1.00, 8),
    "Morocco":                (1.40, 0.90, 8),

    # ── Tier 3: FIFA 16-30 ────────────────────────────────────────────────
    "Mexico":                 (1.55, 1.05, 8),
    "Uruguay":                (1.52, 1.02, 8),
    "Colombia":               (1.50, 1.05, 8),
    "Senegal":                (1.40, 1.00, 8),
    "Japan":                  (1.45, 1.00, 8),
    "Korea Republic":         (1.40, 1.05, 8),
    "South Korea":            (1.40, 1.05, 8),
    "Republic of Korea":      (1.40, 1.05, 8),
    "Canada":                 (1.45, 1.05, 8),
    "Turkey":                 (1.50, 1.08, 8),
    "Austria":                (1.48, 1.05, 8),
    "Ecuador":                (1.38, 1.08, 8),
    "Australia":              (1.32, 1.10, 8),
    "Ukraine":                (1.45, 1.05, 8),
    "Sweden":                 (1.42, 1.05, 8),
    "Chile":                  (1.35, 1.15, 8),
    "Norway":                 (1.45, 1.05, 8),

    # ── Tier 4: FIFA 31-50 ────────────────────────────────────────────────
    "Serbia":                 (1.45, 1.12, 8),
    "Scotland":               (1.40, 1.10, 8),
    "Poland":                 (1.38, 1.12, 8),
    "Iran":                   (1.28, 1.12, 8),
    "IR Iran":                (1.28, 1.12, 8),
    "Saudi Arabia":           (1.30, 1.12, 8),
    "Nigeria":                (1.32, 1.12, 8),
    "Egypt":                  (1.30, 1.10, 8),
    "Tunisia":                (1.25, 1.12, 8),
    "Algeria":                (1.28, 1.12, 8),
    "Ivory Coast":            (1.30, 1.12, 8),
    "Côte d'Ivoire":          (1.30, 1.12, 8),
    "Hungary":                (1.35, 1.12, 8),
    "Slovakia":               (1.32, 1.12, 8),
    "Romania":                (1.30, 1.15, 8),
    "Iraq":                   (1.28, 1.12, 8),
    "Jordan":                 (1.22, 1.12, 8),
    "Uzbekistan":             (1.25, 1.12, 8),
    "Georgia":                (1.28, 1.15, 8),
    "Slovenia":               (1.30, 1.15, 8),
    "Albania":                (1.25, 1.18, 8),
    "Peru":                   (1.28, 1.18, 8),
    "Paraguay":               (1.22, 1.18, 8),
    "Venezuela":              (1.22, 1.20, 8),

    # ── Tier 5: FIFA 51-80 ────────────────────────────────────────────────
    "Ghana":                  (1.25, 1.18, 8),
    "Cameroon":               (1.25, 1.18, 8),
    "Panama":                 (1.18, 1.22, 8),
    "Jamaica":                (1.15, 1.22, 8),
    "Honduras":               (1.15, 1.22, 8),
    "Costa Rica":             (1.20, 1.20, 8),
    "Qatar":                  (1.18, 1.22, 8),
    "Mali":                   (1.22, 1.18, 8),
    "DR Congo":               (1.22, 1.18, 8),
    "Cape Verde":             (1.20, 1.18, 8),
    "New Zealand":            (1.12, 1.28, 8),
    "Czech Republic":         (1.38, 1.12, 8),

    # ── Tier 6: FIFA 80+ ──────────────────────────────────────────────────
    "Bolivia":                (1.10, 1.32, 8),
    "El Salvador":            (1.08, 1.30, 8),
    "Cuba":                   (1.05, 1.35, 8),
    "Haiti":                  (1.08, 1.30, 8),
    "Trinidad and Tobago":    (1.10, 1.28, 8),
    "Guatemala":              (1.08, 1.30, 8),
    "Suriname":               (1.05, 1.32, 8),
    "Guyana":                 (1.05, 1.35, 8),
}
