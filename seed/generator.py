"""
seed/generator.py — Converts a MatchReport into a MiroFish seed document.

The seed document is a rich narrative report that MiroFish uses to:
  1. Build a knowledge graph (entities: teams, players, stats)
  2. Generate agent profiles (football analysts, journalists, fans)
  3. Run multi-agent simulation (analysts debating the match)
  4. Produce a structured prediction report

The report is deliberately framed as "content for public debate" so that
OASIS's social simulation engine (built for Twitter/Reddit opinion modelling)
naturally produces analytical discussion about match outcomes.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Optional

from football.models import MatchReport, TeamStats, TeamForm

logger = logging.getLogger(__name__)


class SeedGenerator:
    """Generates MiroFish-compatible seed documents from match data."""

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, report: MatchReport) -> str:
        """
        Build a complete seed document for the given match.
        Returns a markdown string ready to be uploaded to MiroFish.
        """
        f = report.fixture
        sections = [
            self._header(report),
            self._competition_context(report),
            self._team_profile(
                f.home_team, report.home_form, report.home_stats,
                report.home_standing, report.home_injuries, is_home=True,
            ),
            self._team_profile(
                f.away_team, report.away_form, report.away_stats,
                report.away_standing, report.away_injuries, is_home=False,
            ),
            self._head_to_head(report),
            self._statistical_matchup(report),
            self._stakes_and_context(report),
            self._simulation_instructions(report),
        ]
        return "\n\n".join(s for s in sections if s.strip())

    def generate_prediction_prompt(self, report: MatchReport) -> str:
        """
        Separate short prompt that tells MiroFish WHAT to predict.
        This is the "prediction requirement" field in the MiroFish UI.
        """
        f = report.fixture
        return (
            f"Simulate expert football analysts and fans debating the outcome of "
            f"{f.home_team} vs {f.away_team} ({f.competition}, "
            f"{f.match_date.strftime('%d %B %Y')}). "
            f"Use all statistical evidence to build consensus predictions. "
            f"In the final report, include a clearly labelled "
            f"'BETTING PREDICTIONS' section with probability percentages for: "
            f"match result (1X2), Over/Under 2.5 goals, Over/Under 3.5 goals, "
            f"Both Teams To Score, Over/Under 9.5 corners, Over/Under 3.5 yellow cards, "
            f"red card probability, and the most likely scoreline. "
            f"Format the predictions as a JSON block inside the report."
        )

    # ── Document sections ─────────────────────────────────────────────────────

    def _header(self, report: MatchReport) -> str:
        f = report.fixture
        date_str = f.match_date.strftime("%A, %d %B %Y — %H:%M UTC")
        return (
            f"# MATCH ANALYSIS REPORT\n"
            f"## {f.home_team} vs {f.away_team}\n"
            f"**Competition:** {f.competition}  \n"
            f"**Date:** {date_str}  \n"
            f"**Matchday/Stage:** {f.matchday or f.stage or 'TBC'}  \n"
        )

    def _competition_context(self, report: MatchReport) -> str:
        f = report.fixture
        hs = report.home_standing
        as_ = report.away_standing

        if f.competition_code == "SA":
            comp_desc = (
                "**Serie A** is the top professional football league in Italy. "
                "Teams play 38 matches in a season. The top 4 qualify for the Champions League."
            )
        elif f.competition_code == "CL":
            comp_desc = (
                "**UEFA Champions League** is Europe's premier club football competition. "
                "The current stage determines which teams advance to the knockout rounds."
            )
        elif f.competition_code == "ECL":
            comp_desc = (
                "**UEFA Europa Conference League** is UEFA's third-tier club competition. "
                "It features clubs from across Europe competing in a knockout format, "
                "with the current stage determining which teams advance."
            )
        elif f.competition_code == "WC":
            comp_desc = (
                "**FIFA World Cup** is the most prestigious international football tournament. "
                "National teams compete for the world championship every four years."
            )
        elif f.competition_code in ("WCQE", "WCQA", "WCQC", "WCQAS", "WCQAF"):
            region_map = {
                "WCQE": "European", "WCQA": "South American",
                "WCQC": "CONCACAF", "WCQAS": "Asian", "WCQAF": "African",
            }
            region = region_map.get(f.competition_code, "")
            comp_desc = (
                f"**{region} World Cup Qualifying** — national teams competing for a place "
                f"at the FIFA World Cup 2026. Every point is crucial as only a limited number "
                f"of spots are available from this confederation."
            )
        else:
            comp_desc = f"**{f.competition}** — an international football competition."

        standing_info = ""
        if hs and as_:
            standing_info = (
                f"\n\n### Current Standings\n"
                f"| Team | Pos | Pts | W | D | L | GF | GA | GD | Form |\n"
                f"|------|-----|-----|---|---|---|----|----|----|-----------|\n"
                f"| {f.home_team} | {hs.position} | {hs.points} | "
                f"{hs.won} | {hs.drawn} | {hs.lost} | "
                f"{hs.goals_for} | {hs.goals_against} | {hs.goal_difference} | "
                f"{hs.form or 'N/A'} |\n"
                f"| {f.away_team} | {as_.position} | {as_.points} | "
                f"{as_.won} | {as_.drawn} | {as_.lost} | "
                f"{as_.goals_for} | {as_.goals_against} | {as_.goal_difference} | "
                f"{as_.form or 'N/A'} |\n"
            )

        return f"## COMPETITION CONTEXT\n{comp_desc}{standing_info}"

    def _team_profile(self, team_name: str, form: TeamForm, stats: TeamStats,
                       standing, injuries, is_home: bool) -> str:
        loc = "HOME" if is_home else "AWAY"
        venue_record = form.home_record if is_home else form.away_record

        form_details = []
        for r, gs, gc in zip(
            form.results[-5:], form.goals_scored[-5:], form.goals_conceded[-5:]
        ):
            form_details.append(f"{r} ({gs}-{gc})")
        form_str = ", ".join(form_details) if form_details else "N/A"

        avg_gs = (
            round(sum(form.goals_scored[-5:]) / len(form.goals_scored[-5:]), 1)
            if form.goals_scored else 0.0
        )
        avg_gc = (
            round(sum(form.goals_conceded[-5:]) / len(form.goals_conceded[-5:]), 1)
            if form.goals_conceded else 0.0
        )

        inj_lines = []
        if injuries.unavailable:
            inj_lines.append(f"  - **Out:** {', '.join(injuries.unavailable)}")
        if injuries.doubtful:
            inj_lines.append(f"  - **Doubtful:** {', '.join(injuries.doubtful)}")
        inj_section = "\n".join(inj_lines) if inj_lines else "  - No confirmed absences"

        btts_pct = f"{stats.btts_rate * 100:.0f}%" if stats.btts_rate else "N/A"
        cs_pct = f"{stats.clean_sheet_rate * 100:.0f}%" if stats.clean_sheet_rate else "N/A"

        return (
            f"## {loc} TEAM: {team_name.upper()}\n\n"
            f"### Recent Form (Last 5)\n"
            f"**Results:** {form_str}  \n"
            f"**Points:** {form.points_last5}/15  \n"
            f"**Avg goals scored (last 5):** {avg_gs}  \n"
            f"**Avg goals conceded (last 5):** {avg_gc}  \n"
            f"**{'Home' if is_home else 'Away'} record this season:** {venue_record}\n\n"
            f"### Season Statistics (per game)\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Goals scored | **{stats.goals_scored_pg}** |\n"
            f"| Goals conceded | **{stats.goals_conceded_pg}** |\n"
            f"| xG (expected goals) | **{stats.xg_pg}** |\n"
            f"| xGA (expected conceded) | **{stats.xga_pg}** |\n"
            f"| Shots | **{stats.shots_pg}** |\n"
            f"| Shots on target | **{stats.shots_on_target_pg}** |\n"
            f"| Corners | **{stats.corners_pg}** |\n"
            f"| Yellow cards | **{stats.yellow_cards_pg}** |\n"
            f"| Red cards | **{stats.red_cards_pg}** |\n"
            f"| Fouls committed | **{stats.fouls_committed_pg}** |\n"
            f"| Clean sheet rate | **{cs_pct}** |\n"
            f"| BTTS rate | **{btts_pct}** |\n\n"
            f"### Injury & Suspension Report\n{inj_section}\n"
        )

    def _head_to_head(self, report: MatchReport) -> str:
        h2h = report.head_to_head
        if not h2h or h2h.total_meetings == 0:
            return "## HEAD-TO-HEAD\nNo historical meetings found in database."

        f = report.fixture
        lines = [
            f"## HEAD-TO-HEAD: {f.home_team} vs {f.away_team}\n",
            f"**Total meetings:** {h2h.total_meetings}  ",
            f"**{f.home_team} wins:** {h2h.home_wins}  ",
            f"**Draws:** {h2h.draws}  ",
            f"**{f.away_team} wins:** {h2h.away_wins}  ",
            f"**Average goals per game:** {h2h.avg_goals}  ",
        ]

        if h2h.avg_corners:
            lines.append(f"**Average corners per game:** {h2h.avg_corners}  ")
        if h2h.avg_yellow_cards:
            lines.append(f"**Average yellow cards per game:** {h2h.avg_yellow_cards}  ")

        if h2h.meetings:
            lines.append("\n### Recent Meetings")
            lines.append("| Date | Home | Score | Away | Competition |")
            lines.append("|------|------|-------|------|-------------|")
            for m in h2h.meetings[-5:]:
                lines.append(
                    f"| {m.get('date','')} | {m.get('home_team','')} | "
                    f"**{m.get('home_score','')}–{m.get('away_score','')}** | "
                    f"{m.get('away_team','')} | {m.get('competition','')} |"
                )

        return "\n".join(lines)

    def _statistical_matchup(self, report: MatchReport) -> str:
        hs = report.home_stats
        as_ = report.away_stats
        f = report.fixture

        LEAGUE_CORNERS_PG = {
            "SA": 5.1, "CL": 4.8, "ECL": 4.6,
            "WC": 4.4, "WCQE": 4.5, "WCQA": 4.3,
            "WCQC": 4.2, "WCQAS": 4.2, "WCQAF": 4.0,
        }
        LEAGUE_CARDS_PG = {
            "SA": 2.2, "CL": 1.8, "ECL": 2.0,
            "WC": 2.0, "WCQE": 2.3, "WCQA": 2.4,
            "WCQC": 2.3, "WCQAS": 2.2, "WCQAF": 2.5,
        }
        avg_corner = LEAGUE_CORNERS_PG.get(f.competition_code, 4.8)
        avg_card = LEAGUE_CARDS_PG.get(f.competition_code, 2.0)

        home_corners = hs.corners_pg if hs.corners_pg > 0 else avg_corner
        away_corners = as_.corners_pg if as_.corners_pg > 0 else avg_corner
        home_cards = hs.yellow_cards_pg if hs.yellow_cards_pg > 0 else avg_card
        away_cards = as_.yellow_cards_pg if as_.yellow_cards_pg > 0 else avg_card

        projected_corners = round(home_corners + away_corners, 1)
        projected_goals = round(hs.xg_pg + as_.xg_pg, 1)
        projected_cards = round(home_cards + away_cards, 1)

        corners_data_note = (
            "" if (hs.corners_pg > 0 and as_.corners_pg > 0)
            else " *(league avg used — FBref data unavailable)*"
        )
        cards_data_note = (
            "" if (hs.yellow_cards_pg > 0 and as_.yellow_cards_pg > 0)
            else " *(league avg used)*"
        )

        if projected_corners >= 11.0:
            corner_signal = "**→ High-corner match: strongly suggests OVER 9.5 corners**"
        elif projected_corners >= 10.0:
            corner_signal = "**→ Above average: leans OVER 9.5 corners**"
        elif projected_corners >= 9.0:
            corner_signal = "**→ Around the line: 9.5 corners is a close call**"
        elif projected_corners >= 8.0:
            corner_signal = "**→ Below average: leans UNDER 9.5 corners**"
        else:
            corner_signal = "**→ Low-corner match: strongly suggests UNDER 9.5 corners**"

        if projected_cards >= 4.0:
            cards_signal = "**→ High-card match: leans OVER 3.5 cards**"
        elif projected_cards >= 3.0:
            cards_signal = "**→ Around the line: 3.5 cards is a close call**"
        else:
            cards_signal = "**→ Low-card match: leans UNDER 3.5 cards**"

        return (
            f"## STATISTICAL MATCHUP\n\n"
            f"| Metric | {f.home_team} | {f.away_team} |\n"
            f"|--------|{'':>{len(f.home_team)}}|{'':>{len(f.away_team)}}|\n"
            f"| Goals scored/game | {hs.goals_scored_pg} | {as_.goals_scored_pg} |\n"
            f"| Goals conceded/game | {hs.goals_conceded_pg} | {as_.goals_conceded_pg} |\n"
            f"| xG/game | {hs.xg_pg} | {as_.xg_pg} |\n"
            f"| xGA/game | {hs.xga_pg} | {as_.xga_pg} |\n"
            f"| Shots/game | {hs.shots_pg} | {as_.shots_pg} |\n"
            f"| Corners/game | {home_corners} | {away_corners} |\n"
            f"| Yellow cards/game | {home_cards} | {away_cards} |\n"
            f"| Red cards/game | {hs.red_cards_pg} | {as_.red_cards_pg} |\n"
            f"| BTTS rate | {hs.btts_rate:.0%} | {as_.btts_rate:.0%} |\n"
            f"| Clean sheet rate | {hs.clean_sheet_rate:.0%} | {as_.clean_sheet_rate:.0%} |\n\n"
            f"**Projected match total corners:** ~{projected_corners}{corners_data_note}  \n"
            f"{corner_signal}  \n"
            f"**Projected match xG (combined):** ~{projected_goals}  \n"
            f"**Projected yellow cards:** ~{projected_cards}{cards_data_note}  \n"
            f"{cards_signal}  \n"
        )

    def _stakes_and_context(self, report: MatchReport) -> str:
        f = report.fixture
        hs = report.home_standing
        as_ = report.away_standing

        lines = ["## STAKES AND MATCH CONTEXT\n"]

        if hs and as_:
            pts_diff = abs(hs.points - as_.points)
            if hs.position <= 4 or as_.position <= 4:
                lines.append(
                    "This match has **Champions League qualification implications**. "
                    "Both teams will be highly motivated."
                )
            if hs.position >= 17 or as_.position >= 17:
                lines.append(
                    "**Relegation battle:** One or both teams are in danger zone. "
                    "Expect a physical, high-pressure match."
                )
            if pts_diff <= 3:
                lines.append(
                    f"The teams are separated by only {pts_diff} points — "
                    "this is a closely matched encounter."
                )

        if f.competition_code == "CL":
            lines.append(
                "Champions League matches often feature more tactical caution, "
                "especially in away fixtures. Home advantage is significant."
            )
        elif f.competition_code == "ECL":
            lines.append(
                "Conference League matches can be unpredictable — many clubs treat "
                "this competition as a genuine route to European glory. "
                "Expect committed performances from both sides."
            )

        if not lines[1:]:
            lines.append(
                f"A competitive {f.competition} fixture between two sides "
                "with motivation to perform."
            )

        return "\n".join(lines)

    def _simulation_instructions(self, report: MatchReport) -> str:
        f = report.fixture

        return (
            f"## SIMULATION INSTRUCTIONS\n\n"
            f"This document is seed material for a multi-agent AI simulation.\n\n"
            f"**Scenario:** A panel of football analysts, journalists, and experienced "
            f"bettors are debating this match across social media in the 48 hours before "
            f"kick-off. They draw on the statistics above, historical patterns, and their "
            f"own expertise to form opinions.\n\n"
            f"**Agent profiles to generate:**\n"
            f"- Football statistician (focus: xG, expected value, data-driven)\n"
            f"- Tactics journalist (focus: formations, pressing, key duels)\n\n"
            f"- Veteran {f.competition} betting analyst (focus: value odds, market trends)\n"
            f"- {f.home_team} supporter (optimistic home bias)\n"
            f"- {f.away_team} supporter (optimistic away bias)\n"
            f"- Neutral football pundit (balanced, historical perspective)\n\n"
            f"**After simulation, the Report Agent MUST produce a "
            f"'BETTING PREDICTIONS' section containing this exact JSON block:**\n\n"
            f"```json\n"
            f"{{\n"
            f'  "match": "{f.home_team} vs {f.away_team}",\n'
            f'  "competition": "{f.competition}",\n'
            f'  "home_win_pct": 0.0,\n'
            f'  "draw_pct": 0.0,\n'
            f'  "away_win_pct": 0.0,\n'
            f'  "over_2_5_pct": 0.0,\n'
            f'  "under_2_5_pct": 0.0,\n'
            f'  "over_3_5_pct": 0.0,\n'
            f'  "under_3_5_pct": 0.0,\n'
            f'  "btts_yes_pct": 0.0,\n'
            f'  "btts_no_pct": 0.0,\n'
            f'  "over_9_5_corners_pct": 0.0,\n'
            f'  "under_9_5_corners_pct": 0.0,\n'
            f'  "over_3_5_cards_pct": 0.0,\n'
            f'  "under_3_5_cards_pct": 0.0,\n'
            f'  "red_card_pct": 0.0,\n'
            f'  "most_likely_scoreline": "X-X",\n'
            f'  "confidence": "low|medium|high"\n'
            f"}}\n"
            f"```\n\n"
            f"Replace all `0.0` values with actual percentage probabilities (0–100) "
            f"derived from the simulation consensus. The values for each pair "
            f"(e.g. over/under) must sum to 100."
        )


# ── Convenience function ──────────────────────────────────────────────────────

def build_seed_document(report: MatchReport) -> tuple[str, str]:
    """Returns (seed_text, prediction_prompt) ready for MiroFish."""
    gen = SeedGenerator()
    seed = gen.generate(report)
    prompt = gen.generate_prediction_prompt(report)
    return seed, prompt
