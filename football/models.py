"""
models.py — Pydantic data models for football match data.
"""

from __future__ import annotations
from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class TeamForm(BaseModel):
    """Last N match results for a team."""
    team_name: str
    results: list[str] = []          # e.g. ["W", "W", "D", "L", "W"]
    goals_scored: list[int] = []
    goals_conceded: list[int] = []
    home_record: str = ""            # e.g. "5W-3D-2L"
    away_record: str = ""

    @property
    def points_last5(self) -> int:
        pts = {"W": 3, "D": 1, "L": 0}
        return sum(pts.get(r, 0) for r in self.results[-5:])

    @property
    def form_string(self) -> str:
        return " ".join(self.results[-5:])


class TeamStats(BaseModel):
    """Per-season aggregated statistics."""
    team_name: str
    season: str = ""
    competition: str = ""

    # Averages per game
    goals_scored_pg: float = 0.0
    goals_conceded_pg: float = 0.0
    xg_pg: float = 0.0             # Expected goals for
    xga_pg: float = 0.0            # Expected goals against
    shots_pg: float = 0.0
    shots_on_target_pg: float = 0.0
    corners_pg: float = 0.0
    yellow_cards_pg: float = 0.0
    red_cards_pg: float = 0.0
    fouls_committed_pg: float = 0.0

    # Season totals
    games_played: int = 0
    clean_sheets: int = 0
    btts_count: int = 0             # Both Teams To Score

    @property
    def btts_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return round(self.btts_count / self.games_played, 2)

    @property
    def clean_sheet_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return round(self.clean_sheets / self.games_played, 2)


class HeadToHead(BaseModel):
    """Historical H2H record between two teams."""
    home_team: str
    away_team: str
    meetings: list[dict] = []       # list of {date, home_score, away_score, competition}
    home_wins: int = 0
    draws: int = 0
    away_wins: int = 0
    avg_goals: float = 0.0
    avg_corners: float = 0.0
    avg_yellow_cards: float = 0.0

    @property
    def total_meetings(self) -> int:
        return self.home_wins + self.draws + self.away_wins


class InjuryReport(BaseModel):
    """Injury / suspension report for a team."""
    team_name: str
    unavailable: list[str] = []    # Confirmed out
    doubtful: list[str] = []       # Fitness doubts


class Standing(BaseModel):
    """League table row."""
    position: int
    team_name: str
    team_id: int
    played: int
    won: int
    drawn: int
    lost: int
    goals_for: int
    goals_against: int
    goal_difference: int
    points: int
    form: str = ""

    @field_validator("form", mode="before")
    @classmethod
    def coerce_form(cls, v):
        return v or ""


class Fixture(BaseModel):
    """An upcoming or past match."""
    fixture_id: int
    competition: str                # "Serie A" | "Champions League"
    competition_code: str           # "SA" | "CL"
    home_team: str
    home_team_id: int
    away_team: str
    away_team_id: int
    match_date: datetime
    status: str = "SCHEDULED"      # SCHEDULED | IN_PLAY | FINISHED
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    matchday: Optional[int] = None
    stage: Optional[str] = None    # for UCL: GROUP_STAGE, KNOCKOUT, etc.


class MatchReport(BaseModel):
    """Aggregated data package for a single match — fed to seed generator."""
    fixture: Fixture
    home_form: TeamForm
    away_form: TeamForm
    home_stats: TeamStats
    away_stats: TeamStats
    home_standing: Optional[Standing] = None
    away_standing: Optional[Standing] = None
    head_to_head: Optional[HeadToHead] = None
    home_injuries: InjuryReport = Field(default_factory=lambda: InjuryReport(team_name=""))
    away_injuries: InjuryReport = Field(default_factory=lambda: InjuryReport(team_name=""))
