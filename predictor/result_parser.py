"""
predictor/result_parser.py

Extracts structured betting predictions from a MiroFish prediction report.

The report is a markdown document produced by the ReportAgent.
We instructed the agent (via the seed document) to include a JSON block
under the heading "BETTING PREDICTIONS".
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class BettingPrediction:
    match: str = ""
    competition: str = ""

    # Match result
    home_win_pct: float = 0.0
    draw_pct: float = 0.0
    away_win_pct: float = 0.0

    # Goals
    over_2_5_pct: float = 0.0
    under_2_5_pct: float = 0.0
    over_3_5_pct: float = 0.0
    under_3_5_pct: float = 0.0

    # BTTS
    btts_yes_pct: float = 0.0
    btts_no_pct: float = 0.0

    # Corners
    over_9_5_corners_pct: float = 0.0
    under_9_5_corners_pct: float = 0.0

    # Cards
    over_3_5_cards_pct: float = 0.0
    under_3_5_cards_pct: float = 0.0
    red_card_pct: float = 0.0

    # Scoreline
    most_likely_scoreline: str = ""
    confidence: str = "medium"

    # Cards (Poisson-computed)
    over_4_5_yellow_pct: float = 0.0
    under_4_5_yellow_pct: float = 0.0
    poisson_cards_lambda: float = 0.0

    # Poisson model output (computed post-LLM)
    poisson_lambda_home: float = 0.0
    poisson_lambda_away: float = 0.0
    poisson_top_scorelines: List = field(default_factory=list)  # [["1-0", 18.2], ...]
    poisson_corners_lambda: float = 0.0

    # Meta
    raw_report: str = ""
    parse_source: str = "json"  # "json" | "text_inference" | "fallback"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "raw_report"}


class ResultParser:
    """Parses a MiroFish report markdown into a BettingPrediction."""

    def parse(self, report_markdown: str) -> BettingPrediction:
        """
        Attempt extraction in order:
          1. JSON block inside BETTING PREDICTIONS section
          2. Keyword-based text inference from the report prose
          3. Fallback with 0.0 values (indicating parse failure)
        """
        pred = BettingPrediction(raw_report=report_markdown)

        # Try JSON first
        json_data = self._extract_json_block(report_markdown)
        if json_data:
            self._populate_from_json(pred, json_data)
            pred.parse_source = "json"
            self._normalise(pred)
            return pred

        # Try text inference
        logger.info("JSON block not found — attempting text inference")
        self._infer_from_text(pred, report_markdown)
        if pred.home_win_pct or pred.over_2_5_pct:
            pred.parse_source = "text_inference"
            self._normalise(pred)
            return pred

        # Fallback
        logger.warning("Could not extract predictions from report")
        pred.parse_source = "fallback"
        return pred

    # ── JSON extraction ───────────────────────────────────────────────────────

    def _extract_json_block(self, text: str) -> Optional[dict]:
        """Find the JSON block in the BETTING PREDICTIONS section."""
        # Locate the section
        section_match = re.search(
            r"BETTING PREDICTIONS.*?```(?:json)?\s*(\{.*?\})\s*```",
            text, re.DOTALL | re.IGNORECASE,
        )
        if section_match:
            try:
                return json.loads(section_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try any JSON block in the document as fallback
        for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
            try:
                data = json.loads(m.group(1))
                if "home_win_pct" in data or "over_2_5_pct" in data:
                    return data
            except json.JSONDecodeError:
                continue

        return None

    def _populate_from_json(self, pred: BettingPrediction, data: dict) -> None:
        for field_name in pred.__dataclass_fields__:
            if field_name in ("raw_report", "parse_source"):
                continue
            if field_name in data:
                setattr(pred, field_name, data[field_name])

    # ── Text inference ────────────────────────────────────────────────────────

    def _infer_from_text(self, pred: BettingPrediction, text: str) -> None:
        """
        Heuristic extraction when JSON is absent.
        Looks for percentage figures near betting keywords.
        """
        text_lower = text.lower()

        def find_pct(pattern: str) -> float:
            m = re.search(pattern, text_lower)
            if m:
                try:
                    return float(m.group(1))
                except (IndexError, ValueError):
                    pass
            return 0.0

        pred.home_win_pct = find_pct(r"home\s+win.*?(\d+(?:\.\d+)?)\s*%")
        pred.draw_pct = find_pct(r"draw.*?(\d+(?:\.\d+)?)\s*%")
        pred.away_win_pct = find_pct(r"away\s+win.*?(\d+(?:\.\d+)?)\s*%")
        pred.over_2_5_pct = find_pct(r"over\s+2\.5.*?(\d+(?:\.\d+)?)\s*%")
        pred.btts_yes_pct = find_pct(r"both\s+teams.*?score.*?(\d+(?:\.\d+)?)\s*%")
        pred.over_9_5_corners_pct = find_pct(r"over\s+9\.5\s+corners.*?(\d+(?:\.\d+)?)\s*%")
        pred.over_3_5_cards_pct = find_pct(r"over\s+3\.5\s+(?:yellow\s+)?cards.*?(\d+(?:\.\d+)?)\s*%")
        pred.red_card_pct = find_pct(r"red\s+card.*?(\d+(?:\.\d+)?)\s*%")

        # Scoreline
        score_m = re.search(r"most\s+likely\s+scoreline[:\s]+(\d+[-–]\d+)", text_lower)
        if score_m:
            pred.most_likely_scoreline = score_m.group(1).replace("–", "-")

    # ── Normalisation ─────────────────────────────────────────────────────────

    def _normalise(self, pred: BettingPrediction) -> None:
        """Ensure paired markets sum to 100. Cap values at 100."""

        def cap(v: float) -> float:
            return max(0.0, min(100.0, v))

        def normalise_pair(a: str, b: str) -> None:
            va, vb = cap(getattr(pred, a)), cap(getattr(pred, b))
            if va == 0.0 and vb == 0.0:
                return  # no data
            total = va + vb
            if total > 0 and abs(total - 100.0) > 1.0:
                setattr(pred, a, round(va / total * 100, 1))
                setattr(pred, b, round(vb / total * 100, 1))
            else:
                setattr(pred, a, round(va, 1))
                setattr(pred, b, round(vb, 1))

        normalise_pair("over_2_5_pct", "under_2_5_pct")
        normalise_pair("over_3_5_pct", "under_3_5_pct")
        normalise_pair("btts_yes_pct", "btts_no_pct")
        normalise_pair("over_9_5_corners_pct", "under_9_5_corners_pct")
        normalise_pair("over_3_5_cards_pct", "under_3_5_cards_pct")
        normalise_pair("over_4_5_yellow_pct", "under_4_5_yellow_pct")

        # 1X2 — normalise three-way
        total_1x2 = pred.home_win_pct + pred.draw_pct + pred.away_win_pct
        if total_1x2 > 0 and abs(total_1x2 - 100.0) > 1.0:
            pred.home_win_pct = round(pred.home_win_pct / total_1x2 * 100, 1)
            pred.draw_pct = round(pred.draw_pct / total_1x2 * 100, 1)
            pred.away_win_pct = round(pred.away_win_pct / total_1x2 * 100, 1)

        pred.red_card_pct = cap(pred.red_card_pct)
