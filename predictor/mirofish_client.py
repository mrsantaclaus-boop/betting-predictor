"""
predictor/mirofish_client.py — MiroFish "mini-swarm" prediction client.

The original MiroFish runs a full social simulation (graph building, persona
agents on OASIS, a report agent). That needs a running backend, hundreds of
LLM calls per prediction and more memory than the free tiers we use. This
module keeps the spirit at zero cost:

  1. ANALYST ROUND — the six personas the seed document already describes
     (statistician, tactics journalist, betting analyst, home supporter, away
     supporter, neutral pundit) each read the match dossier and answer with
     a short JSON thesis. Calls are spread over the models listed in
     SWARM_PERSONA_MODELS so every model stays inside Groq's free-tier
     budget (8,000 tokens/minute PER MODEL).
  2. JUDGE — the report agent (LLM_MODEL_NAME) reads the dossier + the six
     theses and writes the final report in the exact format the ResultParser
     expects: an "ANALYST PANEL" section, a "DEBATE" section and the
     "BETTING PREDICTIONS" JSON block.

If fewer than SWARM_MIN_THESES analysts answer, we fall back to the old
single-analyst call, so a prediction never fails because of the swarm.

SWARM_ENABLED=false restores the single-call behaviour.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_LLM_API_KEY   = os.getenv("LLM_API_KEY", "")
_LLM_BASE_URL  = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
# Groq decommissioned llama-3.3-70b-versatile on 2026-08-16; gpt-oss-120b is
# the replacement Groq recommends. This is the JUDGE model.
_LLM_MODEL     = os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-120b")

SWARM_ENABLED = os.getenv("SWARM_ENABLED", "true").strip().lower() in ("1", "true", "yes")
# Persona calls rotate over these models (each has its own free-tier budget).
SWARM_PERSONA_MODELS = [m.strip() for m in os.getenv(
    "SWARM_PERSONA_MODELS", "openai/gpt-oss-20b,qwen/qwen3.8-27b").split(",") if m.strip()]
SWARM_MIN_THESES = int(os.getenv("SWARM_MIN_THESES", "3"))
_PERSONA_PAUSE_S = 0.5


def build_request_payload(model: str, messages: list[dict],
                          temperature: float = 0.3, max_tokens: int = 4096) -> dict:
    """
    Chat-completion body. Reasoning models (gpt-oss, qwen3) spend part of
    max_tokens on hidden reasoning, so the budget is larger than the old
    2048 and reasoning effort is kept low: we want the report + JSON block,
    not a long deliberation.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model.startswith("openai/gpt-oss") or model.startswith("qwen/"):
        payload["reasoning_effort"] = "low"
    return payload


_SYSTEM_PROMPT = (
    "You are an elite football betting analyst with deep knowledge of Serie A "
    "and the UEFA Champions League. You analyse statistical data and produce "
    "structured betting predictions with probability percentages. "
    "Always output a 'BETTING PREDICTIONS' section containing a valid JSON block."
)

# (role key, label template, persona instructions). {home}/{away}/{competition}
# are filled from the match label. These mirror the "Agent profiles to
# generate" list in seed/generator.py.
PERSONAS: list[tuple[str, str, str]] = [
    ("statistician", "Football statistician",
     "You reason only from the numbers in the dossier: xG and xGA, goals per game, form, "
     "home advantage, shrinkage toward league averages. You distrust narratives."),
    ("tactician", "Tactics journalist",
     "You focus on how the two sides play: formations, pressing, key duels, injuries and "
     "who controls the tempo. Numbers matter less to you than matchups."),
    ("bettor", "Veteran {competition} betting analyst",
     "You think in value and market terms: which outcomes the crowd overrates, where the "
     "true probability differs from the obvious story, and you are disciplined about draws."),
    ("home_fan", "{home} supporter",
     "You are an optimistic {home} supporter. You argue for {home} but you must stay within "
     "plausibility: your percentages can be biased, not absurd."),
    ("away_fan", "{away} supporter",
     "You are an optimistic {away} supporter. You argue for {away} but you must stay within "
     "plausibility: your percentages can be biased, not absurd."),
    ("pundit", "Neutral football pundit",
     "You are a balanced pundit with a long memory: head-to-head history, how these clubs "
     "usually handle this kind of fixture, and the psychological stakes."),
]

_PERSONA_FORMAT = (
    "Answer with ONE JSON object only, no prose before or after, with exactly these keys: "
    '{"verdict": "1"|"X"|"2", "home_win_pct": number, "draw_pct": number, "away_win_pct": number, '
    '"over_2_5_pct": number, "btts_yes_pct": number, "scoreline": "H-A", '
    '"argument": "two sentences, max 45 words, in English"}. '
    "home_win_pct + draw_pct + away_win_pct must equal 100."
)


class SimulationError(Exception):
    pass


def _split_label(match_label: str) -> tuple[str, str, str]:
    """'Home vs Away — Competition date' -> (home, away, competition)."""
    teams, _, rest = match_label.partition(" — ")
    home, _, away = teams.partition(" vs ")
    competition = rest.rsplit(" ", 1)[0] if rest else ""
    return home.strip() or "the home side", away.strip() or "the away side", competition.strip() or "league"


def _strip_simulation_instructions(seed_text: str) -> str:
    """Personas get the dossier without the report-agent instructions."""
    return seed_text.split("## SIMULATION INSTRUCTIONS")[0].rstrip()


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _num(v, default: float = 0.0) -> float:
    try:
        return round(float(v), 1)
    except (TypeError, ValueError):
        return default


class MiroFishClient:
    """Mini-swarm prediction client (drop-in replacement for the MiroFish HTTP pipeline)."""

    def __init__(self, base_url: str = _LLM_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.api_key  = _LLM_API_KEY
        self.model    = _LLM_MODEL
        self.persona_models = SWARM_PERSONA_MODELS or [_LLM_MODEL]
        self.swarm_enabled = SWARM_ENABLED

    # ── Public API ────────────────────────────────────────────────────────

    def run_match_prediction(
        self,
        seed_text: str,
        prediction_prompt: str,
        match_label: str = "Match Prediction",
        simulation_rounds: int = 10,
    ) -> dict:
        """
        Returns:
          {"status": "success", "report_markdown": "...", "mode": "swarm"|"single",
           "panel": [...theses...], "simulation_id": None, "report_id": None}
          {"status": "error",   "error": "..."}
        """
        if not self.api_key:
            return {"status": "error", "error": "LLM_API_KEY not configured"}

        if self.swarm_enabled:
            try:
                panel = self._analyst_round(seed_text, match_label)
            except Exception as e:  # never let the swarm kill a prediction
                logger.warning("[SWARM] analyst round failed: %s", e)
                panel = []
            if len(panel) >= SWARM_MIN_THESES:
                result = self._judge(seed_text, prediction_prompt, match_label, panel)
                if result["status"] == "success":
                    result["mode"] = "swarm"
                    result["panel"] = panel
                    return result
                logger.warning("[SWARM] judge failed (%s) — falling back to single analyst",
                               result.get("error"))
            else:
                logger.warning("[SWARM] only %d/%d theses — falling back to single analyst",
                               len(panel), len(PERSONAS))

        result = self._single_analyst(seed_text, prediction_prompt, match_label)
        result["mode"] = "single"
        result["panel"] = []
        return result

    # Keep stub so any code that calls list_simulations() doesn't break
    def list_simulations(self) -> list[dict]:
        return []

    # ── Swarm ─────────────────────────────────────────────────────────────

    def _analyst_round(self, seed_text: str, match_label: str) -> list[dict]:
        home, away, competition = _split_label(match_label)
        dossier = _strip_simulation_instructions(seed_text)
        panel: list[dict] = []
        for i, (role, label_tpl, instr_tpl) in enumerate(PERSONAS):
            label = label_tpl.format(home=home, away=away, competition=competition)
            instructions = instr_tpl.format(home=home, away=away, competition=competition)
            model = self.persona_models[i % len(self.persona_models)]
            system = (f"You are {label}, taking part in a pre-match debate about "
                      f"{home} vs {away} ({competition}). {instructions} {_PERSONA_FORMAT}")
            user = f"{dossier}\n\nGive your thesis now."
            try:
                content = self._chat(model, system, user, temperature=0.7, max_tokens=400)
            except Exception as e:
                logger.warning("[SWARM] %s (%s) failed: %s", role, model, e)
                continue
            data = _extract_json(content)
            if not data:
                logger.warning("[SWARM] %s returned no JSON", role)
                continue
            thesis = {
                "role": role,
                "label": label,
                "model": model,
                "verdict": str(data.get("verdict", "")).strip()[:1] or "?",
                "home_win_pct": _num(data.get("home_win_pct")),
                "draw_pct": _num(data.get("draw_pct")),
                "away_win_pct": _num(data.get("away_win_pct")),
                "over_2_5_pct": _num(data.get("over_2_5_pct")),
                "btts_yes_pct": _num(data.get("btts_yes_pct")),
                "scoreline": str(data.get("scoreline", ""))[:7],
                "argument": str(data.get("argument", "")).strip()[:400],
            }
            panel.append(thesis)
            logger.info("[SWARM] %s -> %s (%s/%s/%s) %s", role, thesis["verdict"],
                        thesis["home_win_pct"], thesis["draw_pct"], thesis["away_win_pct"],
                        thesis["scoreline"])
            time.sleep(_PERSONA_PAUSE_S)
        return panel

    def _judge(self, seed_text: str, prediction_prompt: str, match_label: str,
               panel: list[dict]) -> dict:
        theses = "\n".join(
            f"- **{t['label']}** ({t['model']}): verdict {t['verdict']}, "
            f"1X2 {t['home_win_pct']}/{t['draw_pct']}/{t['away_win_pct']}, "
            f"Over 2.5 {t['over_2_5_pct']}%, BTTS {t['btts_yes_pct']}%, "
            f"scoreline {t['scoreline']}. {t['argument']}"
            for t in panel
        )
        user_message = (
            f"{seed_text}\n\n---\n\n"
            f"## ANALYST PANEL THESES\n\n{theses}\n\n---\n\n"
            f"## PREDICTION TASK\n\n{prediction_prompt}\n\n"
            f"You are the Report Agent. The analysts above have debated the match. "
            f"Write the final report with these sections, in this order:\n"
            f"1. 'ANALYST PANEL' — one line per analyst summarising their position.\n"
            f"2. 'DEBATE' — where the panel agrees, where it disagrees, and whose "
            f"arguments carry more weight given the statistics (discount the two "
            f"supporters' bias explicitly).\n"
            f"3. 'BETTING PREDICTIONS' — the JSON block exactly as specified in the "
            f"simulation instructions, with your consensus probabilities. Every "
            f"over/under and yes/no pair must sum to 100."
        )
        try:
            logger.info("[SWARM] judge (%s) for: %s", self.model, match_label)
            report_md = self._chat(self.model, _SYSTEM_PROMPT, user_message,
                                   temperature=0.3, max_tokens=4096)
            logger.info("[SWARM] judge report complete (%d chars)", len(report_md))
            return {"status": "success", "simulation_id": None, "report_id": None,
                    "report_markdown": report_md}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ── Single analyst (legacy path / fallback) ───────────────────────────

    def _single_analyst(self, seed_text: str, prediction_prompt: str, match_label: str) -> dict:
        user_message = (
            f"{seed_text}\n\n"
            f"---\n\n"
            f"## PREDICTION TASK\n\n"
            f"{prediction_prompt}\n\n"
            f"Analyse all the statistics above carefully. Consider home advantage, "
            f"recent form, head-to-head record, and attacking/defensive metrics. "
            f"Produce a detailed analytical report concluding with a "
            f"'BETTING PREDICTIONS' section that contains the JSON block exactly "
            f"as specified in the simulation instructions."
        )
        try:
            logger.info("[LLM] Requesting prediction for: %s", match_label)
            report_md = self._chat(self.model, _SYSTEM_PROMPT, user_message,
                                   temperature=0.3, max_tokens=4096)
            logger.info("[LLM] Prediction complete (%d chars)", len(report_md))
            return {"status": "success", "simulation_id": None, "report_id": None,
                    "report_markdown": report_md}
        except Exception as e:
            err = str(e)
            logger.error(err)
            return {"status": "error", "error": err}

    # ── Transport ─────────────────────────────────────────────────────────

    def _chat(self, model: str, system: str, user: str,
              temperature: float, max_tokens: int, retries: int = 2) -> str:
        """One chat completion; honours Groq 429 retry-after. Raises on failure."""
        payload = build_request_payload(model, [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], temperature=temperature, max_tokens=max_tokens)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        attempt = 0
        while True:
            response = requests.post(f"{self.base_url}/chat/completions",
                                     headers=headers, json=payload, timeout=120)
            if response.status_code == 429 and attempt < retries:
                wait = _retry_after_seconds(response)
                logger.warning("[LLM] 429 from %s — waiting %.0fs", model, wait)
                time.sleep(wait)
                attempt += 1
                continue
            if not response.ok:
                raise SimulationError(
                    f"LLM API error {response.status_code} ({model}): {response.text[:200]}")
            data = response.json()
            try:
                return data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError) as e:
                raise SimulationError(f"Unexpected LLM response format: {e}")


def _retry_after_seconds(response) -> float:
    raw = response.headers.get("retry-after", "") if hasattr(response, "headers") else ""
    try:
        return min(max(float(raw), 2.0), 60.0)
    except (TypeError, ValueError):
        return 20.0
