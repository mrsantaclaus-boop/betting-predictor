"""
predictor/mirofish_client.py

Direct LLM prediction client using Groq (OpenAI-compatible API).
Replaces the MiroFish HTTP pipeline — same interface, no port 5001 needed.

The seed document + prediction prompt are sent as a single chat completion
request to the configured LLM, which returns a structured markdown report
containing the BETTING PREDICTIONS JSON block.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_LLM_API_KEY   = os.getenv("LLM_API_KEY", "")
_LLM_BASE_URL  = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
_LLM_MODEL     = os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")

_SYSTEM_PROMPT = (
    "You are an elite football betting analyst with deep knowledge of Serie A "
    "and the UEFA Champions League. You analyse statistical data and produce "
    "structured betting predictions with probability percentages. "
    "Always output a 'BETTING PREDICTIONS' section containing a valid JSON block."
)


class SimulationError(Exception):
    pass


class MiroFishClient:
    """LLM-based prediction client (drop-in replacement for MiroFish HTTP pipeline)."""

    def __init__(self, base_url: str = _LLM_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.api_key  = _LLM_API_KEY
        self.model    = _LLM_MODEL

    def run_match_prediction(
        self,
        seed_text: str,
        prediction_prompt: str,
        match_label: str = "Match Prediction",
        simulation_rounds: int = 10,
    ) -> dict:
        """
        Call the LLM with full match context and return a report dict.

        Returns:
          {"status": "success", "report_markdown": "...", "simulation_id": None, "report_id": None}
          {"status": "error",   "error": "..."}
        """
        if not self.api_key:
            return {"status": "error", "error": "LLM_API_KEY not configured"}

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
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2048,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            report_md = data["choices"][0]["message"]["content"]
            logger.info("[LLM] Prediction complete (%d chars)", len(report_md))
            return {
                "status": "success",
                "simulation_id": None,
                "report_id": None,
                "report_markdown": report_md,
            }

        except requests.HTTPError as e:
            err = f"LLM API error {e.response.status_code}: {e.response.text[:200]}"
            logger.error(err)
            return {"status": "error", "error": err}
        except requests.RequestException as e:
            err = f"LLM request failed: {e}"
            logger.error(err)
            return {"status": "error", "error": err}
        except (KeyError, IndexError) as e:
            err = f"Unexpected LLM response format: {e}"
            logger.error(err)
            return {"status": "error", "error": err}

    # Keep stub so any code that calls list_simulations() doesn't break
    def list_simulations(self) -> list[dict]:
        return []
