"""Mini-swarm client tested with a fake Groq transport."""
import json

import pytest

import predictor.mirofish_client as mc
from predictor.mirofish_client import MiroFishClient, PERSONAS, _split_label, _strip_simulation_instructions
from predictor.result_parser import ResultParser

SEED = ("# MATCH ANALYSIS REPORT\n## Genoa CFC vs ACF Fiorentina\nstats...\n\n"
        "## SIMULATION INSTRUCTIONS\n\nreport agent must produce JSON")
PROMPT = "Simulate expert analysts... Format the predictions as a JSON block inside the report."
LABEL = "Genoa CFC vs ACF Fiorentina — Serie A 10/10/2026"

JUDGE_REPORT = """## ANALYST PANEL
- Statistician: X
## DEBATE
They disagree.
## BETTING PREDICTIONS
```json
{"home_win_pct": 30.0, "draw_pct": 32.0, "away_win_pct": 38.0, "over_2_5_pct": 48.0,
 "under_2_5_pct": 52.0, "over_3_5_pct": 25.0, "under_3_5_pct": 75.0, "btts_yes_pct": 55.0,
 "btts_no_pct": 45.0, "over_9_5_corners_pct": 50.0, "under_9_5_corners_pct": 50.0,
 "over_3_5_cards_pct": 40.0, "under_3_5_cards_pct": 60.0, "red_card_pct": 12.0,
 "most_likely_scoreline": "1-1", "confidence": "medium"}
```"""


class FakeResponse:
    def __init__(self, status, content="", headers=None):
        self.status_code = status
        self.ok = 200 <= status < 300
        self.headers = headers or {}
        self.text = content
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def make_transport(persona_content=None, judge_content=JUDGE_REPORT, fail_personas=0, first_429=False):
    calls = []
    state = {"429_done": False}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        model = json["model"]
        system = json["messages"][0]["content"]
        if first_429 and not state["429_done"]:
            state["429_done"] = True
            return FakeResponse(429, "rate limited", {"retry-after": "0"})
        if "taking part in a pre-match debate" in system:
            idx = sum(1 for c in calls if "pre-match debate" in c["messages"][0]["content"]) - 1
            if idx < fail_personas:
                return FakeResponse(500, "boom")
            body = persona_content or {"verdict": "2", "home_win_pct": 30, "draw_pct": 30,
                                       "away_win_pct": 40, "over_2_5_pct": 50, "btts_yes_pct": 55,
                                       "scoreline": "1-2", "argument": f"Thesis from {model}."}
            return FakeResponse(200, "Sure:\n" + __import__("json").dumps(body))
        return FakeResponse(200, judge_content)

    return fake_post, calls


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(mc, "_PERSONA_PAUSE_S", 0)
    c = MiroFishClient()
    c.api_key = "test"
    c.model = "openai/gpt-oss-120b"
    c.persona_models = ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    c.swarm_enabled = True
    return c


def test_swarm_runs_six_personas_then_judge(client, monkeypatch):
    post, calls = make_transport()
    monkeypatch.setattr(mc.requests, "post", post)
    r = client.run_match_prediction(SEED, PROMPT, LABEL)
    assert r["status"] == "success" and r["mode"] == "swarm"
    assert len(r["panel"]) == len(PERSONAS) == 6
    assert len(calls) == 7
    # personas rotate over the two models; the judge uses the main model
    assert [c["model"] for c in calls[:6]] == ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"] * 3
    assert calls[6]["model"] == "openai/gpt-oss-120b"
    # judge sees the theses and the original prompt
    judge_user = calls[6]["messages"][1]["content"]
    assert "ANALYST PANEL THESES" in judge_user and "Genoa CFC supporter" in judge_user
    assert PROMPT in judge_user
    # the report is parseable by the existing parser
    pred = ResultParser().parse(r["report_markdown"])
    assert pred.parse_source == "json" and pred.away_win_pct == 38.0


def test_personas_do_not_receive_simulation_instructions(client, monkeypatch):
    post, calls = make_transport()
    monkeypatch.setattr(mc.requests, "post", post)
    client.run_match_prediction(SEED, PROMPT, LABEL)
    assert "SIMULATION INSTRUCTIONS" not in calls[0]["messages"][1]["content"]
    assert "SIMULATION INSTRUCTIONS" in calls[6]["messages"][1]["content"]


def test_fallback_to_single_when_too_few_theses(client, monkeypatch):
    post, calls = make_transport(fail_personas=4)   # only 2 of 6 answer
    monkeypatch.setattr(mc.requests, "post", post)
    r = client.run_match_prediction(SEED, PROMPT, LABEL)
    assert r["status"] == "success" and r["mode"] == "single" and r["panel"] == []
    assert len(calls) == 7   # 6 persona attempts + 1 single-analyst call, no judge


def test_swarm_disabled_uses_single_call(client, monkeypatch):
    client.swarm_enabled = False
    post, calls = make_transport()
    monkeypatch.setattr(mc.requests, "post", post)
    r = client.run_match_prediction(SEED, PROMPT, LABEL)
    assert r["mode"] == "single" and len(calls) == 1


def test_429_is_retried(client, monkeypatch):
    monkeypatch.setattr(mc.time, "sleep", lambda s: None)
    post, calls = make_transport(first_429=True)
    monkeypatch.setattr(mc.requests, "post", post)
    r = client.run_match_prediction(SEED, PROMPT, LABEL)
    assert r["mode"] == "swarm" and len(calls) == 8   # one extra call for the retry


def test_extract_json_tolerates_thinking_and_prose():
    from predictor.mirofish_client import _extract_json, build_request_payload
    wrapped = ('<think>Let me weigh {home: strong} vs {away}...</think>\n'
               'Here is my thesis:\n```json\n{"verdict": "X", "home_win_pct": 33, "argument": "a {tight} game"}\n```')
    assert _extract_json(wrapped) == {"verdict": "X", "home_win_pct": 33, "argument": "a {tight} game"}
    assert _extract_json("no json here { broken") is None
    assert _extract_json("") is None
    q = build_request_payload("qwen/qwen3.8-27b", [])
    assert q["reasoning_format"] == "hidden" and q["reasoning_effort"] == "none"
    g = build_request_payload("openai/gpt-oss-20b", [])
    assert "reasoning_format" not in g and g["reasoning_effort"] == "low"


def test_label_split_and_persona_labels():
    assert _split_label(LABEL) == ("Genoa CFC", "ACF Fiorentina", "Serie A")
    assert _split_label("A vs B") == ("A", "B", "league")
    assert _strip_simulation_instructions(SEED).endswith("stats...")
