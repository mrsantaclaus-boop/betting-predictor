"""LLM call/token monitoring: counted per prediction, summed per day, never read back by the model."""
from datetime import datetime, timezone

import pytest

import predictor.mirofish_client as mc
from predictor.mirofish_client import MiroFishClient, new_usage, _add_usage
from tests.test_swarm import SEED, PROMPT, LABEL, JUDGE_REPORT
from api_server import _llm_usage_report, app


class UsageResponse:
    """Groq-style response, with an OpenAI-compatible `usage` block (unless `usage` is None)."""
    def __init__(self, status, content="", usage=None, headers=None):
        self.status_code = status
        self.ok = 200 <= status < 300
        self.headers = headers or {}
        self.text = content
        self._content = content
        self._usage = usage

    def json(self):
        body = {"choices": [{"message": {"content": self._content}}]}
        if self._usage is not None:
            body["usage"] = self._usage
        return body


def transport(usage=None, first_429=False):
    state = {"429_done": False}
    thesis = {"verdict": "2", "home_win_pct": 30, "draw_pct": 30, "away_win_pct": 40,
              "over_2_5_pct": 50, "btts_yes_pct": 55, "scoreline": "1-2", "argument": "x"}

    def fake_post(url, headers=None, json=None, timeout=None):
        if first_429 and not state["429_done"]:
            state["429_done"] = True
            return UsageResponse(429, "rate limited", headers={"retry-after": "0"})
        system = json["messages"][0]["content"]
        content = ("Sure:\n" + __import__("json").dumps(thesis)) if "pre-match debate" in system else JUDGE_REPORT
        return UsageResponse(200, content, usage)
    return fake_post


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(mc, "_PERSONA_PAUSE_S", 0)
    c = MiroFishClient()
    c.api_key = "test"
    c.model = "openai/gpt-oss-120b"
    c.persona_models = ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    c.swarm_enabled = True
    return c


USAGE = {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200}


def test_tokens_summed_over_the_seven_swarm_calls(client, monkeypatch):
    monkeypatch.setattr(mc.requests, "post", transport(USAGE))
    r = client.run_match_prediction(SEED, PROMPT, LABEL)
    u = r["usage"]
    assert u["calls"] == 7
    assert (u["prompt_tokens"], u["completion_tokens"], u["total_tokens"]) == (7000, 1400, 8400)
    assert u["by_model"]["openai/gpt-oss-20b"] == {"calls": 3, "total_tokens": 3600}
    assert u["by_model"]["qwen/qwen3.8-27b"] == {"calls": 3, "total_tokens": 3600}
    assert u["by_model"]["openai/gpt-oss-120b"] == {"calls": 1, "total_tokens": 1200}
    assert u["rate_limited"] == 0 and u["errors"] == 0


def test_a_429_is_counted_but_is_not_a_call(client, monkeypatch):
    monkeypatch.setattr(mc.requests, "post", transport(USAGE, first_429=True))
    monkeypatch.setattr(mc.time, "sleep", lambda s: None)
    u = client.run_match_prediction(SEED, PROMPT, LABEL)["usage"]
    assert u["rate_limited"] == 1
    assert u["calls"] == 7 and u["total_tokens"] == 8400


def test_response_without_usage_field_does_not_break_the_prediction(client, monkeypatch):
    monkeypatch.setattr(mc.requests, "post", transport(usage=None))
    r = client.run_match_prediction(SEED, PROMPT, LABEL)
    assert r["status"] == "success" and r["mode"] == "swarm"
    assert r["usage"]["calls"] == 7 and r["usage"]["total_tokens"] == 0


def test_total_falls_back_to_prompt_plus_completion():
    u = new_usage()
    _add_usage(u, "m", tokens={"prompt_tokens": 10, "completion_tokens": 5})
    assert u["total_tokens"] == 15


def test_usage_none_is_a_no_op():
    _add_usage(None, "m", tokens=USAGE)   # must not raise


# ── per-day report ──────────────────────────────────────────────────────────

NOW = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)


def _pred(day, total, calls=7, limited=0):
    return {"created_at": f"{day}T08:00:00+00:00",
            "llm_usage": {"calls": calls, "prompt_tokens": total - 100, "completion_tokens": 100,
                          "total_tokens": total, "rate_limited": limited, "errors": 0,
                          "by_model": {"openai/gpt-oss-120b": {"calls": 1, "total_tokens": total // 7}}}}


def test_report_sums_per_day_and_ignores_old_days():
    preds = [_pred("2026-11-20", 8000), _pred("2026-11-20", 9000, limited=2),
             _pred("2026-11-19", 7000), _pred("2026-11-01", 99999),
             {"created_at": "2026-11-20T09:00:00+00:00"}]          # no usage (pre-monitoring prediction)
    rep = _llm_usage_report(preds, days=3, now=NOW)
    assert rep["today"]["total_tokens"] == 17000 and rep["today"]["predictions"] == 3
    assert rep["today"]["with_usage"] == 2 and rep["today"]["rate_limited"] == 2
    assert [d["day"] for d in rep["days"]] == ["2026-11-20", "2026-11-19"]    # 11-01 is outside the window
    assert rep["today_pct_of_daily_budget"] is None


def test_report_percentage_of_budget_when_known():
    rep = _llm_usage_report([_pred("2026-11-20", 50000)], days=1, now=NOW, daily_budget=200000)
    assert rep["today_pct_of_daily_budget"] == 25.0


def test_report_with_no_predictions_is_all_zero():
    rep = _llm_usage_report([], days=7, now=NOW)
    assert rep["today"]["total_tokens"] == 0 and rep["days"] == []


def test_endpoint_is_public_read_only_and_returns_the_report():
    r = app.test_client().get("/api/llm-usage?days=2")
    assert r.status_code == 200
    body = r.get_json()
    assert set(body) >= {"today", "days", "daily_budget_tokens", "note"}
