from predictor.mirofish_client import build_request_payload


def test_reasoning_models_get_low_effort_and_bigger_budget():
    p = build_request_payload("openai/gpt-oss-120b", [{"role": "user", "content": "x"}])
    assert p["reasoning_effort"] == "low"
    assert p["max_tokens"] >= 4096
    assert p["model"] == "openai/gpt-oss-120b"


def test_plain_models_have_no_reasoning_param():
    p = build_request_payload("llama-3.1-8b-instant", [{"role": "user", "content": "x"}])
    assert "reasoning_effort" not in p
    assert p["temperature"] == 0.3
