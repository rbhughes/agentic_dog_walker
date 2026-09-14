"""Service armor and the event stream, offline.

The agent's model calls are faked with a scripted backend (a canned
sequence of chat() replies), so these tests prove the loop's event
vocabulary and the service's front-door defenses without a network
or a model. Same rule as everywhere: one behavioral claim per test.
"""

import json

import pytest
from fastapi.testclient import TestClient

from dog_walker import agent
from dog_walker.presets import PRESETS, build_request, seed_geocode_cache
from dog_walker.service import RateLimiter, app

# ---------------------------------------------------------------------
# scripted backend: chat() replies from a list, in order
# ---------------------------------------------------------------------


def scripted_chat(replies: list[dict]):
    """A stand-in for agent.chat that pops canned replies. Each canned
    reply needs only content/tool_calls; _raw mirrors it (the loop
    appends _raw to state, and the auditor may read it)."""
    remaining = list(replies)

    def fake_chat(backend, messages, think=False):
        reply = dict(remaining.pop(0))
        reply.setdefault("role", "assistant")
        reply.setdefault("content", "")
        reply.setdefault("tool_calls", [])
        reply["_raw"] = {k: v for k, v in reply.items() if k != "_raw"}
        return reply

    return fake_chat


def call(name: str, arguments: dict) -> dict:
    return {"id": "c1", "function": {"name": name, "arguments": arguments}}


GOOD_PLAN = {
    "walks": [{"pet": "Daisy", "walk_start": "13:09", "walk_end": "13:29",
               "verdict": "OK"}],
    "feasible": True,
    "overall_advice": "fine day for it",
}


def _audit_satisfied(messages):
    """Stand-in auditor: these tests pin the stream SHAPE; audit
    policy (route required, coverage) has its own tests in
    test_agent.py."""
    return None


def test_stream_ends_with_final_on_the_happy_path(monkeypatch):
    monkeypatch.setattr(agent, "audit_weather_coverage", _audit_satisfied)
    monkeypatch.setattr(agent, "chat", scripted_chat([
        {"reasoning": "the plan: just submit"},        # plan round
        {"tool_calls": [call("submit_plan", GOOD_PLAN)]},
    ]))
    events = list(agent.run_events("demo request"))
    kinds = [e["event"] for e in events]
    assert kinds == ["start", "plan", "call", "final"]
    assert events[-1]["plan"]["walks"][0]["pet"] == "Daisy"


def test_plan_event_reads_reasoning_not_just_content(monkeypatch):
    # the line-363 finding: OpenRouter puts the plan in `reasoning`
    monkeypatch.setattr(agent, "chat", scripted_chat([
        {"reasoning": "geocode, then route, then weather"},
        {"tool_calls": [call("submit_plan", GOOD_PLAN)]},
    ]))
    events = list(agent.run_events("demo"))
    plan = next(e for e in events if e["event"] == "plan")
    assert "geocode" in plan["text"]


def test_invalid_call_becomes_bounce_event_then_recovers(monkeypatch):
    bad = {"walks": [], "feasible": True, "overall_advice": 42}  # advice: wrong type
    monkeypatch.setattr(agent, "audit_weather_coverage", _audit_satisfied)
    monkeypatch.setattr(agent, "chat", scripted_chat([
        {},                                            # plan: nothing
        {"tool_calls": [call("submit_plan", bad)]},
        {"tool_calls": [call("submit_plan", GOOD_PLAN)]},
    ]))
    kinds = [e["event"] for e in agent.run_events("demo")]
    assert kinds == ["start", "call", "bounce", "call", "final"]


def test_prose_stall_yields_nudges_then_error(monkeypatch):
    monkeypatch.setattr(agent, "chat", scripted_chat(
        [{"content": "here is your plan in prose!"}] * 5
    ))
    events = list(agent.run_events("demo"))
    assert [e["event"] for e in events].count("nudge") == 2
    assert events[-1]["event"] == "error"


def test_exception_in_stream_becomes_error_event(monkeypatch):
    def exploding_chat(backend, messages, think=False):
        raise ConnectionError("backend melted")

    monkeypatch.setattr(agent, "chat", exploding_chat)
    events = list(agent.run_events("demo"))
    assert events[-1]["event"] == "error"
    assert "backend melted" in events[-1]["message"]


# ---------------------------------------------------------------------
# the front door
# ---------------------------------------------------------------------

client = TestClient(app)


def test_info_reports_model_backend_and_picker_list():
    payload = client.get("/info").json()
    assert "model" in payload and "backend" in payload
    ids = [m["id"] for m in payload["models"]]
    assert "qwen/qwen3-8b" in ids


def test_off_list_model_is_a_422():
    r = client.post("/plan", json={"preset": "lakeview-classic",
                                   "model": "openai/o5-preview-ultra"})
    assert r.status_code == 422


def test_presets_endpoint_lists_the_rosters():
    listed = client.get("/presets").json()
    assert {p["id"] for p in listed} == set(PRESETS)


def test_seven_pets_is_a_422():
    pets = [{"name": f"p{i}", "address": "123 Main St, Chicago",
             "walk_minutes": 20} for i in range(7)]
    r = client.post("/plan", json={"start_address": "A St, Chicago",
                                   "start_time": "09:00", "pets": pets})
    assert r.status_code == 422


def test_off_menu_walk_duration_is_a_422():
    pets = [{"name": "Rex", "address": "123 Main St, Chicago",
             "walk_minutes": 45}]
    r = client.post("/plan", json={"start_address": "A St, Chicago",
                                   "start_time": "09:00", "pets": pets})
    assert r.status_code == 422


def test_preset_and_custom_together_is_a_422():
    r = client.post("/plan", json={"preset": "lakeview-classic",
                                   "start_time": "09:00"})
    assert r.status_code == 422


def test_unknown_preset_is_a_422():
    r = client.post("/plan", json={"preset": "does-not-exist"})
    assert r.status_code == 422


def test_rate_limiter_counts_and_resets():
    rl = RateLimiter(limit=2, window_s=3600)
    assert rl.allow("1.2.3.4") and rl.allow("1.2.3.4")
    assert not rl.allow("1.2.3.4")          # third strike
    assert rl.allow("5.6.7.8")              # other IPs unaffected


def test_preset_run_needs_no_geocoding():
    # every preset address must be pre-seeded, so anonymous demo runs
    # never touch Nominatim
    from dog_walker.toolbox import _geocode_cache
    seed_geocode_cache()
    for p in PRESETS.values():
        for addr in [p["start_address"]] + [x["address"] for x in p["pets"]]:
            assert addr.strip().lower() in _geocode_cache, addr


def test_build_request_renders_one_fact_per_sentence():
    text = build_request("A St", "09:00", [
        {"name": "Rex", "address": "B St", "walk_minutes": 60}])
    assert "starting and ending at A St" in text
    assert "Rex is at B St and gets a 60 minute walk." in text


# ---------------------------------------------------------------------
# the qualifier's deterministic parts (discovery filter + gate logic)
# ---------------------------------------------------------------------

from dog_walker import qualify


def catalog_entry(mid, prompt, completion, params=("tools", "reasoning")):
    return {"id": mid, "name": mid, "pricing":
            {"prompt": str(prompt), "completion": str(completion)},
            "supported_parameters": list(params)}


def test_discover_filters_by_price_and_capabilities():
    catalog = [
        catalog_entry("cheap/good", 0.05e-6, 0.2e-6),
        catalog_entry("pricey/good", 5e-6, 20e-6),          # over cap
        catalog_entry("cheap/no-tools", 0.05e-6, 0.2e-6, params=("reasoning",)),
        catalog_entry("cheap/no-think", 0.05e-6, 0.2e-6, params=("tools",)),
        catalog_entry("cheap/free:free", 0, 0),             # free alias
        catalog_entry("qwen/qwen3-8b", 0.05e-6, 0.1e-6),    # pinned: skip
    ]
    ids = [c["id"] for c in qualify.discover(catalog)]
    assert ids == ["cheap/good"]


def test_discover_sorts_cheapest_first():
    catalog = [
        catalog_entry("b/mid", 0.10e-6, 0.4e-6),
        catalog_entry("a/cheapest", 0.02e-6, 0.1e-6),
    ]
    ids = [c["id"] for c in qualify.discover(catalog)]
    assert ids == ["a/cheapest", "b/mid"]


def test_qualify_fails_a_routeless_finish(monkeypatch):
    # the gemini-flash-lite exploit as the gate sees it
    def fabricator(prompt, model=None):
        yield {"event": "start", "model": model, "backend": "openrouter"}
        yield {"event": "final", "plan": {"walks": [], "overall_advice": "x"}}

    monkeypatch.setattr("dog_walker.agent.run_events", fabricator)
    verdict = qualify.qualify("fake/model")
    assert not verdict["passed"] and "route" in verdict["reason"]


def test_qualify_passes_a_routed_finish(monkeypatch):
    def honest(prompt, model=None):
        yield {"event": "start", "model": model, "backend": "openrouter"}
        yield {"event": "result", "name": "optimize_route",
               "result": {"timeline": [], "order": []}}
        yield {"event": "final", "plan": {"walks": [], "overall_advice": "x"}}

    monkeypatch.setattr("dog_walker.agent.run_events", honest)
    verdict = qualify.qualify("fake/model")
    assert verdict["passed"]


def test_discover_excludes_router_batch_and_negative_prices():
    catalog = [
        catalog_entry("openrouter/auto", -1, -1),
        catalog_entry("x/model:batch", 0.05e-6, 0.2e-6),
        catalog_entry("y/honest", 0.05e-6, 0.2e-6),
    ]
    assert [c["id"] for c in qualify.discover(catalog)] == ["y/honest"]
