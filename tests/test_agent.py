"""Agent internals, offline: the referee and the auditor.

The loop itself needs a model; these test the deterministic parts
that make the loop trustworthy. Same rule as test_toolbox: one
behavioral claim per test, no network, no model.
"""

import io
import json

import dog_walker.agent as agent
from dog_walker.agent import (_call_args, audit_daylight, audit_feasibility, audit_priority, audit_rain, audit_terrain_coverage, audit_verdict, audit_weather_coverage, chat, scrub_optionals, validate_call)


def call_with_id(name, args, cid):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": cid, "function": {"name": name, "arguments": args}}]}


def result_for(cid, payload):
    return {"role": "tool", "tool_call_id": cid, "content": json.dumps(payload)}


# ---------------------------------------------------------------------
# chat(): transient-response retry
# ---------------------------------------------------------------------


def _no_sleep(*_a, **_k):
    return None


def test_chat_retries_unusable_body_then_succeeds(monkeypatch):
    # a 200 whose body has no "choices" (measured: mercury) must retry,
    # not escape as a terminal error counted against the model
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(agent.time, "sleep", _no_sleep)
    good = {"choices": [{"message": {"role": "assistant", "content": "ok",
                                     "tool_calls": []}}], "usage": {}}
    responses = iter([{"error": "upstream hiccup"}, good])  # bad, then good

    def fake_urlopen(_req, timeout=None):
        return io.BytesIO(json.dumps(next(responses)).encode())

    monkeypatch.setattr(agent.urllib.request, "urlopen", fake_urlopen)
    result = chat("some/model", [{"role": "user", "content": "hi"}])
    assert result["content"] == "ok"


def test_chat_retries_truncated_tool_json(monkeypatch):
    # a 200 with malformed tool-call arguments (measured: gpt-oss) retries
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(agent.time, "sleep", _no_sleep)
    truncated = {"choices": [{"message": {"role": "assistant", "tool_calls": [
        {"id": "1", "function": {"name": "check_weather",
                                 "arguments": '{"lat": 41.9, "lon":'}}]}}]}
    good = {"choices": [{"message": {"role": "assistant", "tool_calls": [
        {"id": "1", "function": {"name": "check_weather",
                                 "arguments": '{"lat": 41.9, "lon": -87.6}'}}]}}]}
    responses = iter([truncated, good])

    def fake_urlopen(_req, timeout=None):
        return io.BytesIO(json.dumps(next(responses)).encode())

    monkeypatch.setattr(agent.urllib.request, "urlopen", fake_urlopen)
    result = chat("some/model", [{"role": "user", "content": "hi"}])
    assert result["tool_calls"][0]["function"]["arguments"] == {"lat": 41.9, "lon": -87.6}


def _http_429(retry_after=None):
    """Build a 429 HTTPError, optionally carrying a Retry-After header."""
    import email.message
    hdrs = email.message.Message()
    if retry_after is not None:
        hdrs["Retry-After"] = str(retry_after)
    return agent.urllib.error.HTTPError(
        "http://x", 429, "Too Many Requests", hdrs, None)


def test_chat_rides_out_a_429_then_succeeds(monkeypatch):
    # a 429 is throttling, not the model's answer: back off and retry.
    # the server's Retry-After must be honored, and the throttle counted.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    slept = []
    monkeypatch.setattr(agent.time, "sleep", lambda s: slept.append(s))
    good = {"choices": [{"message": {"role": "assistant", "content": "ok",
                                     "tool_calls": []}}], "usage": {}}
    calls = iter([_http_429(retry_after=7), None])  # throttle once, then OK

    def fake_urlopen(_req, timeout=None):
        nxt = next(calls)
        if isinstance(nxt, Exception):
            raise nxt
        return io.BytesIO(json.dumps(good).encode())

    monkeypatch.setattr(agent.urllib.request, "urlopen", fake_urlopen)
    result = chat("some/model", [{"role": "user", "content": "hi"}])
    assert result["content"] == "ok"
    assert result["_meta"]["throttles"] == 1
    assert slept == [7]                    # honored the Retry-After exactly


def test_chat_never_retries_past_the_deadline(monkeypatch):
    # the whole POINT of the redesign: retries are bounded by the run
    # deadline. A deadline already in the past means not even one read is
    # attempted -- a persistently-throttled provider can't hang the run.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(agent.time, "sleep", _no_sleep)
    opened = []

    def fake_urlopen(_req, timeout=None):
        opened.append(timeout)
        raise _http_429()

    monkeypatch.setattr(agent.urllib.request, "urlopen", fake_urlopen)
    past = agent.time.monotonic() - 1
    with __import__("pytest").raises(RuntimeError):
        chat("some/model", [{"role": "user", "content": "hi"}], deadline=past)
    assert opened == []                    # deadline gate fired before any read


def test_backoff_is_exponential_and_capped():
    # 4, 8, 16, 32, 60(cap) -- and Retry-After overrides when present
    assert agent._backoff_seconds(0, None) == 4
    assert agent._backoff_seconds(1, None) == 8
    assert agent._backoff_seconds(2, None) == 16
    assert agent._backoff_seconds(9, None) == agent.CHAT_BACKOFF_MAX_S
    assert agent._backoff_seconds(0, 5) == 5          # server's word wins
    assert agent._backoff_seconds(0, 999) == agent.CHAT_BACKOFF_MAX_S  # but capped

# ---------------------------------------------------------------------
# validate_call: the referee
# ---------------------------------------------------------------------


def test_legal_call_passes():
    assert validate_call("check_weather", {"lat": 41.9, "lon": -87.6,
                                           "date": "2026-09-08"}) is None


def test_unknown_tool_is_an_error_string_not_an_exception():
    error = validate_call("check_wether", {})
    assert "unknown tool" in error and "check_weather" in error


def test_wrong_type_is_named_and_located():
    error = validate_call("check_weather", {"lat": "north", "lon": -87.6,
                                            "date": "2026-09-08"})
    assert "lat" in error and "north" in error


def test_invented_argument_is_rejected():
    # the classic invented argument ("time"), refused at the gate
    error = validate_call("check_weather", {"lat": 41.9, "lon": -87.6,
                                            "date": "2026-09-08", "time": "4pm"})
    assert "time" in error


def test_walk_duration_enum_is_enforced():
    stops = [{"name": "home", "lat": 0.0, "lon": 0.0},
             {"name": "Rex", "lat": 0.0, "lon": 0.1, "walk_minutes": 45}]
    error = validate_call("optimize_route", {"stops": stops})
    assert "45" in error and "walk_minutes" in error


def test_submit_plan_is_validated_like_any_tool():
    error = validate_call("submit_plan", {
        "walks": [{"pet": "Rex", "walk_start": "13:00",
                   "walk_end": "14:00", "verdict": "MAYBE"}],
        "feasible": True,
        "overall_advice": "eh",
    })
    assert "MAYBE" in error


# ---------------------------------------------------------------------
# audit_weather_coverage: the auditor
# ---------------------------------------------------------------------

ZOO = (41.9212, -87.6337)


def assistant_call(name: str, arguments: dict, as_string: bool = False) -> dict:
    """A raw assistant message carrying one tool call. as_string=True
    mimics the OpenAI dialect (arguments as a JSON string)."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "c1",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments) if as_string else arguments,
            },
        }],
    }


def tool_result(payload: dict) -> dict:
    return {"role": "tool", "content": json.dumps(payload)}


def route_exchange() -> list[dict]:
    """A route call + its timeline: Daisy walks 16:09-16:29 at the zoo."""
    call = assistant_call("optimize_route", {
        "stops": [
            {"name": "home", "lat": 41.948, "lon": -87.655},
            {"name": "Daisy", "lat": ZOO[0], "lon": ZOO[1], "walk_minutes": 20},
        ],
        "start_time": "13:00",
    })
    result = tool_result({
        "order": ["home", "Daisy"],
        "timeline": [
            {"stop": "Daisy", "arrive": "16:09", "walk_start": "16:09",
             "walk_end": "16:29", "walk_minutes": 20},
            {"stop": "home", "arrive": "17:00"},
        ],
    })
    return [call, result]


def weather_call(lat: float, lon: float, start: int, end: int) -> dict:
    return assistant_call("check_weather", {
        "lat": lat, "lon": lon, "date": "2026-09-08",
        "start_hour": start, "end_hour": end,
    })


def test_submitting_without_a_route_is_rejected():
    # the gemini-2.5-flash-lite exploit: geocode, then submit invented
    # times and verdicts with no route and no weather check
    messages = [weather_call(*ZOO, 13, 14)]
    gap = audit_weather_coverage(messages)
    assert "no route" in gap and "optimize_route" in gap


def test_uncovered_walk_is_reported_by_name_and_interval():
    # the start-location-only failure, now caught: weather checked 13-14,
    # Daisy actually walks 16:09-16:29
    messages = [weather_call(*ZOO, 13, 14), *route_exchange()]
    gap = audit_weather_coverage(messages)
    assert "Daisy" in gap and "16:09" in gap


def test_covering_check_satisfies_the_auditor():
    messages = [weather_call(*ZOO, 16, 17), *route_exchange()]
    assert audit_weather_coverage(messages) is None


def test_right_time_wrong_place_is_still_a_gap():
    # checked 16-17 at the START location, not at Daisy's
    messages = [weather_call(41.948, -87.655, 16, 17), *route_exchange()]
    assert "Daisy" in audit_weather_coverage(messages)


def test_all_gaps_reported_in_one_veto():
    # serial revelation cost 2 rounds per dog and exhausted the round
    # budget live; the auditor now lists every gap at once
    two_dogs = assistant_call("optimize_route", {
        "stops": [
            {"name": "home", "lat": 41.948, "lon": -87.655},
            {"name": "Daisy", "lat": ZOO[0], "lon": ZOO[1], "walk_minutes": 20},
            {"name": "Rex", "lat": 41.9764, "lon": -87.6685, "walk_minutes": 60},
        ],
        "start_time": "13:00",
    })
    result = tool_result({"timeline": [
        {"stop": "Rex", "arrive": "13:30", "walk_start": "13:30",
         "walk_end": "14:30", "walk_minutes": 60},
        {"stop": "Daisy", "arrive": "16:09", "walk_start": "16:09",
         "walk_end": "16:29", "walk_minutes": 20},
    ]})
    gap = audit_weather_coverage([two_dogs, result])
    assert "Rex" in gap and "Daisy" in gap and "ALL" in gap


def test_openai_dialect_string_arguments_are_understood():
    call = assistant_call("optimize_route", {
        "stops": [
            {"name": "home", "lat": 41.948, "lon": -87.655},
            {"name": "Daisy", "lat": ZOO[0], "lon": ZOO[1], "walk_minutes": 20},
        ],
        "start_time": "13:00",
    }, as_string=True)
    result = tool_result({"timeline": [
        {"stop": "Daisy", "arrive": "16:09", "walk_start": "16:09",
         "walk_end": "16:29", "walk_minutes": 20},
    ]})
    assert "Daisy" in audit_weather_coverage([call, result])


def test_minutes_timeline_is_rejected_not_waved_through():
    # no start_time -> intervals unauditable; unauditable != unaudited
    call = assistant_call("optimize_route", {"stops": [
        {"name": "home", "lat": 41.948, "lon": -87.655},
        {"name": "Daisy", "lat": ZOO[0], "lon": ZOO[1], "walk_minutes": 20},
    ]})
    result = tool_result({"timeline": [
        {"stop": "Daisy", "arrive": 10, "walk_start": 10,
         "walk_end": 30, "walk_minutes": 20},
    ]})
    assert "start_time" in audit_weather_coverage([call, result])


def test_band_mismatch_is_a_gap():
    # Daisy's band is [45, 95]: a default-band weather check judges
    # her by the wrong edges, so it does not cover her
    call = assistant_call("optimize_route", {
        "stops": [
            {"name": "home", "lat": 41.948, "lon": -87.655},
            {"name": "Daisy", "lat": ZOO[0], "lon": ZOO[1],
             "walk_minutes": 20, "comfort_min_f": 45, "comfort_max_f": 95},
        ],
        "start_time": "13:00",
    })
    result = tool_result({"timeline": [
        {"stop": "Daisy", "arrive": "16:09", "walk_start": "16:09",
         "walk_end": "16:29", "walk_minutes": 20,
         "comfort_min_f": 45, "comfort_max_f": 95},
    ]})
    plain_check = weather_call(*ZOO, 16, 17)   # default band
    gap = audit_weather_coverage([plain_check, call, result])
    assert gap and "comfort_min_f=45" in gap


def test_matching_band_satisfies_the_auditor():
    call = assistant_call("optimize_route", {
        "stops": [
            {"name": "home", "lat": 41.948, "lon": -87.655},
            {"name": "Daisy", "lat": ZOO[0], "lon": ZOO[1],
             "walk_minutes": 20, "comfort_min_f": 45, "comfort_max_f": 95},
        ],
        "start_time": "13:00",
    })
    result = tool_result({"timeline": [
        {"stop": "Daisy", "arrive": "16:09", "walk_start": "16:09",
         "walk_end": "16:29", "walk_minutes": 20,
         "comfort_min_f": 45, "comfort_max_f": 95},
    ]})
    banded_check = assistant_call("check_weather", {
        "lat": ZOO[0], "lon": ZOO[1], "date": "2026-09-11",
        "start_hour": 16, "end_hour": 17,
        "comfort_min_f": 45, "comfort_max_f": 95,
    })
    assert audit_weather_coverage([banded_check, call, result]) is None


def test_out_of_range_band_is_bounced():
    error = validate_call("check_weather", {
        "lat": 41.9, "lon": -87.6, "date": "2026-09-11",
        "comfort_min_f": -40,
    })
    assert "comfort_min_f" in error and "-40" in error


def test_oversized_buffer_is_bounced():
    stops = [{"name": "home", "lat": 0.0, "lon": 0.0},
             {"name": "Rex", "lat": 0.0, "lon": 0.1,
              "walk_minutes": 60, "buffer_minutes": 90}]
    error = validate_call("optimize_route", {"stops": stops})
    assert "buffer_minutes" in error and "90" in error


def test_null_optional_field_is_stripped_not_bounced():
    # a model that fills an unset optional with null (measured: qwen3-8b)
    # must not livelock -- null == omitted, so the call is legal
    stops = [{"name": "home", "lat": 0.0, "lon": 0.0},
             {"name": "Rex", "lat": 0.0, "lon": 0.1, "walk_minutes": 30,
              "max_relief_m": None, "buffer_minutes": None}]
    cleaned = scrub_optionals({"stops": stops})
    assert "max_relief_m" not in cleaned["stops"][1]
    assert "buffer_minutes" not in cleaned["stops"][1]
    assert validate_call("optimize_route", cleaned) is None


def test_zero_max_relief_is_stripped_not_bounced():
    # a model that writes max_relief_m: 0 to mean "no hill limit"
    # (measured: qwen3-8b, lakeview 0/5) must not livelock -- the schema
    # minimum is 1, so 0 == unset, so the call is legal without it
    stops = [{"name": "home", "lat": 0.0, "lon": 0.0},
             {"name": "Rex", "lat": 0.0, "lon": 0.1, "walk_minutes": 30,
              "max_relief_m": 0}]
    cleaned = scrub_optionals({"stops": stops})
    assert "max_relief_m" not in cleaned["stops"][1]
    assert validate_call("optimize_route", cleaned) is None


def test_scrub_keeps_real_values():
    # null and non-positive max_relief_m go; genuine zeros stay
    kept = scrub_optionals({"a": 0, "b": False, "c": None,
                            "buffer_minutes": 0, "max_relief_m": 0,
                            "d": [{"e": None, "f": 1, "max_relief_m": 30}]})
    assert kept == {"a": 0, "b": False, "buffer_minutes": 0,
                    "d": [{"f": 1, "max_relief_m": 30}]}


def test_call_args_strips_nulls_so_auditors_dont_crash():
    # a null optional in a raw check_weather call reached the weather
    # auditor as None and crashed float() (measured: two gpt-oss-120b
    # runs). _call_args must strip it the same way dispatch does.
    tc = {"function": {"name": "check_weather", "arguments":
          '{"lat": 41.9, "lon": -87.6, "comfort_min_f": null, "comfort_max_f": null}'}}
    args = _call_args(tc)
    assert "comfort_min_f" not in args and "comfort_max_f" not in args
    assert args["lat"] == 41.9
    # dict-dialect calls are stripped too
    tc2 = {"function": {"name": "optimize_route", "arguments":
           {"stops": [{"name": "Rex", "lat": 1.0, "lon": 2.0, "max_relief_m": None}]}}}
    assert "max_relief_m" not in _call_args(tc2)["stops"][0]


def test_feasibility_oracle_vetoes_rosy_plan_over_infeasible_route():
    msgs = [tool_result({"feasible": False,
                         "reason": "no single outing fits every morning/afternoon window"})]
    gap = audit_feasibility(msgs, {"feasible": True})
    assert gap and "infeasible" in gap


def test_feasibility_oracle_accepts_honest_infeasible():
    msgs = [tool_result({"feasible": False, "reason": "x"})]
    assert audit_feasibility(msgs, {"feasible": False}) is None


def test_feasibility_oracle_vetoes_false_alarm():
    # crying infeasible over a workable route (the weather-skip loophole)
    msgs = [tool_result({"feasible": True, "timeline": []})]
    assert "feasible" in audit_feasibility(msgs, {"feasible": False})


def test_feasibility_oracle_passes_when_no_route_yet():
    assert audit_feasibility([], {"feasible": True}) is None


def test_terrain_gap_when_sensitive_dog_unchecked():
    call = assistant_call("optimize_route", {
        "stops": [
            {"name": "home", "lat": 41.948, "lon": -87.655},
            {"name": "Cliff", "lat": ZOO[0], "lon": ZOO[1],
             "walk_minutes": 20, "max_relief_m": 15},
        ],
        "start_time": "13:00",
    })
    result = tool_result({"feasible": True, "timeline": [
        {"stop": "Cliff", "walk_start": "13:09", "walk_end": "13:29",
         "walk_minutes": 20, "max_relief_m": 15},
    ]})
    gap = audit_terrain_coverage([call, result])
    assert gap and "Cliff" in gap and "max_relief_m=15" in gap


def test_terrain_coverage_satisfied():
    call = assistant_call("optimize_route", {
        "stops": [
            {"name": "home", "lat": 41.948, "lon": -87.655},
            {"name": "Cliff", "lat": ZOO[0], "lon": ZOO[1],
             "walk_minutes": 20, "max_relief_m": 15},
        ],
        "start_time": "13:00",
    })
    terrain = assistant_call("check_terrain", {
        "lat": ZOO[0], "lon": ZOO[1], "max_relief_m": 15})
    assert audit_terrain_coverage([call, terrain]) is None


# ---------------------------------------------------------------------
# the daylight oracle
# ---------------------------------------------------------------------


def test_daylight_gap_when_afternoon_dog_and_no_sunset():
    call = assistant_call("optimize_route", {
        "stops": [
            {"name": "home", "lat": 41.948, "lon": -87.655},
            {"name": "Dusk", "lat": ZOO[0], "lon": ZOO[1],
             "walk_minutes": 60, "walk_window": "afternoon"},
        ],
    })
    gap = audit_daylight([call])
    assert gap and "Dusk" in gap and "check_daylight" in gap


def test_daylight_satisfied_when_checked_and_sunset_passed():
    daylight = assistant_call("check_daylight", {
        "lat": 41.948, "lon": -87.655, "date": "2026-12-15"})
    call = assistant_call("optimize_route", {
        "stops": [
            {"name": "home", "lat": 41.948, "lon": -87.655},
            {"name": "Dusk", "lat": ZOO[0], "lon": ZOO[1],
             "walk_minutes": 60, "walk_window": "afternoon"},
        ],
        "sunset_min": 989,
    })
    assert audit_daylight([daylight, call]) is None


def test_daylight_not_required_without_an_afternoon_dog():
    # morning / any dogs are scheduled early, so darkness can't bite
    call = assistant_call("optimize_route", {
        "stops": [
            {"name": "home", "lat": 41.948, "lon": -87.655},
            {"name": "Sunny", "lat": ZOO[0], "lon": ZOO[1],
             "walk_minutes": 30, "walk_window": "morning"},
        ],
    })
    assert audit_daylight([call]) is None


# ---------------------------------------------------------------------
# the rain oracle
# ---------------------------------------------------------------------


def _rain_context(max_precip_mm: float):
    """optimize_route (a skip_rain dog) + its wet/dry weather check."""
    route = assistant_call("optimize_route", {"stops": [
        {"name": "home", "lat": 41.948, "lon": -87.655},
        {"name": "Puddle", "lat": ZOO[0], "lon": ZOO[1],
         "walk_minutes": 30, "skip_rain": True}]})
    weather = assistant_call("check_weather", {"lat": ZOO[0], "lon": ZOO[1]})
    result = {"role": "tool", "tool_call_id": "c1",
              "content": json.dumps({"window": {"max_precip_mm": max_precip_mm}})}
    return [route, weather, result]


def test_rain_refuser_in_rain_needs_a_minimal_visit():
    ctx = _rain_context(3.0)  # wet
    full = {"walks": [{"pet": "Puddle", "walk_start": "13:00", "walk_end": "13:30"}]}
    gap = audit_rain(ctx, full)
    assert gap and "Puddle" in gap


def test_rain_refuser_with_a_minimal_visit_passes():
    ctx = _rain_context(3.0)  # wet
    visit = {"walks": [{"pet": "Puddle", "walk_start": "13:00", "walk_end": "13:10"}]}
    assert audit_rain(ctx, visit) is None


def test_rain_refuser_on_a_dry_day_walks_normally():
    ctx = _rain_context(0.0)  # dry
    full = {"walks": [{"pet": "Puddle", "walk_start": "13:00", "walk_end": "13:30"}]}
    assert audit_rain(ctx, full) is None


# ---------------------------------------------------------------------
# synthesis oracles: derived priority, combined verdict
# ---------------------------------------------------------------------


def _route_with(dog):
    return assistant_call("optimize_route", {"stops": [
        {"name": "home", "lat": 41.948, "lon": -87.655}, dog]})


def test_priority_veto_when_missing_or_wrong():
    # a meds dog's priority is 2; missing -> veto
    dog = {"name": "Doc", "lat": ZOO[0], "lon": ZOO[1], "walk_minutes": 30,
           "needs_meds": True}
    gap = audit_priority([_route_with(dog)])
    assert gap and "Doc" in gap and "2" in gap
    dog["priority"] = 5  # wrong
    assert audit_priority([_route_with(dog)]) is not None


def test_priority_satisfied_when_correct():
    # meds (+2) + afternoon (+1) + narrow band (+2) = 5
    dog = {"name": "Doc", "lat": ZOO[0], "lon": ZOO[1], "walk_minutes": 30,
           "needs_meds": True, "walk_window": "afternoon",
           "comfort_min_f": 55, "comfort_max_f": 75, "priority": 5}
    assert audit_priority([_route_with(dog)]) is None


def test_verdict_must_be_worst_of_weather_and_terrain():
    route = assistant_call("optimize_route", {"stops": [
        {"name": "home", "lat": 41.948, "lon": -87.655},
        {"name": "Hilly", "lat": ZOO[0], "lon": ZOO[1], "walk_minutes": 30}]})
    w = call_with_id("check_weather", {"lat": ZOO[0], "lon": ZOO[1]}, "w1")
    wr = result_for("w1", {"verdict": "OK", "window": {}})
    t = call_with_id("check_terrain", {"lat": ZOO[0], "lon": ZOO[1]}, "t1")
    tr = result_for("t1", {"verdict": "AVOID"})
    msgs = [route, w, wr, t, tr]
    # weather OK + terrain AVOID -> the plan verdict must be DO_NOT_WALK
    bad = {"walks": [{"pet": "Hilly", "verdict": "OK",
                      "walk_start": "13:00", "walk_end": "13:30"}]}
    gap = audit_verdict(msgs, bad)
    assert gap and "DO_NOT_WALK" in gap
    good = {"walks": [{"pet": "Hilly", "verdict": "DO_NOT_WALK",
                       "walk_start": "13:00", "walk_end": "13:30"}]}
    assert audit_verdict(msgs, good) is None


# ---------------------------------------------------------------------
# chat(): latency diagnostics
# ---------------------------------------------------------------------


def test_chat_logs_each_attempt_with_status_and_backoff(monkeypatch):
    # every HTTP attempt is timed and labeled, and a 429's backoff is kept
    # with it, so throttling time is never mistaken for model time
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(agent.time, "sleep", _no_sleep)
    good = {"id": "gen-abc", "choices": [{"message": {"role": "assistant",
            "content": "ok", "tool_calls": []}}], "usage": {}}
    calls = iter([_http_429(retry_after=7), None])

    def fake_urlopen(_req, timeout=None):
        nxt = next(calls)
        if isinstance(nxt, Exception):
            raise nxt
        return io.BytesIO(json.dumps(good).encode())

    monkeypatch.setattr(agent.urllib.request, "urlopen", fake_urlopen)
    meta = chat("some/model", [{"role": "user", "content": "hi"}])["_meta"]
    log = meta["attempts_log"]
    assert [a["status"] for a in log] == ["http_429", "ok"]
    assert log[0]["backoff_s"] == 7
    assert log[0]["retry_after"] == 7
    assert meta["generation_id"] == "gen-abc"
    assert all("total_s" in a for a in log)


def test_chat_failure_carries_its_attempts(monkeypatch):
    # a call that gives up still spent real time; the log must survive
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(agent.time, "sleep", _no_sleep)

    def fake_urlopen(_req, timeout=None):
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(agent.urllib.request, "urlopen", fake_urlopen)
    with __import__("pytest").raises(agent.ChatFailed) as exc:
        chat("some/model", [{"role": "user", "content": "hi"}])
    assert len(exc.value.attempts_log) == agent.CHAT_ATTEMPTS
    assert {a["status"] for a in exc.value.attempts_log} == {"timeout"}
    assert "model backend unusable" in str(exc.value)   # classify() still matches


def test_call_log_writes_start_before_the_request_returns(monkeypatch, tmp_path):
    # the point of the live log: a hung call shows as a start with no end
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    log_path = tmp_path / "calls.jsonl"
    monkeypatch.setattr(agent, "CALL_LOG", log_path)
    monkeypatch.setattr(agent, "CALL_LOG_CONTEXT", {"n": 3})
    seen_during_request = []
    good = {"choices": [{"message": {"role": "assistant", "content": "ok",
                                     "tool_calls": []}}], "usage": {}}

    def fake_urlopen(_req, timeout=None):
        seen_during_request.append(log_path.read_text())
        return io.BytesIO(json.dumps(good).encode())

    monkeypatch.setattr(agent.urllib.request, "urlopen", fake_urlopen)
    chat("some/model", [{"role": "user", "content": "hi"}])
    assert '"attempt_start"' in seen_during_request[0]
    assert '"attempt_end"' not in seen_during_request[0]
    lines = [json.loads(x) for x in log_path.read_text().splitlines()]
    assert [x["event"] for x in lines] == ["attempt_start", "attempt_end"]
    assert all(x["n"] == 3 for x in lines)


def test_chat_retries_a_dropped_connection(monkeypatch):
    # IncompleteRead is not an OSError; it used to end the run on first drop
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(agent.time, "sleep", _no_sleep)
    good = {"choices": [{"message": {"role": "assistant", "content": "ok",
                                     "tool_calls": []}}], "usage": {}}
    calls = iter(["drop", "ok"])

    class Dropped(io.BytesIO):
        def read(self, *a):
            raise agent.http_client.IncompleteRead(b"\n   \n")

    def fake_urlopen(_req, timeout=None):
        if next(calls) == "drop":
            return Dropped()
        return io.BytesIO(json.dumps(good).encode())

    monkeypatch.setattr(agent.urllib.request, "urlopen", fake_urlopen)
    meta = chat("some/model", [{"role": "user", "content": "hi"}])["_meta"]
    assert [a["status"] for a in meta["attempts_log"]] == ["network:IncompleteRead", "ok"]


def test_chat_stops_at_once_when_reply_hit_the_token_cap(monkeypatch):
    # finish_reason=length + broken tool JSON is OUR cap, and a retry of the
    # same request would be cut off again: one attempt, labeled truncated
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(agent.time, "sleep", _no_sleep)
    cut = {"id": "gen-1", "choices": [{"finish_reason": "length", "message": {
        "role": "assistant", "tool_calls": [{"id": "1", "function": {
            "name": "check_weather", "arguments": '{"lat": 41.9, "lon":'}}]}}],
        "usage": {"completion_tokens": 2000}}
    opened = []

    def fake_urlopen(_req, timeout=None):
        opened.append(1)
        return io.BytesIO(json.dumps(cut).encode())

    monkeypatch.setattr(agent.urllib.request, "urlopen", fake_urlopen)
    with __import__("pytest").raises(agent.ChatFailed) as exc:
        chat("some/model", [{"role": "user", "content": "hi"}])
    assert len(opened) == 1
    assert "reply truncated at max_tokens" in str(exc.value)
    a = exc.value.attempts_log[0]
    assert a["finish_reason"] == "length" and a["completion_tokens"] == 2000


def test_chat_keeps_the_body_of_a_refused_request(monkeypatch):
    import email.message
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(agent.time, "sleep", _no_sleep)

    def fake_urlopen(_req, timeout=None):
        raise agent.urllib.error.HTTPError(
            "http://x", 404, "Not Found", email.message.Message(),
            io.BytesIO(b'{"error":{"message":"No endpoints found"}}'))

    monkeypatch.setattr(agent.urllib.request, "urlopen", fake_urlopen)
    with __import__("pytest").raises(agent.ChatFailed) as exc:
        chat("some/model", [{"role": "user", "content": "hi"}])
    assert "No endpoints found" in str(exc.value)
    assert exc.value.attempts_log[0]["status"] == "http_404"


def test_budget_exhaustion_says_so(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with __import__("pytest").raises(agent.ChatFailed) as exc:
        chat("some/model", [{"role": "user", "content": "hi"}],
             deadline=agent.time.monotonic() - 1)
    assert "run budget exhausted before attempt 1" in str(exc.value)


def test_cap_detected_from_native_reason_when_openrouter_normalizes_it(monkeypatch):
    # measured 2026-09-17: finish_reason "tool_calls", native "length"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(agent.time, "sleep", _no_sleep)
    cut = {"id": "g", "choices": [{"finish_reason": "tool_calls",
           "native_finish_reason": "length", "message": {"role": "assistant",
           "tool_calls": [{"id": "1", "function": {"name": "check_weather",
                                                   "arguments": '{"lat": 4'}}]}}]}
    monkeypatch.setattr(agent.urllib.request, "urlopen",
                        lambda _r, timeout=None: io.BytesIO(json.dumps(cut).encode()))
    with __import__("pytest").raises(agent.ChatFailed) as exc:
        chat("some/model", [{"role": "user", "content": "hi"}])
    assert "reply truncated" in str(exc.value)
    assert len(exc.value.attempts_log) == 1


def test_parseable_but_cut_off_reply_is_flagged(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    ok_but_cut = {"choices": [{"finish_reason": "length", "message": {
        "role": "assistant", "content": "", "tool_calls": []}}], "usage": {}}
    monkeypatch.setattr(agent.urllib.request, "urlopen",
                        lambda _r, timeout=None: io.BytesIO(json.dumps(ok_but_cut).encode()))
    meta = chat("some/model", [{"role": "user", "content": "hi"}])["_meta"]
    assert meta["attempts_log"][0]["cap_hit"] is True
