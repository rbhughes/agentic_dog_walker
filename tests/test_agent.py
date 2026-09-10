"""Agent internals, offline: the referee and the auditor.

The loop itself needs a model; these test the deterministic parts
that make the loop trustworthy. Same rule as test_toolbox: one
behavioral claim per test, no network, no model.
"""

import json

from dog_walker.agent import audit_weather_coverage, validate_call

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
