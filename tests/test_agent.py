"""Agent internals, offline: the referee and the auditor.

The loop itself needs a model; these test the deterministic parts
that make the loop trustworthy. Same rule as test_toolbox: one
behavioral claim per test, no network, no model.
"""

import json

from dog_walker.agent import (audit_feasibility, audit_terrain_coverage, audit_weather_coverage, validate_call, without_nulls)

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
    cleaned = without_nulls({"stops": stops})
    assert "max_relief_m" not in cleaned["stops"][1]
    assert "buffer_minutes" not in cleaned["stops"][1]
    assert validate_call("optimize_route", cleaned) is None


def test_without_nulls_keeps_real_values():
    kept = without_nulls({"a": 0, "b": False, "c": None, "d": [{"e": None, "f": 1}]})
    assert kept == {"a": 0, "b": False, "d": [{"f": 1}]}


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
