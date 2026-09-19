"""Offline tests for the measurement harness's pure logic: the Wilson
interval, the failure classifier, and the aggregation. No model, no
network -- these prove the MEASUREMENT is sound independent of any run.
"""

from dog_walker.measure import (
    classify,
    dominant_failure,
    round_breakdown,
    summarize,
    wilson,
)


def test_round_breakdown_traces_corrections_to_round_type():
    records = [
        {"model": "m", "passed": True, "outcome": "pass",
         "calls_by_tool": {"geocode_addresses": 1, "optimize_route": 2,
                           "submit_plan": 1},
         "bounces_by_tool": {"optimize_route": 1},
         "vetoes_by_auditor": {"weather": 1}, "fail_after": None},
        {"model": "m", "passed": False, "outcome": "malformed_output",
         "calls_by_tool": {"geocode_addresses": 1, "optimize_route": 1},
         "bounces_by_tool": {}, "vetoes_by_auditor": {},
         "fail_after": "optimize_route"},
    ]
    rows = {r["round"]: r for r in round_breakdown(records)["rows"]}
    assert rows["geocode_addresses"] == {
        "round": "geocode_addresses", "attempts": 2, "corrections": 0, "rate": 0.0}
    assert rows["optimize_route"]["attempts"] == 3
    assert rows["optimize_route"]["corrections"] == 1  # one schema bounce
    # a veto is a rejected submit_plan, so it counts as a submit correction
    assert rows["submit_plan"]["attempts"] == 1
    assert rows["submit_plan"]["corrections"] == 1
    b = round_breakdown(records)
    assert b["malformed_output"] == 1
    assert b["fail_after"] == {"optimize_route": 1}


# --- Wilson score interval -------------------------------------------


def test_wilson_perfect_score_is_not_certainty():
    # 5/5 is a point estimate of 100%, but the interval must admit doubt
    lo, hi = wilson(5, 5)
    assert hi == 1.0
    assert lo < 0.6          # only 5 samples: the floor is well under 60%


def test_wilson_narrows_with_more_samples():
    lo5, _ = wilson(5, 5)
    lo50, _ = wilson(50, 50)
    assert lo50 > lo5        # 50/50 is far more confident than 5/5


def test_wilson_empty_is_zero_zero():
    assert wilson(0, 0) == (0.0, 0.0)


def test_wilson_stays_in_unit_interval():
    for passes in range(0, 6):
        lo, hi = wilson(passes, 5)
        assert 0.0 <= lo <= hi <= 1.0


# --- failure classification ------------------------------------------


def test_dominant_failure_prefers_the_biggest_struggle():
    assert dominant_failure(bounces=1, vetoes=9, nudges=0) == "veto_livelock"
    assert dominant_failure(bounces=7, vetoes=0, nudges=1) == "schema_thrash"
    assert dominant_failure(bounces=0, vetoes=0, nudges=2) == "prose_stall"
    assert dominant_failure(bounces=0, vetoes=0, nudges=0) == "no_submit"


def test_classify_pass_on_matching_feasibility():
    final = {"plan": {"feasible": True}}
    assert classify(final, None, True, 0, 0, 0) == "pass"
    final_false = {"plan": {"feasible": False}}
    assert classify(final_false, None, False, 0, 0, 0) == "pass"


def test_classify_fabrication_is_its_own_category():
    # scenario is IMPOSSIBLE (expected False) but the model claimed True
    final = {"plan": {"feasible": True}}
    assert classify(final, None, False, 0, 0, 0) == "fabricated_feasible"


def test_classify_false_alarm_is_distinct():
    final = {"plan": {"feasible": False}}
    assert classify(final, None, True, 0, 0, 0) == "false_infeasible"


def test_classify_backend_error_vs_loop_exhaustion():
    exhausted = {"message": "no submit_plan within 14 rounds; last message: x"}
    assert classify(None, exhausted, True, 0, 5, 0) == "veto_livelock"
    crashed = {"message": "RuntimeError: backend unreachable"}
    assert classify(None, crashed, True, 0, 0, 0) == "backend_error"
    slow = {"message": "harness deadline 240s"}
    assert classify(None, slow, True, 0, 0, 0) == "timeout"


def test_classify_relay_fidelity_when_feasibility_is_not_fixed():
    # expected_feasible None (stochastic/date-dependent scenario): any
    # validated final is a pass, feasible True or False
    assert classify({"plan": {"feasible": True}}, None, None, 0, 0, 0) == "pass"
    assert classify({"plan": {"feasible": False}}, None, None, 0, 0, 0) == "pass"
    # but a run that never finishes still fails
    exhausted = {"message": "no submit_plan within 14 rounds; x"}
    assert classify(None, exhausted, None, 0, 0, 2) == "prose_stall"


# --- aggregation ------------------------------------------------------


def test_summarize_counts_passes_and_buckets_failures():
    records = [
        {"passed": True, "outcome": "pass", "cost": 0.01,
         "seconds": 10, "rounds": 4},
        {"passed": True, "outcome": "pass", "cost": 0.02,
         "seconds": 20, "rounds": 6},
        {"passed": False, "outcome": "veto_livelock", "cost": 0.05,
         "seconds": 40, "rounds": 14},
    ]
    agg = summarize(records)
    assert agg["n"] == 3
    assert agg["passes"] == 2
    assert agg["pass_rate"] == round(2 / 3, 3)
    assert agg["failures"] == {"veto_livelock": 1}
    assert agg["median_cost"] == 0.02
    assert agg["total_cost"] == 0.08


def test_summarize_empty_is_safe():
    agg = summarize([])
    assert agg["n"] == 0
    assert agg["pass_rate"] == 0.0
    assert agg["failures"] == {}


def test_summarize_drops_provider_faults_from_denominator():
    # a timeout is the provider's fault: it must not drag the pass rate down.
    # two clean passes + one provider timeout = 2/2, not 2/3.
    records = [
        {"passed": True, "outcome": "pass", "cost": 0.01, "seconds": 10,
         "rounds": 4, "provider_excluded": False, "retries": 0},
        {"passed": True, "outcome": "pass", "cost": 0.02, "seconds": 20,
         "rounds": 5, "provider_excluded": False, "retries": 0},
        {"passed": False, "outcome": "timeout", "cost": 0.0, "seconds": 90,
         "rounds": 0, "provider_excluded": True, "retries": 2},
    ]
    agg = summarize(records)
    assert agg["n"] == 2                      # denominator excludes the fault
    assert agg["passes"] == 2
    assert agg["pass_rate"] == 1.0
    assert agg["attempted"] == 3
    assert agg["provider_excluded"] == 1
    assert agg["failures"] == {}              # the timeout is not a competence fail
    # the excluded run's 90s hang must not skew the latency median
    assert agg["median_seconds"] == 15.0


def test_summarize_counts_flakiness_across_all_runs():
    # retries are counted even on runs that ultimately passed: a model that
    # only got there after retries is a different reliability story.
    records = [
        {"passed": True, "outcome": "pass", "cost": 0.01, "seconds": 10,
         "rounds": 4, "provider_excluded": False, "retries": 2},
        {"passed": True, "outcome": "pass", "cost": 0.02, "seconds": 20,
         "rounds": 5, "provider_excluded": False, "retries": 0},
    ]
    agg = summarize(records)
    assert agg["total_retries"] == 2
    assert agg["flaky_runs"] == 1
    assert agg["pass_rate"] == 1.0


# ---------------------------------------------------------------------
# latency attribution
# ---------------------------------------------------------------------

from dog_walker.measure import (attach_generation_stats, time_breakdown,  # noqa: E402
                                timing_summary)


def _calls():
    return [
        {"kind": "chat", "wall_s": 40.0, "attempts": [
            {"status": "http_429", "total_s": 1.0, "backoff_s": 7.0},
            {"status": "ok", "total_s": 30.0, "generation_id": "g1"}]},
        {"kind": "tool", "name": "optimize_route", "seconds": 5.0},
        {"kind": "chat", "wall_s": 20.0, "attempts": [
            {"status": "http_502", "total_s": 2.0, "backoff_s": 4.0},
            {"status": "unusable:JSONDecodeError", "total_s": 3.0, "backoff_s": 8.0},
            {"status": "ok", "total_s": 10.0, "generation_id": "g2"}]},
    ]


def test_time_breakdown_buckets_attempts_and_backoffs():
    t = time_breakdown(75.0, _calls())
    assert t["model_s"] == 40.0
    assert t["throttle_s"] == 8.0          # the 429 AND its wait
    assert t["provider_fail_s"] == 6.0
    assert t["malformed_s"] == 11.0
    assert t["tool_s"] == 5.0
    assert t["other_s"] == 5.0             # remainder: our loop + auditors
    assert t["chat_calls"] == 2 and t["http_attempts"] == 5
    assert t["slow_run"] is False


def test_time_breakdown_flags_slow_runs_and_never_goes_negative():
    t = time_breakdown(125.0, [])
    assert t["slow_run"] is True and t["other_s"] == 125.0
    assert time_breakdown(10.0, _calls())["other_s"] == 0.0


def test_generation_stats_roll_up_and_expose_hidden_failovers():
    rec = {"calls": _calls(), "timing": time_breakdown(75.0, _calls())}

    def fake_fetch(ids):
        assert ids == ["g1", "g2"]
        return {"g1": {"first_token_s": 5.0, "generation_s": 28.0, "provider": "A",
                       "completion_tokens": 460, "reasoning_tokens": 400,
                       "finish_reason": "tool_calls", "cancelled": False,
                       "provider_attempts": [{"provider": "B", "status": 503, "latency_s": 1},
                                             {"provider": "A", "status": 200, "latency_s": 5}]}}

    attach_generation_stats(rec, fetch=fake_fetch)
    o = rec["timing"]["openrouter"]
    assert o["stats_found"] == 1 and o["stats_missing"] == 1
    assert o["openrouter_overhead_s"] == 2.0     # 30s wall - 28s generation
    assert o["decode_s"] == 23.0
    assert o["tokens_per_s"] == 20.0
    assert o["hidden_provider_failures"] == 1
    assert rec["calls"][0]["attempts"][1]["openrouter"]["provider"] == "A"


def test_timing_summary_skips_runs_recorded_before_diagnostics():
    new = {"model": "m", "seconds": 75.0, "timing": time_breakdown(75.0, _calls())}
    old = {"model": "m", "seconds": 999.0}
    t = timing_summary([new, old])
    assert t["runs"] == 1 and t["total_s"] == 75.0
    assert abs(sum(t["shares"].values()) - 1.0) < 0.01


# ---------------------------------------------------------------------
# attribution: truncation, budget, blame
# ---------------------------------------------------------------------

from dog_walker.measure import blame  # noqa: E402


def test_classify_separates_cap_truncation_and_budget_from_model_errors():
    trunc = {"message": "ChatFailed: reply truncated at max_tokens=700 (3s): "
                        "JSONDecodeError: Unterminated string"}
    budget = {"message": "ChatFailed: model backend unusable after 6 attempts "
                         "(0s, 0 throttled): RuntimeError: run budget exhausted before attempt 1"}
    assert classify(None, trunc, True, 0, 0, 0) == "truncated"
    assert classify(None, budget, True, 0, 0, 0) == "budget_exhausted"


def test_blame_follows_the_agreed_rule():
    slow_model = {"model_s": 250.0, "provider_fail_s": 20.0, "throttle_s": 0.0, "tool_s": 10.0}
    flaky = {"model_s": 60.0, "provider_fail_s": 200.0, "throttle_s": 30.0, "tool_s": 10.0}
    assert blame("pass", slow_model) == "none"
    assert blame("pass", slow_model, host_sleep_s=900) == "host"
    assert blame("malformed_output", {}) == "model"
    assert blame("truncated", {}, cap_tokens=700) == "harness"
    assert blame("truncated", {}, cap_tokens=4000) == "model"
    assert blame("backend_error", {}) == "provider"
    assert blame("budget_exhausted", slow_model) == "model"
    assert blame("timeout", flaky) == "provider"
    assert blame("timeout", {"model_s": 20.0, "tool_s": 280.0}) == "harness"


def test_a_run_on_straight_line_distances_is_not_graded():
    # feasibility is what a run is graded on; with ORS down it is fiction
    assert blame("pass", {}, degraded_routes=1) == "harness"
    assert blame("fabricated_feasible", {}, degraded_routes=2) == "harness"
    assert blame("fabricated_feasible", {}, degraded_routes=0) == "model"
