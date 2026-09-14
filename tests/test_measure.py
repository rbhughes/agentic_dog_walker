"""Offline tests for the measurement harness's pure logic: the Wilson
interval, the failure classifier, and the aggregation. No model, no
network -- these prove the MEASUREMENT is sound independent of any run.
"""

from dog_walker.measure import (
    classify,
    dominant_failure,
    summarize,
    wilson,
)


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
