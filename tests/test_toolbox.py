"""Toolbox behavior, one claim per test. Read top-to-bottom as docs.

All tests run OFFLINE: network calls are faked with monkeypatch, so
these verify OUR logic (ladders, bumps, windowing, caching, error
slots), never Nominatim's or Open-Meteo's uptime.

Run:  uv run --with pytest python -m pytest tests/ -v
"""

import pytest

from dog_walker import toolbox
from dog_walker.toolbox import (
    VERDICTS,
    _walk_ladder,
    assess_walk_safety,
    bump,
    check_weather,
    geocode_addresses,
)

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def day(**overrides):
    """A synthetic 24-hour forecast: mild everywhere unless overridden."""
    base = {
        "time": [f"2026-09-07T{h:02d}:00" for h in range(24)],
        "temp_c": [15.0] * 24,
        "feels_like_c": [15.0] * 24,
        "precip_mm": [0.0] * 24,
        "wind_kph": [10.0] * 24,
    }
    base.update(overrides)
    return base


class FakeResponse:
    """Just enough of a requests.Response for the geocoder."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _fresh_geocode_cache():
    """The cache is module-level state; isolate every test from it."""
    toolbox._geocode_cache.clear()
    yield
    toolbox._geocode_cache.clear()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Nominatim politeness (1.1s spacing) is right in production and
    wrong in tests: strip it so the suite stays instant."""
    monkeypatch.setattr("time.sleep", lambda s: None)


# ---------------------------------------------------------------------
# the primitives: bump and _walk_ladder
# ---------------------------------------------------------------------


def test_bump_moves_one_rung_toward_severe():
    assert bump("OK") == "CAUTION"
    assert bump("SHORTEN") == "DO_NOT_WALK"


def test_bump_clamps_at_the_top():
    # stacked escalations can't overflow the scale
    assert bump("DO_NOT_WALK") == "DO_NOT_WALK"
    assert bump("SHORTEN", rungs=5) == "DO_NOT_WALK"


def test_ladder_returns_most_severe_triggered_rung():
    ladder = [(20, "CAUTION"), (30, "SHORTEN"), (35, "DO_NOT_WALK")]
    assert _walk_ladder(32, ladder) == (30, "SHORTEN")


def test_ladder_returns_none_when_nothing_triggers():
    ladder = [(20, "CAUTION"), (30, "SHORTEN")]
    assert _walk_ladder(15, ladder) is None


def test_cold_ladder_compares_downward():
    # colder_is_worse flips the comparison: value <= threshold triggers
    ladder = [(-5, "CAUTION"), (-10, "SHORTEN")]
    assert _walk_ladder(-12, ladder, colder_is_worse=True) == (-10, "SHORTEN")
    assert _walk_ladder(-3, ladder, colder_is_worse=True) is None


# ---------------------------------------------------------------------
# assess_walk_safety: ladders
# ---------------------------------------------------------------------


def test_mild_day_is_ok_with_no_reasons():
    result = assess_walk_safety(day(), 8, 20)
    assert result["verdict"] == "OK"
    assert result["reasons"] == []


def test_extreme_cold_forbids_the_walk():
    # the Phase-1 fixture blizzard: feels-like -33
    result = assess_walk_safety(day(feels_like_c=[-33.0] * 24), 8, 20)
    assert result["verdict"] == "DO_NOT_WALK"
    assert any("feels-like low" in r for r in result["reasons"])


def test_heat_keys_on_feels_like_not_air_temp():
    # humid 32C feels like 36: air temp alone would miss DO_NOT_WALK
    result = assess_walk_safety(
        day(temp_c=[32.0] * 24, feels_like_c=[36.0] * 24), 8, 20
    )
    assert result["verdict"] == "DO_NOT_WALK"


def test_strictest_ladder_wins_and_all_reasons_report():
    # cold at CAUTION (-6) plus wind at SHORTEN (42): verdict takes the
    # worst, but BOTH rules explain themselves
    result = assess_walk_safety(
        day(feels_like_c=[-6.0] * 24, wind_kph=[42.0] * 24), 8, 20
    )
    assert result["verdict"] == "SHORTEN"
    assert len(result["reasons"]) == 2


# ---------------------------------------------------------------------
# assess_walk_safety: the window is the point
# ---------------------------------------------------------------------


def test_cold_dawn_does_not_cancel_a_warm_afternoon():
    # -25 until 8am, 15C after: a 2-5pm walk never sees the dawn.
    # (The 2024 tool averaged the whole day and got this wrong.)
    feels = [-25.0] * 8 + [15.0] * 16
    result = assess_walk_safety(day(feels_like_c=feels), 14, 17)
    assert result["verdict"] == "OK"
    assert result["window"]["min_feels_like_c"] == 15.0


def test_window_stats_report_the_worst_hour_in_window():
    precip = [0.0] * 24
    precip[15] = 22.0  # one violent hour inside the window
    result = assess_walk_safety(day(precip_mm=precip), 8, 20)
    assert result["window"]["max_precip_mm"] == 22.0
    assert result["verdict"] == "SHORTEN"


def test_empty_window_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="no forecast hours"):
        assess_walk_safety(day(), 20, 8)  # inverted window


# ---------------------------------------------------------------------
# assess_walk_safety: combo escalations
# ---------------------------------------------------------------------


def test_wet_cold_bumps_a_verdict_the_numbers_alone_miss():
    # +1C drizzle: no ladder fires (base OK), but wet coat at
    # near-freezing defeats insulation -> CAUTION
    result = assess_walk_safety(
        day(feels_like_c=[1.0] * 24, precip_mm=[1.0] * 24), 8, 20
    )
    assert result["verdict"] == "CAUTION"
    assert any("wet-cold" in r for r in result["reasons"])


def test_escalation_still_reports_when_already_at_the_top():
    # -33 and raining: verdict can't get worse than DO_NOT_WALK, but
    # the wet-cold reason still appears -- the report stays complete
    result = assess_walk_safety(
        day(feels_like_c=[-33.0] * 24, precip_mm=[2.0] * 24), 8, 20
    )
    assert result["verdict"] == "DO_NOT_WALK"
    assert any("wet-cold" in r for r in result["reasons"])


# ---------------------------------------------------------------------
# geocode_addresses
# ---------------------------------------------------------------------


def test_geocode_parses_a_hit(monkeypatch):
    monkeypatch.setattr(
        "requests.get",
        lambda *a, **k: FakeResponse(
            [{"lat": "51.0443", "lon": "-114.0631", "display_name": "Calgary Tower"}]
        ),
    )
    result = geocode_addresses(["Calgary Tower"])
    hit = result["results"][0]
    assert hit["lat"] == pytest.approx(51.0443)
    assert hit["display_name"] == "Calgary Tower"


def test_geocode_failure_lands_in_its_slot_not_as_an_exception(monkeypatch):
    # one bad address must not torch the good one's result
    monkeypatch.setattr(
        "requests.get",
        lambda url, params, **k: FakeResponse(
            [] if "zzzz" in params["q"] else [{"lat": "1", "lon": "2"}]
        ),
    )
    result = geocode_addresses(["real place", "zzzz"])
    assert result["results"][0]["lat"] == 1.0
    assert result["results"][1] == {"address": "zzzz", "error": "not found"}


def test_geocode_cache_answers_repeats_without_the_network(monkeypatch):
    calls = []

    def counting_get(*a, **k):
        calls.append(1)
        return FakeResponse([{"lat": "1", "lon": "2"}])

    monkeypatch.setattr("requests.get", counting_get)
    geocode_addresses(["Somewhere"])
    geocode_addresses(["  somewhere  "])  # same place, sloppier typing
    assert len(calls) == 1


def test_geocode_refuses_oversized_batches():
    result = geocode_addresses(["x"] * 13)
    assert "error" in result
    assert "max 12" in result["error"]


# ---------------------------------------------------------------------
# check_weather composes fetch + assess
# ---------------------------------------------------------------------


def test_check_weather_is_fetch_then_assess(monkeypatch):
    monkeypatch.setattr(
        "dog_walker.toolbox.fetch_forecast",
        lambda lat, lon, date: day(feels_like_c=[-33.0] * 24),
    )
    result = check_weather(51.0, -114.0, "2026-09-07", 8, 20)
    assert result["verdict"] == "DO_NOT_WALK"
