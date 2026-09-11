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
        "temp_f": [59.0] * 24,
        "feels_like_f": [59.0] * 24,
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


def _sleep_noop(seconds: float) -> None:
    """Stand-in for time.sleep: do nothing, instantly."""


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Nominatim politeness (1.1s spacing) is right in production and
    wrong in tests: strip it so the suite stays instant."""
    monkeypatch.setattr("time.sleep", _sleep_noop)


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
    # the fixture blizzard: feels-like -27F, 47F below the default
    # comfort minimum
    result = assess_walk_safety(day(feels_like_f=[-27.0] * 24), 8, 20)
    assert result["verdict"] == "DO_NOT_WALK"
    assert any("below comfort minimum" in r for r in result["reasons"])


def test_heat_keys_on_feels_like_not_air_temp():
    # humid 90F feels like 97: 13F above the default band maximum ->
    # SHORTEN (air temp alone, 6F above, would only say CAUTION)
    result = assess_walk_safety(
        day(temp_f=[90.0] * 24, feels_like_f=[97.0] * 24), 8, 20
    )
    assert result["verdict"] == "SHORTEN"


def test_strictest_ladder_wins_and_all_reasons_report():
    # cold at CAUTION (18F, under the 20F rung) plus wind at SHORTEN
    # (42 kph): verdict takes the worst, but BOTH rules explain themselves
    result = assess_walk_safety(
        day(feels_like_f=[18.0] * 24, wind_kph=[42.0] * 24), 8, 20
    )
    assert result["verdict"] == "SHORTEN"
    assert len(result["reasons"]) == 2


# ---------------------------------------------------------------------
# assess_walk_safety: the window is the point
# ---------------------------------------------------------------------


def test_cold_dawn_does_not_cancel_a_warm_afternoon():
    # -13F until 8am, 59F after: a 2-5pm walk never sees the dawn.
    # (The 2024 tool averaged the whole day and got this wrong.)
    feels = [-13.0] * 8 + [59.0] * 16
    result = assess_walk_safety(day(feels_like_f=feels), 14, 17)
    assert result["verdict"] == "OK"
    assert result["window"]["min_feels_like_f"] == 59.0


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
    # 34F drizzle: no ladder fires (base OK), but wet coat at
    # near-freezing defeats insulation -> CAUTION
    result = assess_walk_safety(
        day(feels_like_f=[34.0] * 24, precip_mm=[1.0] * 24), 8, 20
    )
    assert result["verdict"] == "CAUTION"
    assert any("wet-cold" in r for r in result["reasons"])


def test_escalation_still_reports_when_already_at_the_top():
    # -27F and raining: verdict can't get worse than DO_NOT_WALK, but
    # the wet-cold reason still appears -- the report stays complete
    result = assess_walk_safety(
        day(feels_like_f=[-27.0] * 24, precip_mm=[2.0] * 24), 8, 20
    )
    assert result["verdict"] == "DO_NOT_WALK"
    assert any("wet-cold" in r for r in result["reasons"])


# ---------------------------------------------------------------------
# geocode_addresses
# ---------------------------------------------------------------------


def test_geocode_parses_a_hit(monkeypatch):
    def fake_get(url, **kwargs):
        return FakeResponse(
            [{"lat": "51.0443", "lon": "-114.0631", "display_name": "Calgary Tower"}]
        )

    monkeypatch.setattr("requests.get", fake_get)
    result = geocode_addresses(["Calgary Tower"])
    hit = result["results"][0]
    assert hit["lat"] == pytest.approx(51.0443)
    assert hit["display_name"] == "Calgary Tower"


def test_geocode_failure_lands_in_its_slot_not_as_an_exception(monkeypatch):
    # one bad address must not torch the good one's result
    def fake_get(url, params, **kwargs):
        if "zzzz" in params["q"]:
            return FakeResponse([])  # Nominatim's "no match" shape
        return FakeResponse([{"lat": "1", "lon": "2"}])

    monkeypatch.setattr("requests.get", fake_get)
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


def test_within_one_hour_window_is_forgiven(monkeypatch):
    # 13:29-13:59 floors to [13, 13) -- crashed two models live;
    # an empty window now means "that hour"
    def fake_forecast(lat, lon, date):
        return day()

    monkeypatch.setattr("dog_walker.toolbox.fetch_forecast", fake_forecast)
    result = check_weather(41.9, -87.6, "2026-09-11", 13, 13)
    assert result["verdict"] == "OK"
    assert result["window"]["end_hour"] == 14


def test_check_weather_is_fetch_then_assess(monkeypatch):
    def fake_forecast(lat, lon, date):
        return day(feels_like_f=[-27.0] * 24)

    monkeypatch.setattr("dog_walker.toolbox.fetch_forecast", fake_forecast)
    result = check_weather(51.0, -114.0, "2026-09-07", 8, 20)
    assert result["verdict"] == "DO_NOT_WALK"


# ---------------------------------------------------------------------
# route optimizer: the offline parts
# ---------------------------------------------------------------------

from dog_walker.toolbox import (  # noqa: E402
    _haversine_m,
    _solve_order,
    optimize_route,
)


def test_haversine_knows_a_degree_of_latitude():
    # one degree of latitude is ~111.2 km everywhere on Earth
    d = _haversine_m((41.0, -87.0), (42.0, -87.0))
    assert d == pytest.approx(111_200, rel=0.01)


def test_solver_unscrambles_stops_on_a_line():
    # four stops on a straight line, listed scrambled: 0, 2, 3, 1.
    # visiting in line order (0->1->2->3->home) is obviously shortest,
    # and the solver must find it from the matrix alone.
    line = [0, 2, 3, 1]

    def dist(i, j):
        return abs(line[i] - line[j]) * 1000

    matrix = [[dist(i, j) for j in range(4)] for i in range(4)]
    assert _solve_order(matrix) == [0, 3, 1, 2]  # positions of 1, 2, 3


LINE_MATRIX = [
    [0, 833, 1666, 2499],
    [833, 0, 833, 1666],
    [1666, 833, 0, 833],
    [2499, 1666, 833, 0],
]


def fake_line_matrix(coords):
    """Stand-in for _walking_matrix: 833m between neighbours on a line
    (= exactly 10 minutes at 5 km/h), flagged as NOT real streets."""
    return LINE_MATRIX, False


def no_geometry(coords_in_order):
    """Stand-in for _street_geometry: pretend ORS is unreachable."""
    return None


STOPS = [
    {"name": "home", "lat": 0.0, "lon": 0.0},
    {"name": "Daisy", "lat": 0.0, "lon": 0.01, "walk_minutes": 20},
    {"name": "Rex", "lat": 0.0, "lon": 0.02, "walk_minutes": 60},
    {"name": "Biscuit", "lat": 0.0, "lon": 0.03, "walk_minutes": 30},
]


@pytest.fixture()
def line_route(monkeypatch):
    """optimize_route over the hand-checkable line world."""
    monkeypatch.setattr("dog_walker.toolbox._walking_matrix", fake_line_matrix)
    monkeypatch.setattr("dog_walker.toolbox._street_geometry", no_geometry)
    return optimize_route(STOPS, start_time="13:00")


def test_route_visits_the_line_in_order(line_route):
    assert line_route["order"] == ["home", "Daisy", "Rex", "Biscuit"]


def test_route_legs_include_the_trip_home(line_route):
    assert line_route["legs"][-1]["to"] == "home"
    assert len(line_route["legs"]) == 4  # 3 outbound + return


def test_timeline_is_hand_checkable(line_route):
    # 833m legs = 10 min each. 13:00 depart -> Daisy 13:09 (rounding),
    # 20 min walk -> 13:29; +10 transit -> Rex 13:39, 60 min -> 14:39;
    # +10 -> Biscuit 14:49, 30 min -> 15:19; 2499m home = 30 min -> 15:49
    tl = line_route["timeline"]
    assert [row["stop"] for row in tl] == ["Daisy", "Rex", "Biscuit", "home"]
    assert tl[0]["walk_start"] == "13:09"
    assert tl[0]["walk_end"] == "13:29"
    assert tl[1]["walk_start"] == "13:39"
    assert tl[1]["walk_end"] == "14:39"
    assert tl[3] == {"stop": "home", "arrive": "15:49"}


def test_timeline_without_start_time_is_minutes_from_start(monkeypatch):
    monkeypatch.setattr("dog_walker.toolbox._walking_matrix", fake_line_matrix)
    monkeypatch.setattr("dog_walker.toolbox._street_geometry", no_geometry)
    tl = optimize_route(STOPS)["timeline"]
    assert tl[0]["walk_start"] == 10 and tl[0]["walk_end"] == 30


def test_route_totals_add_up(line_route):
    assert line_route["total_walk_meters"] == 833 * 3 + 2499
    assert line_route["dog_walk_minutes"] == 110
    assert line_route["total_minutes"] == 60 + 110


def test_fallback_flag_reaches_the_caller(line_route):
    # fake matrix said "not real streets"; the tool must not upgrade it
    assert line_route["uses_real_streets"] is False
    assert line_route["geometry"] is None


def test_route_rejects_too_few_or_too_many_stops():
    assert "error" in optimize_route([{"name": "solo", "lat": 0, "lon": 0}])
    too_many = [{"name": f"s{i}", "lat": 0, "lon": 0} for i in range(11)]
    assert "error" in optimize_route(too_many)


# ---------------------------------------------------------------------
# per-dog tolerance and buffer
# ---------------------------------------------------------------------


def test_inside_the_band_is_ok_outside_escalates_by_distance():
    # default band [20, 84]; one rung per 10F beyond an edge
    assert assess_walk_safety(day(feels_like_f=[18.0] * 24), 8, 20)[
        "verdict"] == "CAUTION"        # 2F below
    assert assess_walk_safety(day(feels_like_f=[8.0] * 24), 8, 20)[
        "verdict"] == "SHORTEN"        # 12F below
    assert assess_walk_safety(day(feels_like_f=[-1.0] * 24), 8, 20)[
        "verdict"] == "DO_NOT_WALK"    # 21F below
    assert assess_walk_safety(day(feels_like_f=[95.0] * 24), 8, 20)[
        "verdict"] == "SHORTEN"        # 11F above


def test_husky_band_shrugs_off_what_triggers_the_default():
    chilly = day(feels_like_f=[18.0] * 24)
    assert assess_walk_safety(chilly, 8, 20)["verdict"] == "CAUTION"
    assert assess_walk_safety(
        chilly, 8, 20, comfort_min_f=-10, comfort_max_f=70
    )["verdict"] == "OK"


def test_narrow_band_bulldog_flags_both_directions():
    # the case the single-axis design could not express
    mild = day(feels_like_f=[78.0] * 24)
    assert assess_walk_safety(mild, 8, 20)["verdict"] == "OK"
    assert assess_walk_safety(
        mild, 8, 20, comfort_min_f=45, comfort_max_f=75
    )["verdict"] == "CAUTION"
    cool = day(feels_like_f=[40.0] * 24)
    assert assess_walk_safety(
        cool, 8, 20, comfort_min_f=45, comfort_max_f=75
    )["verdict"] == "CAUTION"


def test_degenerate_bands_are_forgiven():
    # inverted swaps; too-narrow widens to the 10F minimum
    mild = day(feels_like_f=[60.0] * 24)
    r = assess_walk_safety(mild, 8, 20, comfort_min_f=80, comfort_max_f=30)
    assert r["window"]["comfort_min_f"] == 30
    r = assess_walk_safety(mild, 8, 20, comfort_min_f=60, comfort_max_f=61)
    assert r["window"]["comfort_max_f"] - r["window"]["comfort_min_f"] == 10


def test_wet_cold_stays_absolute_physics():
    # near-freezing drizzle bumps even a husky-band dog: wet fur is
    # wet fur
    drizzle = day(feels_like_f=[34.0] * 24, precip_mm=[1.0] * 24)
    r = assess_walk_safety(drizzle, 8, 20, comfort_min_f=-10, comfort_max_f=70)
    assert r["verdict"] == "CAUTION"
    assert any("wet-cold" in x for x in r["reasons"])


def test_buffer_delays_the_walk_but_not_the_arrival(monkeypatch):
    monkeypatch.setattr("dog_walker.toolbox._walking_matrix", fake_line_matrix)
    monkeypatch.setattr("dog_walker.toolbox._street_geometry", no_geometry)
    stops = [dict(s) for s in STOPS]
    stops[1]["buffer_minutes"] = 15   # Daisy: slow elevator
    r = optimize_route(stops, start_time="13:00")
    daisy = r["timeline"][0]
    assert daisy["arrive"] == "13:09"
    assert daisy["walk_start"] == "13:24"      # 15 min prep
    assert daisy["walk_end"] == "13:44"
    assert daisy["buffer_minutes"] == 15
    assert r["buffer_minutes"] == 15
    assert r["total_minutes"] == 60 + 110 + 15


def test_timeline_echoes_the_band_for_the_auditor(monkeypatch):
    monkeypatch.setattr("dog_walker.toolbox._walking_matrix", fake_line_matrix)
    monkeypatch.setattr("dog_walker.toolbox._street_geometry", no_geometry)
    stops = [dict(s) for s in STOPS]
    stops[2]["comfort_min_f"] = 45
    stops[2]["comfort_max_f"] = 95
    r = optimize_route(stops, start_time="13:00")
    rex = next(e for e in r["timeline"] if e["stop"] == "Rex")
    assert rex["comfort_min_f"] == 45 and rex["comfort_max_f"] == 95
