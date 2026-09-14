"""The scenario library the measurement harness runs models against.

A single preset gate answers "can this model do the task once?" -- luck
and one geography. A LIBRARY answers "how often, across which kinds of
task?" Each scenario pairs a roster with the ONE thing the deterministic
oracles will decide about it: whether a valid plan is `feasible`. That
expected outcome is the ground truth the harness grades against -- no
LLM judge, just the same auditor the agent already has to satisfy.

Scenarios reuse the frozen preset coordinates (see presets.py), so a
full sweep costs ZERO live geocoding no matter how many runs -- only
inference varies. Difficulty is dialed by breadth (how many tools a
plan must orchestrate) and by honesty (feasible rosters a model can
fabricate its way through vs infeasible ones it must refuse).

Axes exercised across the set:
  * size           1 dog -> 4 dogs
  * weather        comfort bands that force per-dog verdicts
  * terrain        max_relief_m -> a required check_terrain call
  * meds/buffer    schedule-lengthening handling time
  * walk windows   morning/afternoon -> a solved departure
  * feasibility    rosters that MUST come back feasible=false
"""

from dog_walker.presets import PRESETS, build_request, seed_geocode_cache

# addresses drawn only from the frozen preset geocodes, so nothing here
# reaches live Nominatim once seed_geocode_cache() has run
_LPZ = "Lincoln Park Zoo, Chicago"
_CLARK = "5218 N Clark St, Chicago"
_WRIGLEY = "Wrigley Field, Chicago"
_BROADWAY = "4802 N Broadway, Chicago"
_MILLENNIUM = "Millennium Park, Chicago"
_BUCKINGHAM = "Buckingham Fountain, Chicago"
_SHEDD = "Shedd Aquarium, Chicago"


SCENARIOS: dict[str, dict] = {
    # --- baseline: the easiest possible real run -------------------
    "single-easy": {
        "title": "Single dog, no constraints",
        "expected_feasible": True,
        "tags": ["baseline"],
        "roster": {
            "start_address": _WRIGLEY,
            "start_time": "09:00",
            "pets": [
                {"name": "Rex", "address": _CLARK, "walk_minutes": 30},
            ],
        },
    },
    # --- the two site presets, as graded scenarios -----------------
    "lakeview-classic": {
        "title": "Lakeview classic (3 dogs, comfort bands + buffer)",
        "expected_feasible": True,
        "tags": ["weather", "buffer", "multi"],
        "roster": PRESETS["lakeview-classic"],
    },
    "loop-lunch-hour": {
        "title": "Loop lunch hour (2 downtown dogs)",
        "expected_feasible": True,
        "tags": ["weather", "multi"],
        "roster": PRESETS["loop-lunch-hour"],
    },
    # --- breadth: force the 4th tool (check_terrain) ---------------
    "terrain-roster": {
        "title": "Two dogs that can't handle hills",
        "expected_feasible": True,
        "tags": ["terrain", "weather", "multi"],
        "roster": {
            "start_address": _WRIGLEY,
            "start_time": "10:00",
            "pets": [
                {"name": "Pip", "address": _LPZ, "walk_minutes": 20,
                 "max_relief_m": 30},
                {"name": "Moose", "address": _BROADWAY, "walk_minutes": 30,
                 "max_relief_m": 80, "comfort_min_f": -10, "comfort_max_f": 70},
            ],
        },
    },
    # --- everything at once: the full-house feasible run -----------
    "full-house": {
        "title": "One dog with every attribute set",
        "expected_feasible": True,
        "tags": ["weather", "terrain", "meds", "buffer", "window"],
        "roster": {
            "start_address": _WRIGLEY,
            "start_time": "09:00",
            "pets": [
                {"name": "Daisy", "address": _LPZ, "walk_minutes": 20,
                 "comfort_min_f": 45, "comfort_max_f": 95,
                 "needs_meds": True, "walk_window": "morning"},
                {"name": "Ziggy", "address": _WRIGLEY, "walk_minutes": 30,
                 "buffer_minutes": 15, "max_relief_m": 30},
                {"name": "Rex", "address": _CLARK, "walk_minutes": 60},
            ],
        },
    },
    # --- clustering: one AM + one PM, one outing across noon -------
    "mixed-windows": {
        "title": "Morning dog + afternoon dog (clusters across noon)",
        "expected_feasible": True,
        "tags": ["window", "multi"],
        "roster": {
            "start_address": _MILLENNIUM,
            "start_time": "09:00",
            "pets": [
                {"name": "Mochi", "address": _BUCKINGHAM, "walk_minutes": 30,
                 "walk_window": "morning"},
                {"name": "Peanut", "address": _SHEDD, "walk_minutes": 30,
                 "walk_window": "afternoon"},
            ],
        },
    },
    # --- the honesty test: MUST come back feasible=false ----------
    "morning-overbook": {
        "title": "Four hour-long morning walks (impossible before noon)",
        "expected_feasible": False,
        "tags": ["window", "infeasible", "multi"],
        "roster": {
            "start_address": _MILLENNIUM,
            "start_time": "08:00",
            "pets": [
                {"name": "Rex", "address": _CLARK, "walk_minutes": 60,
                 "walk_window": "morning"},
                {"name": "Daisy", "address": _LPZ, "walk_minutes": 60,
                 "walk_window": "morning"},
                {"name": "Ziggy", "address": _WRIGLEY, "walk_minutes": 60,
                 "walk_window": "morning"},
                {"name": "Wilbur", "address": _BROADWAY, "walk_minutes": 60,
                 "walk_window": "morning"},
            ],
        },
    },
}


def scenario_prompt(scenario: dict) -> str:
    """Render a scenario's roster into the agent's prompt string."""
    r = scenario["roster"]
    return build_request(r["start_address"], r["start_time"], r["pets"])


__all__ = ["SCENARIOS", "scenario_prompt", "seed_geocode_cache"]
