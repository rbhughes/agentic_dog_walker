"""The new toolbox: plain Python functions returning JSON-able dicts.

One definition, three consumers:
  * the agent dispatches these in-process via REGISTRY
  * the MCP server facade (mcp_server.py, Phase 2b) wraps the same REGISTRY
  * tests call the pure parts directly, offline

Design rules (earned in Phase 1):
  * tools return STRUCTURED DATA, never prose -- the 2024 version
    trapped its judgment inside formatted strings
  * judgment is deterministic and lives HERE, not in the model: the
    weather tool computes a safety verdict with testable thresholds;
    the model's job is to plan and narrate around it
  * every tool has a JSON Schema (the same dialect the model reads
    and jsonschema validates) registered alongside the function
"""

from __future__ import annotations

from typing import Any

import requests

# ---------------------------------------------------------------------
# weather
# ---------------------------------------------------------------------

# Walk-safety verdicts, strictest wins. The model never invents these;
# it relays them.
VERDICTS = ("OK", "CAUTION", "SHORTEN", "DO_NOT_WALK")

# Fahrenheit, owner-rounded policy (2026-09-10). Started as exact
# conversions of the original Celsius ladders; cold was rounded to
# 20/10/-5 and heat CAUTION raised to 84F (68F fired on pleasant
# days). Changing any rung is a POLICY change and will trip tests --
# that friction is intentional.
COLD_LADDER = [
    (20, "CAUTION"),
    (10, "SHORTEN"),
    (-5, "DO_NOT_WALK"),
]
HEAT_LADDER = [
    (84, "CAUTION"),
    (88, "SHORTEN"),
    (95, "DO_NOT_WALK"),
]
WIND_LADDER = [
    (35, "CAUTION"),
    (40, "SHORTEN"),
    (45, "DO_NOT_WALK"),
]
PRECIP_LADDER = [
    (10, "CAUTION"),
    (20, "SHORTEN"),
    (30, "DO_NOT_WALK"),
]


def _wet_cold(window: dict) -> bool:
    """Rain near freezing: a soaked coat loses its insulation, so the
    combination is worse than either number alone suggests."""
    return window["max_precip_mm"] > 0.5 and window["min_feels_like_f"] <= 35.6


# Combo escalations: (name, predicate over the window summary, reason).
# A triggered escalation bumps the base verdict one rung; reasons
# always append, even at the top rung.
ESCALATIONS = [
    ("wet-cold", _wet_cold, "rain near freezing soaks the coat and defeats insulation"),
]


def bump(verdict: str, rungs: int = 1) -> str:
    """One rung more severe, clamped at DO_NOT_WALK."""
    i = VERDICTS.index(verdict) + rungs
    return VERDICTS[min(i, len(VERDICTS) - 1)]


def _walk_ladder(value: float, ladder: list, colder_is_worse: bool = False):
    """Most severe rung a value triggers, or None.

    Ladders are ordered mild -> severe; a rung triggers when the value
    crosses its threshold (<= for cold ladders, >= otherwise)."""
    hit = None
    for threshold, verdict in ladder:
        if (value <= threshold) if colder_is_worse else (value >= threshold):
            hit = (threshold, verdict)
    return hit


def fetch_forecast(lat: float, lon: float, date: str) -> dict[str, list]:
    """Hourly forecast arrays for one date (ported from the 2024 tool,
    plus apparent_temperature ('feels like'), which drives cold thresholds
    better than air temperature).

    Returns {"time": [...], "temp_f": [...], "feels_like_f": [...],
             "precip_mm": [...], "wind_kph": [...]} -- temperatures in
             Fahrenheit (Open-Meteo converts server-side)
    """
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": date,
            "end_date": date,
            "hourly": "temperature_2m,apparent_temperature,"
            "precipitation,wind_speed_10m",
            "temperature_unit": "fahrenheit",
            "timezone": "auto",
        },
        timeout=10,
    )
    resp.raise_for_status()
    h = resp.json()["hourly"]
    return {
        "time": h["time"],
        "temp_f": h["temperature_2m"],
        "feels_like_f": h["apparent_temperature"],
        "precip_mm": h["precipitation"],
        "wind_kph": h["wind_speed_10m"],
    }


def assess_walk_safety(hours: dict[str, list], start_hour: int, end_hour: int) -> dict:
    """The judgment seat of the whole system. Deterministic, offline,
    unit-testable: no model, no network.

    Looks only at the [start_hour, end_hour) window -- the 2024 tool
    averaged the whole day, which erased the fact that timing IS the
    decision (a -13F dawn doesn't cancel a 23F afternoon walk).

    Returns:
        {
          "verdict": one of VERDICTS (strictest triggered rule wins),
          "reasons": [short human-readable strings, one per rule hit],
          "window": {"start_hour": ..., "end_hour": ...,
                     "min_feels_like_f": ..., "max_precip_mm": ...,
                     "max_wind_kph": ...},
        }

    """
    idx = [
        i for i, t in enumerate(hours["time"]) if start_hour <= int(t[11:13]) < end_hour
    ]
    if not idx:
        raise ValueError(f"no forecast hours in window [{start_hour}, {end_hour})")

    def pick(key: str, fn) -> float:
        """Reduce one forecast array over the in-window hours."""

        return fn(hours[key][i] for i in idx)

    window = {
        "start_hour": start_hour,
        "end_hour": end_hour,
        "min_feels_like_f": pick("feels_like_f", min),
        "max_feels_like_f": pick("feels_like_f", max),
        "max_wind_kph": pick("wind_kph", max),
        "max_precip_mm": pick("precip_mm", max),
    }

    verdict, reasons = "OK", []
    # (value, ladder, colder_is_worse, label) -- heat uses feels-like
    # too: apparent_temperature folds in humidity, which is the part
    # of heat that kills dogs
    ladder_checks = [
        (window["min_feels_like_f"], COLD_LADDER, True, "feels-like low"),
        (window["max_feels_like_f"], HEAT_LADDER, False, "feels-like high"),
        (window["max_wind_kph"], WIND_LADDER, False, "wind"),
        (window["max_precip_mm"], PRECIP_LADDER, False, "precipitation"),
    ]
    for value, ladder, colder, label in ladder_checks:
        if hit := _walk_ladder(value, ladder, colder):
            # print(f"  hit is currently {hit}")
            threshold, rung = hit
            reasons.append(f"{label} {value:g} crosses {rung} threshold {threshold:g}")
            if VERDICTS.index(rung) > VERDICTS.index(verdict):
                verdict = rung

    # print(f"verdict after _walk_ladder: {verdict}")

    for name, hits, why in ESCALATIONS:
        if hits(window):
            verdict = bump(verdict)
            reasons.append(f"{name}: {why}")

    # print(f"verdict after ESCALATIONS: {verdict}")

    # print({"verdict": verdict, "reasons": reasons, "window": window})
    return {"verdict": verdict, "reasons": reasons, "window": window}


def check_weather(
    lat: float, lon: float, date: str, start_hour: int = 8, end_hour: int = 20
) -> dict[str, Any]:
    """The tool the agent (and MCP facade) exposes: fetch + assess.

    Note the typed signature -- the 2024 version took one comma-packed
    string ("lat,lon,YYYY-MM-DD") because ReAct-era tools ate prose.

    end_hour is exclusive, which invites an off-by-one whenever a walk
    fits inside one clock hour (13:29-13:59 floors to [13, 13) --
    observed crashing two models live). Forgive it: an empty or
    inverted window means "that hour".
    """
    if end_hour <= start_hour:
        end_hour = min(start_hour + 1, 24)
    hours = fetch_forecast(lat, lon, date)
    return assess_walk_safety(hours, start_hour, end_hour)


CHECK_WEATHER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_weather",
        "description": (
            "Get a deterministic dog-walk safety verdict for a location "
            "and date. Returns verdict (OK/CAUTION/SHORTEN/DO_NOT_WALK), "
            "reasons, and the worst readings in the time window."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "latitude"},
                "lon": {"type": "number", "description": "longitude"},
                "date": {
                    "type": "string",
                    "format": "date",
                    "description": "ISO date YYYY-MM-DD",
                },
                "start_hour": {
                    "type": "integer",
                    "description": "walk window start hour, 0-23 local (default 8)",
                },
                "end_hour": {
                    "type": "integer",
                    "description": (
                        "walk window end hour, EXCLUSIVE (default 20): a "
                        "walk 13:29-13:59 is start_hour 13, end_hour 14"
                    ),
                },
            },
            "required": ["lat", "lon", "date"],
            "additionalProperties": False,
        },
    },
}

# ---------------------------------------------------------------------
# geocoding (Nominatim / OpenStreetMap)
# ---------------------------------------------------------------------

# Nominatim usage policy (https://operations.osmfoundation.org/policies/nominatim/):
# max 1 request/second, identifying User-Agent, cache results. The cap
# on batch size doubles as an abuse limit once this is public.
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "agentic-dog-walker/2.0 (walker.purr.io)"
MAX_ADDRESSES = 12
_geocode_cache: dict[str, dict] = {}


def geocode_addresses(addresses: list[str]) -> dict[str, Any]:
    """Convert street addresses to coordinates.

    Batched (one tool call = one model round-trip, however many
    addresses). Failed addresses return an "error" entry in their slot
    rather than raising -- the agent needs to know WHICH address broke.

    Compare legacy/dog_walker_2024/tools/geocoding.py, whose first 15
    lines un-mangle a single string that might be JSON, Python-repr,
    or bare text. Typed parameters made that layer evaporate.
    """
    import time

    if len(addresses) > MAX_ADDRESSES:
        return {"error": f"too many addresses (max {MAX_ADDRESSES})"}

    results = []
    for address in addresses:
        key = address.strip().lower()
        if key in _geocode_cache:
            results.append(_geocode_cache[key])
            continue
        try:
            resp = requests.get(
                NOMINATIM_URL,
                params={"q": address, "format": "json", "limit": 1},
                headers={"User-Agent": USER_AGENT},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data:
                hit = {
                    "address": address,
                    "lat": float(data[0]["lat"]),
                    "lon": float(data[0]["lon"]),
                    "display_name": data[0].get("display_name", address),
                }
            else:
                hit = {"address": address, "error": "not found"}
        except requests.RequestException as e:
            hit = {"address": address, "error": str(e)}
        else:
            _geocode_cache[key] = hit
        results.append(hit)
        time.sleep(1.1)  # Nominatim: max 1 req/s, no exceptions

    return {"results": results}


GEOCODE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "geocode_addresses",
        "description": (
            "Convert street addresses to lat/lon coordinates. Batched: "
            "pass ALL addresses in one call. Each result carries either "
            "lat/lon or an error for that address."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "addresses": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": MAX_ADDRESSES,
                    "description": "full street addresses",
                },
            },
            "required": ["addresses"],
            "additionalProperties": False,
        },
    },
}

# ---------------------------------------------------------------------
# registry: name -> (callable, schema). Agent + MCP facade both read
# ---------------------------------------------------------------------
# route optimization (OR-Tools + OpenRouteService)
# ---------------------------------------------------------------------

ORS_MATRIX_URL = "https://api.openrouteservice.org/v2/matrix/foot-walking"
ORS_DIRECTIONS_URL = (
    "https://api.openrouteservice.org/v2/directions/foot-walking/geojson"
)
MAX_STOPS = 10  # abuse cap for the public API, and ORS-polite
WALK_SPEED_M_PER_MIN = 83.33  # 5 km/h


def _ors_key() -> str:
    import os

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    return os.environ.get("OPENROUTESERVICE_API_KEY", "")


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Straight-line ('as the crow flies') metres between two lat/lon
    points -- the honest fallback when street distances are unavailable."""
    import math

    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 6_371_000 * 2 * math.asin(math.sqrt(h))


def _walking_matrix(coords: list[tuple[float, float]]) -> tuple[list[list[int]], bool]:
    """All-pairs walking distances in metres.

    Tries OpenRouteService (real streets); on any failure falls back to
    straight-line distances. Returns (matrix, uses_real_streets) -- the
    flag travels all the way to the user, never silently degraded.
    """
    key = _ors_key()
    if key:
        try:
            resp = requests.post(
                ORS_MATRIX_URL,
                json={
                    # ORS speaks [lon, lat] -- the GeoJSON axis order,
                    # opposite of the [lat, lon] convention everywhere else
                    "locations": [[lon, lat] for lat, lon in coords],
                    "metrics": ["distance"],
                },
                headers={"Authorization": key},
                timeout=15,
            )
            resp.raise_for_status()
            rows = resp.json()["distances"]
            return [[int(d) for d in row] for row in rows], True
        except (requests.RequestException, KeyError):
            pass
    n = len(coords)
    matrix = [
        [0 if i == j else int(_haversine_m(coords[i], coords[j])) for j in range(n)]
        for i in range(n)
    ]
    return matrix, False


def _solve_order(matrix: list[list[int]]) -> list[int]:
    """Best visiting order for the distance matrix (the Traveling
    Salesman step), via OR-Tools. Starts and ends at stop 0 -- the
    walker's own start point. Exact for our tiny stop counts."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    n = len(matrix)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # n stops, 1 walker, depot 0
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        return matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    solution = routing.SolveWithParameters(params)

    order, index = [], routing.Start(0)
    while not routing.IsEnd(index):
        order.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    return order  # e.g. [0, 3, 1, 2]; return to 0 is implicit


def _street_geometry(coords_in_order: list[tuple[float, float]]) -> dict | None:
    """Street-following path through the ordered stops, as a GeoJSON
    LineString the browser map draws directly. None on any failure --
    the route itself is already decided; drawing degrades to straight
    lines client-side."""
    key = _ors_key()
    if not key:
        return None
    try:
        resp = requests.post(
            ORS_DIRECTIONS_URL,
            json={"coordinates": [[lon, lat] for lat, lon in coords_in_order]},
            headers={"Authorization": key},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["features"][0]["geometry"]
    except (requests.RequestException, KeyError, IndexError):
        return None


WALK_DURATIONS = (20, 30, 60)  # the products a dog walker actually sells


def optimize_route(stops: list[dict], start_time: str | None = None) -> dict[str, Any]:
    """Best transit order through the stops, plus the resulting schedule.

    Each stop: {"name": ..., "lat": ..., "lon": ..., "walk_minutes": ...}
    where walk_minutes is that dog's own walk (20/30/60), taken as a
    loop from its home -- the walker arrives, walks the dog out and
    back, returns it, and transits to the next stop. Stop 0 is the
    walker's start/end and takes no walk_minutes.

    Because each dog's loop anchors to its own home, the visiting
    order is pure geography (the Traveling Salesman solve over transit
    distances); the durations shape the TIMELINE -- when the walker
    reaches each dog and the interval that dog is actually outside.
    That per-stop interval is what downstream weather checks and pet
    time-window constraints consume.

    start_time ("HH:MM", optional) renders the timeline as clock
    times; otherwise it's minutes-from-start.
    """
    if not 2 <= len(stops) <= MAX_STOPS:
        return {"error": f"need 2-{MAX_STOPS} stops, got {len(stops)}"}

    coords = [(float(s["lat"]), float(s["lon"])) for s in stops]
    matrix, real_streets = _walking_matrix(coords)
    order = _solve_order(matrix)

    loop = order + [0]  # explicit return home
    legs = []
    for a, b in zip(loop, loop[1:]):
        legs.append(
            {
                "from": stops[a]["name"],
                "to": stops[b]["name"],
                "meters": matrix[a][b],
            }
        )
    total_m = sum(leg["meters"] for leg in legs)
    transit_min = total_m / WALK_SPEED_M_PER_MIN

    # timeline: transit legs and per-dog walk loops, in visit order
    def clock(minutes: float) -> Any:
        """Render an offset as HH:MM if start_time given, else minutes."""
        if start_time is None:
            return round(minutes)
        h, m = map(int, start_time.split(":"))
        total = h * 60 + m + minutes
        return f"{int(total // 60) % 24:02d}:{int(total % 60):02d}"

    timeline, t = [], 0.0
    for pos, (a, b) in enumerate(zip(loop, loop[1:])):
        t += matrix[a][b] / WALK_SPEED_M_PER_MIN
        if b == 0:
            timeline.append({"stop": stops[0]["name"], "arrive": clock(t)})
            break
        walk = int(stops[b].get("walk_minutes", 0))
        timeline.append(
            {
                "stop": stops[b]["name"],
                "arrive": clock(t),
                "walk_start": clock(t),
                "walk_end": clock(t + walk),
                "walk_minutes": walk,
            }
        )
        t += walk

    dog_min = sum(int(s.get("walk_minutes", 0)) for s in stops)
    return {
        "order": [stops[i]["name"] for i in order],
        "legs": legs,
        "timeline": timeline,
        "total_walk_meters": total_m,
        "transit_minutes": round(transit_min),
        "dog_walk_minutes": dog_min,
        "total_minutes": round(transit_min + dog_min),
        "uses_real_streets": real_streets,
        "geometry": _street_geometry([coords[i] for i in loop]),
    }


OPTIMIZE_ROUTE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "optimize_route",
        "description": (
            "Find the best transit order through the stops (round trip "
            "from stop 0) and the resulting schedule: when the walker "
            "reaches each dog and each dog's walk interval. Returns "
            "order, legs, timeline, distances, and street-path geometry "
            "for the map. Call AFTER geocoding."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "stops": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": MAX_STOPS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "lat": {"type": "number"},
                            "lon": {"type": "number"},
                            "walk_minutes": {
                                "type": "integer",
                                "enum": [0, *WALK_DURATIONS],
                                "description": (
                                    "this dog's walk length, looped "
                                    "from its own home; 0 = the start "
                                    "stop (no dog there)"
                                ),
                            },
                        },
                        "required": ["name", "lat", "lon"],
                        "additionalProperties": False,
                    },
                    "description": "stop 0 = walker's start/end, no walk_minutes",
                },
                "start_time": {
                    "type": "string",
                    "description": "HH:MM; renders the timeline as clock times",
                },
            },
            "required": ["stops"],
            "additionalProperties": False,
        },
    },
}

# ---------------------------------------------------------------------
# registry: name -> (callable, schema). Agent + MCP facade both read
# this; adding a tool means adding a function, a schema, and one row.
# ---------------------------------------------------------------------

REGISTRY: dict[str, tuple[Any, dict]] = {
    "check_weather": (check_weather, CHECK_WEATHER_SCHEMA),
    "geocode_addresses": (geocode_addresses, GEOCODE_SCHEMA),
    "optimize_route": (optimize_route, OPTIMIZE_ROUTE_SCHEMA),
}
