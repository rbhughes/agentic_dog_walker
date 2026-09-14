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
# Per-dog comfort band, in F on feels-like. The band's POSITION is
# the husky-vs-iggy axis; its WIDTH is the hardy-vs-bulldog axis
# (a two-thumb slider in the UI). Inside the band: OK. Beyond either
# edge, severity climbs one rung per BEYOND_STEP_F degrees. The
# default band edges reproduce the old global CAUTION rungs.
DEFAULT_COMFORT_MIN_F = 20
DEFAULT_COMFORT_MAX_F = 84
COMFORT_BOUNDS_F = (-20, 110)   # slider limits
MIN_BAND_WIDTH_F = 10
BEYOND_STEP_F = 10              # each 10F beyond the band = one rung worse

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


def _band_verdict(distance_beyond: float) -> str:
    """Severity from how far outside the comfort band the feels-like
    went: one rung per BEYOND_STEP_F degrees, clamped at the top."""
    if distance_beyond <= 0:
        return "OK"
    rung = 1 + int(distance_beyond // BEYOND_STEP_F)
    return VERDICTS[min(rung, len(VERDICTS) - 1)]


def _wet_cold(window: dict) -> bool:
    """Rain near freezing: a soaked coat loses its insulation, so the
    combination is worse than either number alone suggests. Absolute
    (physics of wet fur near frost), not band-relative."""
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


def assess_walk_safety(
    hours: dict[str, list],
    start_hour: int,
    end_hour: int,
    comfort_min_f: float = DEFAULT_COMFORT_MIN_F,
    comfort_max_f: float = DEFAULT_COMFORT_MAX_F,
) -> dict:
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

    # forgive degenerate bands (front-door validation is stricter):
    # inverted -> swap; too narrow -> widen to the minimum
    lo, hi = float(comfort_min_f), float(comfort_max_f)
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo < MIN_BAND_WIDTH_F:
        mid = (lo + hi) / 2
        lo, hi = mid - MIN_BAND_WIDTH_F / 2, mid + MIN_BAND_WIDTH_F / 2
    window["comfort_min_f"] = lo
    window["comfort_max_f"] = hi

    verdict, reasons = "OK", []
    # the comfort band: one rung per BEYOND_STEP_F degrees outside it.
    # Heat and cold both judge feels-like (apparent temperature folds
    # in humidity, the part of heat that kills dogs).
    cold_beyond = lo - window["min_feels_like_f"]
    if (cold_v := _band_verdict(cold_beyond)) != "OK":
        reasons.append(
            f"feels-like {window['min_feels_like_f']:g} is "
            f"{cold_beyond:g}F below comfort minimum {lo:g}"
        )
        verdict = cold_v
    heat_beyond = window["max_feels_like_f"] - hi
    if (heat_v := _band_verdict(heat_beyond)) != "OK":
        reasons.append(
            f"feels-like {window['max_feels_like_f']:g} is "
            f"{heat_beyond:g}F above comfort maximum {hi:g}"
        )
        if VERDICTS.index(heat_v) > VERDICTS.index(verdict):
            verdict = heat_v

    ladder_checks = [
        (window["max_wind_kph"], WIND_LADDER, False, "wind"),
        (window["max_precip_mm"], PRECIP_LADDER, False, "precipitation"),
    ]
    for value, ladder, colder, label in ladder_checks:
        if hit := _walk_ladder(value, ladder, colder):
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
    lat: float,
    lon: float,
    date: str,
    start_hour: int = 8,
    end_hour: int = 20,
    comfort_min_f: float = DEFAULT_COMFORT_MIN_F,
    comfort_max_f: float = DEFAULT_COMFORT_MAX_F,
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
    return assess_walk_safety(
        hours, start_hour, end_hour, comfort_min_f, comfort_max_f
    )


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
                "comfort_min_f": {
                    "type": "number", "minimum": -20, "maximum": 110,
                    "description": (
                        "this dog's comfort-band minimum, F feels-like "
                        "(default 20); severity climbs one rung per 10F "
                        "below it"
                    ),
                },
                "comfort_max_f": {
                    "type": "number", "minimum": -20, "maximum": 110,
                    "description": (
                        "this dog's comfort-band maximum, F feels-like "
                        "(default 84); severity climbs one rung per 10F "
                        "above it"
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
# terrain (Open-Meteo elevation) -- a per-dog hilliness advisory.
#
# Walks are abstract loops from a dog's home, not routed paths, so
# "avoid hills" can only mean: is this dog's NEIGHBORHOOD too hilly
# for it? We sample elevation on a small grid around the home and
# score the relief (max minus min). A dog with a stated tolerance
# (max_relief_m) gets OK / CAUTION / AVOID. Same shape as weather:
# fetch a per-location value, judge against a per-dog threshold in
# code, let the auditor require the check was made.
# ---------------------------------------------------------------------

ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
DEFAULT_MAX_RELIEF_M = 1000   # generous: a dog with no stated limit never flags
_TERRAIN_GRID_DEG = 0.003     # ~330m; a 3x3 grid spans ~660m around the home


def fetch_elevation_grid(lat: float, lon: float) -> list[float]:
    """Elevations (m) on a 3x3 grid around the point, one API call."""
    import math

    dlon = _TERRAIN_GRID_DEG / max(math.cos(math.radians(lat)), 0.1)
    lats, lons = [], []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            lats.append(lat + i * _TERRAIN_GRID_DEG)
            lons.append(lon + j * dlon)
    resp = requests.get(
        ELEVATION_URL,
        params={
            "latitude": ",".join(f"{x:.5f}" for x in lats),
            "longitude": ",".join(f"{x:.5f}" for x in lons),
        },
        timeout=10,
    )
    resp.raise_for_status()
    return [float(e) for e in resp.json()["elevation"]]


def assess_terrain(elevations: list[float], max_relief_m: float) -> dict[str, Any]:
    """Deterministic, offline: relief vs the dog's tolerance.
    OK within tolerance, CAUTION up to 1.5x, AVOID beyond. The 1.5x
    band and the tier idea are policy, like the weather ladders."""
    relief = round(max(elevations) - min(elevations), 1)
    if relief <= max_relief_m:
        verdict = "OK"
    elif relief <= 1.5 * max_relief_m:
        verdict = "CAUTION"
    else:
        verdict = "AVOID"
    return {"verdict": verdict, "relief_m": relief, "max_relief_m": max_relief_m}


def check_terrain(
    lat: float, lon: float, max_relief_m: float = DEFAULT_MAX_RELIEF_M
) -> dict[str, Any]:
    """Hilliness advisory for a dog's neighborhood: fetch + assess."""
    return assess_terrain(fetch_elevation_grid(lat, lon), max_relief_m)


CHECK_TERRAIN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_terrain",
        "description": (
            "Assess how hilly a dog's neighborhood is (elevation relief "
            "over a small grid around its home) against that dog's "
            "max_relief_m tolerance. Returns OK/CAUTION/AVOID. Call for "
            "any dog that has a max_relief_m, at that dog's location."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "latitude"},
                "lon": {"type": "number", "description": "longitude"},
                "max_relief_m": {
                    "type": "number", "minimum": 1, "maximum": 2000,
                    "description": (
                        "this dog's hill tolerance in metres of relief; "
                        "e.g. 15 for a dog that needs flat ground"
                    ),
                },
            },
            "required": ["lat", "lon", "max_relief_m"],
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


def _solve_order_timed(
    matrix: list[list[int]],
    service_min: list[int],
    deadline_min: list[int | None],
    speed_m_per_min: float,
) -> list[int] | None:
    """Like _solve_order, but honors per-stop deadlines (medication
    urgency). Adds an OR-Tools time dimension: cumulative time at a
    node is the walker's ARRIVAL there, and a deadline caps it. Among
    all deadline-satisfying orders it still minimizes distance.
    Returns None when NO order can meet every deadline -- real,
    checkable infeasibility, not a distance-order artifact.

    service_min[i]  minutes the walker spends at stop i before leaving
                    (buffer + meds + walk); 0 at the depot.
    deadline_min[i] latest arrival at i in minutes-from-start, or None.
    """
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    n = len(matrix)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(i: int, j: int) -> int:
        return matrix[manager.IndexToNode(i)][manager.IndexToNode(j)]

    dist_idx = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)  # still shortest

    def time_cb(i: int, j: int) -> int:
        a = manager.IndexToNode(i)
        travel = round(matrix[a][manager.IndexToNode(j)] / speed_m_per_min)
        return travel + service_min[a]  # leaving a costs its service

    time_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_idx, 0, 24 * 60, True, "Time")  # fixed start = 0
    time_dim = routing.GetDimensionOrDie("Time")
    for node, deadline in enumerate(deadline_min):
        if deadline is not None:
            time_dim.CumulVar(manager.NodeToIndex(node)).SetMax(deadline)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    solution = routing.SolveWithParameters(params)
    if solution is None:
        return None

    order, index = [], routing.Start(0)
    while not routing.IsEnd(index):
        order.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    return order


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

    Each stop: {"name", "lat", "lon", "walk_minutes", and optionally
    "buffer_minutes" (prep time -- elevators, feeding, parking --
    spent BEFORE the walk: it delays walk_start and lengthens the
    schedule but not the dog's time outside) and "comfort_min_f"/
    "comfort_max_f" (the dog's comfort band, echoed into the
    timeline so weather checks can be audited against them)}. walk_minutes is that dog's
    own walk (20/30/60), taken as a loop from its home -- the walker arrives, walks the dog out and
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

    # validate the deadline/start_time contract BEFORE any network work
    has_deadlines = any(s.get("med_deadline") for s in stops)
    if has_deadlines and start_time is None:
        return {"error": "medication deadlines require start_time"}

    coords = [(float(s["lat"]), float(s["lon"])) for s in stops]
    matrix, real_streets = _walking_matrix(coords)

    def start_min() -> int:
        h, m = map(int, start_time.split(":"))
        return h * 60 + m

    # medication deadlines (urgency) turn the plain TSP into a
    # time-windowed solve; infeasibility is a real, honest outcome
    if has_deadlines:
        service = [
            int(s.get("buffer_minutes", 0))
            + int(s.get("med_minutes", 0))
            + int(s.get("walk_minutes", 0))
            for s in stops
        ]
        deadlines: list[int | None] = []
        for s in stops:
            d = s.get("med_deadline")
            deadlines.append(
                (int(d[:2]) * 60 + int(d[3:5])) - start_min() if d else None
            )
        order = _solve_order_timed(matrix, service, deadlines, WALK_SPEED_M_PER_MIN)
        if order is None:
            due = [
                {"stop": s["name"], "med_deadline": s["med_deadline"]}
                for s in stops
                if s.get("med_deadline")
            ]
            return {
                "feasible": False,
                "reason": "no visiting order meets every medication deadline",
                "deadlines": due,
                "uses_real_streets": real_streets,
            }
    else:
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

    all_met = True
    timeline, t = [], 0.0
    for pos, (a, b) in enumerate(zip(loop, loop[1:])):
        t += matrix[a][b] / WALK_SPEED_M_PER_MIN
        if b == 0:
            timeline.append({"stop": stops[0]["name"], "arrive": clock(t)})
            break
        arrive = t
        walk = int(stops[b].get("walk_minutes", 0))
        buffer = int(stops[b].get("buffer_minutes", 0))
        meds = int(stops[b].get("med_minutes", 0))
        entry = {
            "stop": stops[b]["name"],
            "arrive": clock(arrive),
            # meds are administered on arrival; the walk follows the
            # prep + med handling
            "walk_start": clock(t + buffer + meds),
            "walk_end": clock(t + buffer + meds + walk),
            "walk_minutes": walk,
        }
        if buffer:
            entry["buffer_minutes"] = buffer
        if meds:
            entry["med_minutes"] = meds
        deadline = stops[b].get("med_deadline")
        if deadline:
            met = round(arrive) <= (int(deadline[:2]) * 60 + int(deadline[3:5])
                                    - start_min())
            entry["med_deadline"] = deadline
            entry["deadline_met"] = met
            all_met = all_met and met
        for key in ("comfort_min_f", "comfort_max_f", "max_relief_m"):
            if stops[b].get(key) is not None:
                entry[key] = float(stops[b][key])
        timeline.append(entry)
        t += buffer + meds + walk

    dog_min = sum(int(s.get("walk_minutes", 0)) for s in stops)
    buffer_min = sum(int(s.get("buffer_minutes", 0)) for s in stops)
    med_min = sum(int(s.get("med_minutes", 0)) for s in stops)
    return {
        "feasible": all_met,
        "order": [stops[i]["name"] for i in order],
        "legs": legs,
        "timeline": timeline,
        "total_walk_meters": total_m,
        "transit_minutes": round(transit_min),
        "dog_walk_minutes": dog_min,
        "buffer_minutes": buffer_min,
        "med_minutes": med_min,
        "total_minutes": round(transit_min + dog_min + buffer_min + med_min),
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
                            "buffer_minutes": {
                                "type": "integer", "minimum": 0, "maximum": 60,
                                "description": (
                                    "prep minutes before the walk "
                                    "(elevators, feeding, parking)"
                                ),
                            },
                            "comfort_min_f": {
                                "type": "number", "minimum": -20, "maximum": 110,
                                "description": "comfort-band minimum, F",
                            },
                            "comfort_max_f": {
                                "type": "number", "minimum": -20, "maximum": 110,
                                "description": "comfort-band maximum, F",
                            },
                            "max_relief_m": {
                                "type": "number", "minimum": 1, "maximum": 2000,
                                "description": (
                                    "this dog's hill tolerance in metres of "
                                    "relief; set it to require a terrain check"
                                ),
                            },
                            "med_deadline": {
                                "type": "string",
                                "description": (
                                    "HH:MM: this dog needs medication and "
                                    "must be REACHED by this time (urgency). "
                                    "Requires start_time. May make the route "
                                    "infeasible -- then feasible=false."
                                ),
                            },
                            "med_minutes": {
                                "type": "integer", "minimum": 0, "maximum": 30,
                                "description": (
                                    "extra handling minutes to administer "
                                    "medication (difficulty)"
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
                    "description": (
                        "HH:MM; renders the timeline as clock times and "
                        "anchors medication deadlines"
                    ),
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
    "check_terrain": (check_terrain, CHECK_TERRAIN_SCHEMA),
    "geocode_addresses": (geocode_addresses, GEOCODE_SCHEMA),
    "optimize_route": (optimize_route, OPTIMIZE_ROUTE_SCHEMA),
}
