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

COLD_LADDER = [
    (-5, "CAUTION"),
    (-10, "SHORTEN"),
    (-20, "DO_NOT_WALK"),
]
HEAT_LADDER = [
    (20, "CAUTION"),
    (30, "SHORTEN"),
    (35, "DO_NOT_WALK"),
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
    return window["max_precip_mm"] > 0.5 and window["min_feels_like_c"] <= 2


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
    plus apparent_temperature -- 'feels like' drives cold thresholds
    better than air temperature).

    Returns {"time": [...], "temp_c": [...], "feels_like_c": [...],
             "precip_mm": [...], "wind_kph": [...]}
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
            "timezone": "auto",
        },
        timeout=10,
    )
    resp.raise_for_status()
    h = resp.json()["hourly"]
    return {
        "time": h["time"],
        "temp_c": h["temperature_2m"],
        "feels_like_c": h["apparent_temperature"],
        "precip_mm": h["precipitation"],
        "wind_kph": h["wind_speed_10m"],
    }


def assess_walk_safety(hours: dict[str, list], start_hour: int, end_hour: int) -> dict:
    """The judgment seat of the whole system. Deterministic, offline,
    unit-testable: no model, no network.

    Looks only at the [start_hour, end_hour) window -- the 2024 tool
    averaged the whole day, which erased the fact that timing IS the
    decision (a -25C dawn doesn't cancel a -5C afternoon walk).

    Returns:
        {
          "verdict": one of VERDICTS (strictest triggered rule wins),
          "reasons": [short human-readable strings, one per rule hit],
          "window": {"start_hour": ..., "end_hour": ...,
                     "min_feels_like_c": ..., "max_precip_mm": ...,
                     "max_wind_kph": ...},
        }

    The thresholds (ladders/escalations above) are policy and carry
    Bryan's signature; this function is just the mechanism:
    ladders -> base verdict, escalations -> bumps, everything reports
    its reason. TODO(Bryan): citations for the chosen thresholds.
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
        "min_feels_like_c": pick("feels_like_c", min),
        "max_feels_like_c": pick("feels_like_c", max),
        "max_wind_kph": pick("wind_kph", max),
        "max_precip_mm": pick("precip_mm", max),
    }

    verdict, reasons = "OK", []
    # (value, ladder, colder_is_worse, label) -- heat uses feels-like
    # too: apparent_temperature folds in humidity, which is the part
    # of heat that kills dogs
    ladder_checks = [
        (window["min_feels_like_c"], COLD_LADDER, True, "feels-like low"),
        (window["max_feels_like_c"], HEAT_LADDER, False, "feels-like high"),
        (window["max_wind_kph"], WIND_LADDER, False, "wind"),
        (window["max_precip_mm"], PRECIP_LADDER, False, "precipitation"),
    ]
    for value, ladder, colder, label in ladder_checks:
        if hit := _walk_ladder(value, ladder, colder):
            threshold, rung = hit
            reasons.append(f"{label} {value:g} crosses {rung} threshold {threshold:g}")
            if VERDICTS.index(rung) > VERDICTS.index(verdict):
                verdict = rung

    for name, hits, why in ESCALATIONS:
        if hits(window):
            verdict = bump(verdict)
            reasons.append(f"{name}: {why}")

    return {"verdict": verdict, "reasons": reasons, "window": window}


def check_weather(
    lat: float, lon: float, date: str, start_hour: int = 8, end_hour: int = 20
) -> dict[str, Any]:
    """The tool the agent (and MCP facade) exposes: fetch + assess.

    Note the typed signature -- the 2024 version took one comma-packed
    string ("lat,lon,YYYY-MM-DD") because ReAct-era tools ate prose.
    """
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
                    "description": "walk window start, 0-23 local (default 8)",
                },
                "end_hour": {
                    "type": "integer",
                    "description": "walk window end, 0-23 local (default 20)",
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
# this; adding a tool means adding a function, a schema, and one row.
# ---------------------------------------------------------------------

REGISTRY: dict[str, tuple[Any, dict]] = {
    "check_weather": (check_weather, CHECK_WEATHER_SCHEMA),
    "geocode_addresses": (geocode_addresses, GEOCODE_SCHEMA),
    # "optimize_route": ...    (Phase 2, after that)
}
