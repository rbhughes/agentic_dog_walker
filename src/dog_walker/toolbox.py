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

    TODO(Bryan): this is yours -- the thresholds ARE the product.
    Sketch of the questions your if-statements must answer:
      * at what feels-like temperature does CAUTION become SHORTEN
        become DO_NOT_WALK? (paw injury and frostbite for dogs start
        surprisingly warm; look up guidance you trust and cite it in
        a comment)
      * heat: hot pavement and heatstroke -- where are those lines?
      * precipitation: rain is CAUTION-ish, but freezing rain?
      * wind: when does it amplify cold (use feels_like) vs stand
        alone as a hazard (debris, stress)?
      * combinations: does cold + wet deserve a bump the individual
        numbers don't trigger?
    Keep every rule one `if` + one reasons.append(...) so each is
    individually testable.
    """
    raise NotImplementedError("TODO(Bryan): thresholds")


def check_weather(lat: float, lon: float, date: str,
                  start_hour: int = 8, end_hour: int = 20) -> dict[str, Any]:
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
# registry: name -> (callable, schema). Agent + MCP facade both read
# this; adding a tool means adding a function, a schema, and one row.
# ---------------------------------------------------------------------

REGISTRY: dict[str, tuple[Any, dict]] = {
    "check_weather": (check_weather, CHECK_WEATHER_SCHEMA),
    # "geocode_address": ...   (Phase 2, next step)
    # "optimize_route": ...    (Phase 2, after that)
}
