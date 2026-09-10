"""Preset demo rosters for the public site.

Why presets exist (owner's decision, 2026-09-10): anonymous visitors
get curated demos, never free text. Each preset ships with
pre-resolved coordinates, and seed_geocode_cache() plants them in the
geocoder's cache at service start -- so preset runs cost ZERO
Nominatim requests no matter how many visitors click. Only the
structured custom mode can trigger real geocoding, and the service
rate-limits that.

Coordinates were resolved once via the geocode tool itself and frozen
here. If an address changes, re-resolve; don't hand-edit.
"""

from dog_walker.toolbox import _geocode_cache

PRESETS: dict[str, dict] = {
    "lakeview-classic": {
        "title": "Lakeview classic",
        "description": (
            "Three dogs across Chicago's north side, out of Wrigley "
            "Field at 1pm. The route that built this project."
        ),
        "start_address": "Wrigley Field, Chicago",
        "start_time": "13:00",
        "pets": [
            {
                "name": "Daisy",
                "address": "Lincoln Park Zoo, Chicago",
                "walk_minutes": 20,
            },
            {"name": "Rex", "address": "5218 N Clark St, Chicago", "walk_minutes": 60},
            {
                "name": "Wilbur",
                "address": "4802 N Broadway, Chicago",
                "walk_minutes": 30,
            },
        ],
    },
    "loop-lunch-hour": {
        "title": "Loop lunch hour",
        "description": (
            "Two downtown dogs on a tight noon window, out of Millennium Park."
        ),
        "start_address": "Millennium Park, Chicago",
        "start_time": "12:00",
        "pets": [
            {
                "name": "Mochi",
                "address": "Buckingham Fountain, Chicago",
                "walk_minutes": 20,
            },
            {
                "name": "Peanut",
                "address": "Shedd Aquarium, Chicago",
                "walk_minutes": 30,
            },
        ],
    },
}

# address (lowercased, stripped -- the geocoder's cache key) -> hit.
# Frozen from live Nominatim lookups so preset runs never re-ask.
_FROZEN_GEOCODES: dict[str, dict] = {
    "wrigley field, chicago": {
        "address": "Wrigley Field, Chicago",
        "lat": 41.9481846,
        "lon": -87.655559,
        "display_name": "Wrigley Field, 1060 W Addison St, Chicago",
    },
    "lincoln park zoo, chicago": {
        "address": "Lincoln Park Zoo, Chicago",
        "lat": 41.9212558,
        "lon": -87.6337656,
        "display_name": "Lincoln Park Zoo, 2001 N Clark St, Chicago",
    },
    "5218 n clark st, chicago": {
        "address": "5218 N Clark St, Chicago",
        "lat": 41.9764363,
        "lon": -87.6685282,
        "display_name": "5218 N Clark St, Andersonville, Chicago",
    },
    "4802 n broadway, chicago": {
        "address": "4802 N Broadway, Chicago",
        "lat": 41.9691831,
        "lon": -87.6598913,
        "display_name": "Green Mill Cocktail Lounge, 4802 N Broadway, Chicago",
    },
    "millennium park, chicago": {
        "address": "Millennium Park, Chicago",
        "lat": 41.8825754,
        "lon": -87.6225361,
        "display_name": "Millennium Park, 201 E Randolph St, Chicago",
    },
    "buckingham fountain, chicago": {
        "address": "Buckingham Fountain, Chicago",
        "lat": 41.875802,
        "lon": -87.6189718,
        "display_name": "Buckingham Fountain, 301 S Columbus Dr, Chicago",
    },
    "shedd aquarium, chicago": {
        "address": "Shedd Aquarium, Chicago",
        "lat": 41.8676361,
        "lon": -87.6136789,
        "display_name": "Shedd Aquarium, 1200 S Lakefront Trail, Chicago",
    },
}


def seed_geocode_cache() -> int:
    """Plant the frozen coordinates in the geocoder's cache. Returns
    how many entries were seeded. Call once at service startup."""
    _geocode_cache.update(_FROZEN_GEOCODES)
    return len(_FROZEN_GEOCODES)


def build_request(start_address: str, start_time: str, pets: list[dict]) -> str:
    """Render a structured request (preset or custom form) into the
    prompt string run_events() takes. One sentence per fact -- the
    language boundary of the sandwich, kept deliberately dull."""
    lines = [
        f"Plan today's dog walks starting and ending at {start_address}. "
        f"I leave at {start_time}."
    ]
    for pet in pets:
        lines.append(
            f"{pet['name']} is at {pet['address']} and gets a "
            f"{pet['walk_minutes']} minute walk."
        )
    return " ".join(lines)
