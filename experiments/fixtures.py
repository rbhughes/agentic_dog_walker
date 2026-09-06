"""Phase-1 fixtures: scripted tool-call scenarios the bake-off runner
scores against each model.

A fixture is plain data: the exact conversation we send, the tools we
offer, and a declarative `expect` block the runner checks. No fixture
ever hits a real API -- when a scenario needs a tool result, we script
the result (same trick as lesson 1's fake blizzard).

Five fixtures, one per lesson-1 axis: call structure, relative dates,
over-eager calling, tool choice, and safety-flag fidelity.

LIMITATION -- prose checks are TRIAGE ONLY. `prose_mentions` /
`prose_mentions_any` grade free prose by substring, which is weak in
both directions: false FAILs (honest phrasings we didn't anticipate --
bit us on the first run) and false PASSes (negation-blind: "no danger
at all" matches "danger"). Tolerable for a 5-fixture bake-off where
every FAIL gets its transcript read; not a durable test strategy.
The real agent returns a STRUCTURED verdict (e.g. safety field), so
downstream fixtures grade field equality, and these prose checks
retire to smoke-test duty.
"""

from datetime import datetime, timedelta

NOW = datetime.now().astimezone()
TODAY = NOW.date()
TOMORROW = TODAY + timedelta(days=1)

SYSTEM = {
    "role": "system",
    "content": (
        f"Today is {TODAY:%A} {TODAY.isoformat()}. Pass dates as ISO YYYY-MM-DD."
    ),
}

# ---- tool schemas (shared by fixtures; the runner also validates the
# model's arguments against these with jsonschema) --------------------

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "check_weather",
        "description": "Get the weather forecast for a city on a date.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "date": {
                    "type": "string",
                    "description": "ISO date YYYY-MM-DD",
                },
            },
            "required": ["city", "date"],
            "additionalProperties": False,  # enforced by OUR runner, not the model
        },
    },
}

GEOCODE_TOOL = {
    "type": "function",
    "function": {
        "name": "geocode_address",
        "description": "Convert a street address to lat/lon coordinates.",
        "parameters": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "Full street address to geocode",
                },
            },
            "required": ["address"],
            "additionalProperties": False,
        },
    },
}

# ---- fixtures -------------------------------------------------------

FIXTURES = [
    {
        # short slug; shows up in the scorecard table
        "id": "basic-structure",
        # which lesson-1 axis this probes (documentation, not logic)
        "axis": "call structure + explicit date",
        # exactly what we send as `messages` on the first request
        "messages": [
            SYSTEM,
            {"role": "user", "content": "Should I walk Rex in Denver today?"},
        ],
        # exactly what we send as `tools`
        "tools": [WEATHER_TOOL],
        "expect": {
            # name of the tool the model MUST call -- or None, meaning it
            # must answer in prose WITHOUT calling anything (fixture 3's
            # whole point). The runner fails over-eager calls either way.
            "calls_tool": "check_weather",
            # arguments must validate against that tool's parameters
            # schema (jsonschema, additionalProperties and all)
            "args_valid": True,
            # subset of arguments that must match exactly. TODAY makes
            # the fixture true whenever it runs, never hardcode a date.
            "args_include": {"city": "Denver", "date": TODAY.isoformat()},
        },
        # OPTIONAL second round: if present, the runner appends the
        # model's reply + this scripted result, calls the model again,
        # and applies `final_expect` to the closing prose.
        "tool_result": {"temp_c": -21, "precip_mm": 40, "wind_kph": 30},
        "final_expect": {
            # every string here must appear in the final content
            # (case-insensitive) -- numeric fidelity, lesson 1 axis 2
            "prose_mentions": ["-21"],
        },
    },
    {
        "id": "relative-date",
        "axis": "relative date, system role",
        "messages": [
            SYSTEM,
            {"role": "user", "content": "Should I walk Rex in Denver tomorrow?"},
        ],
        "tools": [WEATHER_TOOL],
        "expect": {
            "calls_tool": "check_weather",
            "args_valid": True,
            "args_include": {"city": "Denver", "date": TOMORROW.isoformat()},
        },
    },
    {
        "id": "no-tool-trap",
        "axis": "over-eager tool calling",
        "messages": [
            SYSTEM,
            {
                "role": "user",
                "content": "Is a corgi a good apartment dog for a first-time owner?",
            },
        ],
        "tools": [WEATHER_TOOL],
        "expect": {"calls_tool": None},
    },
    {
        "id": "tool-choice",
        "axis": "select single appropriate tool",
        "messages": [
            SYSTEM,
            {
                "role": "user",
                # city stated explicitly: without one, the model would
                # have to invent a city (uninformed-required-field trap
                # again) and the fixture would test two things at once
                "content": "What is the temperature in Boise today?",
            },
        ],
        "tools": [WEATHER_TOOL, GEOCODE_TOOL],
        "expect": {
            # calls_tool is always the NAME string, never the tool dict
            "calls_tool": "check_weather",
            "args_valid": True,
            "args_include": {"city": "Boise"},
        },
    },
    {
        "id": "severity",
        "axis": "relaying a safety flag without softening it",
        # Design decision (resolving the open question from lesson 1):
        # we do NOT grade the model's own judgment of raw numbers --
        # that proved brittle, and the real system won't rely on it.
        # The real weather tool will compute deterministic safety
        # flags, so this fixture's scripted result includes the flag,
        # and we test the thing the app actually needs: does the model
        # RELAY the tool's verdict faithfully, or soften it into
        # "bring a coat"?
        "messages": [
            SYSTEM,
            {"role": "user", "content": "Should I walk Rex in Denver today?"},
        ],
        "tools": [WEATHER_TOOL],
        "expect": {
            "calls_tool": "check_weather",
            "args_valid": True,
        },
        "tool_result": {
            "temp_c": -21,
            "wind_kph": 30,
            "feels_like_c": -33,
            "safety": "DO_NOT_WALK",
            "safety_reason": "extreme cold; frostbite risk for dogs in minutes",
        },
        "final_expect": {
            # prose_mentions_any: at least ONE must appear (vs
            # prose_mentions where ALL must). Any honest refusal
            # phrasing passes; only softening the verdict fails.
            # first bake-off run taught us: models refuse in many
            # honest phrasings ("not safe", "avoid walking",
            # "DO_NOT_WALK" verbatim, "postponing") -- substring
            # grading of prose is leaky by nature; the real agent will
            # return a STRUCTURED verdict instead. Wide net for triage.
            "prose_mentions_any": [
                "go not walk",
                "don't walk",
                "not walk",
                "do_not_walk",
                "not safe",
                "unsafe",
                "avoid walk",
                "skip",
                "reschedul",
                "postpon",
                "stay in",
                "indoor",
                "danger",
            ],
        },
    },
]
