"""The agent: a hand-rolled plan / act / reflect loop over the toolbox.

No framework. The loop is a while-loop around a message list; the
judgment lives in three places:
  plan    -- one reasoning-mode call writes the plan into the state
  act     -- validate every tool call against REGISTRY schemas BEFORE
             dispatch; validation errors bounce back as tool results
  reflect -- a deterministic auditor reads the state and injects a
             corrective message when a dog's walk interval lacks a
             covering weather check (code notices; model acts)

The loop ends when the model calls submit_plan -- the structured
finish line. Its arguments ARE the final answer; prose is garnish.

Provided: chat() (multi-backend, from lesson 2), dispatch(),
tool_feedback(), the submit_plan schema, the CLI harness.
TODO(Bryan): validate_call(), run() -- the loop itself -- and
audit_weather_coverage(). That's the learning core.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime
from typing import Any

import jsonschema

from dog_walker.toolbox import REGISTRY

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------
# backends (settled in lesson 1/2; LOCAL_LLM=fossil selects local)
# ---------------------------------------------------------------------

BACKENDS = {
    "fossil": {
        "style": "ollama",
        "url": "http://fossil:11434/api/chat",
        "model": "qwen3:8b",
    },
    "openrouter": {
        "style": "openai",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "qwen/qwen3-8b",
        "key_env": "OPENROUTER_API_KEY",
    },
}

MAX_ROUNDS = 10


# ---------------------------------------------------------------------
# the structured finish line: the model ENDS by calling this "tool".
# It has no implementation -- its validated arguments are the answer.
# ---------------------------------------------------------------------

SUBMIT_PLAN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": (
            "Submit the finished walk plan. Call this exactly once, as "
            "the last step, after weather is verified for every dog's "
            "actual walk interval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "walks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pet": {"type": "string"},
                            "walk_start": {"type": "string"},
                            "walk_end": {"type": "string"},
                            "verdict": {
                                "type": "string",
                                "enum": ["OK", "CAUTION", "SHORTEN", "DO_NOT_WALK"],
                            },
                            "notes": {"type": "string"},
                        },
                        "required": ["pet", "walk_start", "walk_end", "verdict"],
                        "additionalProperties": False,
                    },
                },
                "route_summary": {"type": "string"},
                "overall_advice": {"type": "string"},
            },
            "required": ["walks", "overall_advice"],
            "additionalProperties": False,
        },
    },
}

TOOLS = [schema for _fn, schema in REGISTRY.values()] + [SUBMIT_PLAN_SCHEMA]


# ---------------------------------------------------------------------
# provided plumbing (understood in lessons 1-2)
# ---------------------------------------------------------------------


def chat(backend: dict, messages: list, think: bool = False) -> dict:
    """One model round, normalized across the two wire dialects.
    Returns {"role", "content", "tool_calls": [args as dicts], "_raw"}."""
    if backend["style"] == "ollama":
        body = {
            "model": backend["model"],
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
            "think": think,
            "options": {"num_thread": 10, "temperature": 0},
        }
        headers = {"Content-Type": "application/json"}
    else:
        key = os.environ.get(backend["key_env"], "")
        if not key:
            raise RuntimeError(f"set {backend['key_env']} in env or .env")
        body = {
            "model": backend["model"],
            "messages": messages,
            "tools": TOOLS,
            "temperature": 0,
            "max_tokens": 2000 if think else 700,
            "reasoning": {"enabled": think},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
    req = urllib.request.Request(backend["url"], json.dumps(body).encode(), headers)
    resp = json.load(urllib.request.urlopen(req, timeout=300))
    if backend["style"] == "ollama":
        raw = resp["message"]
        norm = dict(raw)
    else:
        raw = resp["choices"][0]["message"]
        norm = dict(raw)
        norm["tool_calls"] = [
            {
                "id": tc.get("id"),
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"]["arguments"]),
                },
            }
            for tc in (raw.get("tool_calls") or [])
        ]
    norm["_raw"] = raw
    return norm


def dispatch(name: str, arguments: dict) -> dict:
    """Execute one validated tool call; failures become error results."""
    fn = REGISTRY[name][0]
    try:
        return fn(**arguments)
    except Exception as e:  # noqa: BLE001 -- boundary: everything becomes data
        return {"error": f"{type(e).__name__}: {e}"}


def tool_feedback(backend: dict, call: dict, result: dict) -> dict:
    """Package a result in the backend's dialect (lesson-1 notes)."""
    msg = {"role": "tool", "content": json.dumps(result)}
    if backend["style"] == "openai":
        msg["tool_call_id"] = call["id"]
    return msg


# ---------------------------------------------------------------------
# the referee: nobody else enforces the schemas (lesson-1 finding)
# ---------------------------------------------------------------------

# every schema the model may call, submit_plan included
_ALL_SCHEMAS: dict[str, dict] = {
    **{name: schema for name, (_fn, schema) in REGISTRY.items()},
    "submit_plan": SUBMIT_PLAN_SCHEMA,
}


def validate_call(name: str, arguments: dict) -> str | None:
    """None if the call is legal; else a short error the MODEL can act
    on -- these strings are prompts, not log lines. A validation
    failure never raises: it becomes the tool's 'result' and the model
    corrects itself next round (the bounce)."""
    if name not in _ALL_SCHEMAS:
        return f"unknown tool {name!r}; available: {sorted(_ALL_SCHEMAS)}"
    try:
        jsonschema.validate(arguments, _ALL_SCHEMAS[name]["function"]["parameters"])
    except jsonschema.ValidationError as e:
        # e.json_path pinpoints WHERE ($.stops[1].walk_minutes);
        # e.message says WHAT (45 is not one of [0, 20, 30, 60])
        return f"invalid arguments for {name}: {e.json_path}: {e.message}"
    return None


# ---------------------------------------------------------------------
# the auditor: deterministic reflection. Code notices what the model
# omits (models are bad at noticing absences); the model then acts on
# a specific, injected instruction (what models are good at).
# ---------------------------------------------------------------------

# ~1 km at these latitudes. First draft was 0.03 (~3 km) and a test
# caught it: a check at the start location "covered" a dog 3 km away,
# exactly the lesson-2 failure the auditor exists to catch.
_NEAR_DEG = 0.01


def _call_args(tool_call: dict) -> dict:
    """Arguments from a RAW tool call, either dialect: Ollama stores a
    dict, the OpenAI dialect a JSON string."""
    arguments = tool_call["function"]["arguments"]
    return json.loads(arguments) if isinstance(arguments, str) else arguments


def _clock_to_hours(hhmm: str) -> float:
    """'13:44' -> 13.73; the auditor compares hours as floats."""
    h, m = hhmm.split(":")
    return int(h) + int(m) / 60


def audit_weather_coverage(messages: list) -> str | None:
    """None if every dog's walk interval has a covering weather check,
    else ONE corrective sentence to inject. One gap per audit -- the
    loop comes back around for the next.

    Pure code, no model: reads the raw conversation, reconstructs the
    facts, compares. Only auditable once a route timeline with clock
    times exists (i.e. optimize_route was called with start_time).
    """
    import math

    # reconstruct: every tool CALL the assistant made, in order
    route_call, weather_calls = None, []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            name = tc["function"]["name"]
            if name == "optimize_route":
                route_call = _call_args(tc)
            elif name == "check_weather":
                weather_calls.append(_call_args(tc))

    # ...and the latest route RESULT carrying a timeline
    timeline = None
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        try:
            payload = json.loads(msg.get("content") or "")
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "timeline" in payload:
            timeline = payload["timeline"]

    if route_call is None or timeline is None:
        return None  # nothing to audit yet

    # stop name -> coordinates, from the route call's own arguments
    coords = {s["name"]: (s["lat"], s["lon"]) for s in route_call["stops"]}

    for entry in timeline:
        if not entry.get("walk_minutes"):
            continue  # the start stop / return home
        start, end = entry.get("walk_start"), entry.get("walk_end")
        if not (isinstance(start, str) and ":" in start):
            return None  # minutes-from-start timeline: hours unknowable
        lat, lon = coords.get(entry["stop"], (None, None))
        if lat is None:
            continue
        walk_a, walk_b = _clock_to_hours(start), _clock_to_hours(end)

        def covered(w: dict) -> bool:
            """Does this weather call cover the walk in space and time?"""
            near = (
                abs(w.get("lat", 999) - lat) <= _NEAR_DEG
                and abs(w.get("lon", 999) - lon) <= _NEAR_DEG
            )
            # check_weather windows are whole hours [start_hour, end_hour)
            in_time = w.get("start_hour", 8) <= math.floor(walk_a) and w.get(
                "end_hour", 20
            ) >= math.ceil(walk_b)
            return near and in_time

        if not any(covered(w) for w in weather_calls):
            # Prescribe the exact call, don't just name the gap: first
            # draft said only "no call covers that interval" and the
            # model livelocked, re-making the same too-short check
            # three times. The auditor knows the needed window --
            # say it.
            return (
                f"GAP: {entry['stop']} walks {start}-{end}. Call "
                f"check_weather with lat={lat}, lon={lon}, "
                f"start_hour={math.floor(walk_a)}, "
                f"end_hour={math.ceil(walk_b)}, then submit again."
            )
    return None


# ---------------------------------------------------------------------
# TODO(Bryan) 3: the loop
# ---------------------------------------------------------------------


def _brief(value: Any, limit: int = 120) -> str:
    """One-line summary for the trace printout."""
    text = value if isinstance(value, str) else json.dumps(value)
    return text[:limit] + ("..." if len(text) > limit else "")


def run(request: str, backend_name: str | None = None, verbose: bool = True) -> dict:
    """The agent: plan, then act with validation, reflect via the
    auditor, finish through submit_plan. Returns the validated
    submit_plan arguments -- structured data, never parsed prose.
    """
    backend = BACKENDS[backend_name or os.environ.get("LOCAL_LLM") or "openrouter"]
    today = datetime.now().astimezone()

    def trace(text: str) -> None:
        if verbose:
            print(text)

    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                f"You are a dog-walking planner. Today is {today:%A} "
                f"{today:%Y-%m-%d}. Call tools directly, without asking "
                "permission. Batch all addresses into ONE geocode call. "
                "Weather must be verified for each dog's ACTUAL walk "
                "interval from the route timeline, at that dog's "
                "location. Finish by calling submit_plan exactly once. "
                "Dates are ISO YYYY-MM-DD. Be concise."
            ),
        },
        {"role": "user", "content": request},
    ]

    # ---- PLAN: one reasoning-mode round. Measured in lesson 2:
    # think-on plans tools per-dog; think-off doesn't. The reply joins
    # the state so every later (cheaper, think-off) round sees it.
    # If the plan round already proposes tool calls, they are handled
    # by the same act machinery below -- planning and acting are
    # allowed to overlap.
    plan_reply = chat(backend, messages, think=True)
    messages.append(plan_reply["_raw"])
    if plan_reply.get("content"):
        trace(f"[plan] {_brief(plan_reply['content'], 200)}")
    pending = plan_reply.get("tool_calls") or []

    nudges = 0
    for round_no in range(1, MAX_ROUNDS + 1):
        # ---- ACT on whatever calls are pending
        for call in pending:
            name = call["function"]["name"]
            arguments = call["function"]["arguments"]
            trace(f"[round {round_no}] {name}({_brief(arguments)})")

            # the referee, BEFORE anything runs
            error = validate_call(name, arguments)
            if error:
                trace(f"    bounce: {error}")
                messages.append(tool_feedback(backend, call, {"error": error}))
                continue

            if name == "submit_plan":
                # ---- REFLECT: the deterministic auditor gets a veto
                gap = audit_weather_coverage(messages)
                if gap:
                    trace(f"    audit: {gap}")
                    messages.append(tool_feedback(backend, call, {"error": gap}))
                    continue
                trace("    accepted.")
                return arguments  # the structured finish line

            result = dispatch(name, arguments)
            trace(f"    -> {_brief(result)}")
            messages.append(tool_feedback(backend, call, result))

        # ---- next model round (think off: plan is already in state)
        reply = chat(backend, messages, think=False)
        messages.append(reply["_raw"])
        pending = reply.get("tool_calls") or []

        # prose without a finish is not an answer -- nudge, twice max
        if not pending:
            if nudges >= 2:
                break
            nudges += 1
            trace(f"[round {round_no}] prose without submit_plan; nudging")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Finish by calling submit_plan with the "
                        "structured walk plan."
                    ),
                }
            )

    raise RuntimeError(
        f"no submit_plan within {MAX_ROUNDS} rounds; last message: "
        f"{_brief(messages[-1].get('content') or '', 200)}"
    )


# ---------------------------------------------------------------------
# CLI harness: `uv run python -m dog_walker.agent "<request>"`
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    demo = (
        "Plan this afternoon's walks starting and ending at Wrigley "
        "Field, Chicago. Daisy is at Lincoln Park Zoo, Chicago (20 "
        "minutes). Rex is at 5218 N Clark St, Chicago (60 minutes). "
        "I leave at 13:00."
    )
    result = run(sys.argv[1] if len(sys.argv) > 1 else demo)
    print(json.dumps(result, indent=2))
