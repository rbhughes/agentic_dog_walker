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

Reading order for annotation: SUBMIT_PLAN_SCHEMA (the finish line),
validate_call (the referee), run (the loop), audit_weather_coverage
(the auditor). chat/dispatch/tool_feedback are wire plumbing.
Known shape-change ahead: Phase 4 turns the print-based trace into
an event stream for the web UI, so run()'s internals will move.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
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
# one backend, one dialect: OpenRouter's OpenAI-compatible API.
# (The Ollama dialect was retired 2026-09-11 -- fossil serves the
# service, not inference. If local inference ever returns, Ollama
# speaks this same dialect at /v1/chat/completions.)
# ---------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3-8b"

MAX_ROUNDS = 14


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
# wire plumbing
# ---------------------------------------------------------------------


def chat(model: str, messages: list, think: bool = False) -> dict:
    """One model round. Returns the normalized message:
    {"role", "content", "tool_calls": [arguments as dicts], "_raw"}.
    _raw is the wire-format original -- always resend THAT."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("set OPENROUTER_API_KEY in env or .env")
    body = {
        "model": model,
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
    # Armor: one model round gets 3 attempts with short backoff.
    # Retryable: network faults, timeouts, rate limits (429), and
    # server-side errors (5xx). Client errors (4xx) are OUR bug --
    # fail fast so they surface.
    req = urllib.request.Request(OPENROUTER_URL, json.dumps(body).encode(), headers)
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=180))
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                last_error = e
            elif e.code == 400 and body.pop("reasoning", None) is not None:
                # some models reject the reasoning block outright;
                # drop it and retry once without
                req = urllib.request.Request(
                    OPENROUTER_URL, json.dumps(body).encode(), headers
                )
                last_error = e
            else:
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = e
        time.sleep(2**attempt)  # 1s, 2s, 4s between attempts
    else:
        raise RuntimeError(f"model backend unreachable after 4 attempts: {last_error}")

    raw = resp["choices"][0]["message"]
    norm = dict(raw)  # shallow copy; passenger fields ride through
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


def tool_feedback(call: dict, result: dict) -> dict:
    """Package a result as the tool message answering one call."""
    return {
        "role": "tool",
        "tool_call_id": call["id"],
        "content": json.dumps(result),
    }


# ---------------------------------------------------------------------
# the referee: nobody else enforces the schemas
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
# exactly the start-location-only failure the auditor exists to catch.
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
        # A model that never routes could otherwise submit FABRICATED
        # times and verdicts (observed live: gemini-2.5-flash-lite
        # geocoded, then invented walk times and OK verdicts wholesale).
        # No timeline, no acceptance.
        return (
            "REJECTED: no route exists. Call optimize_route with the "
            "stops and start_time, then check_weather for each dog's "
            "walk interval, then submit again."
        )

    # stop name -> coordinates, from the route call's own arguments
    coords = {s["name"]: (s["lat"], s["lon"]) for s in route_call["stops"]}

    gaps: list[str] = []
    for entry in timeline:
        if not entry.get("walk_minutes"):
            continue  # the start stop / return home
        start, end = entry.get("walk_start"), entry.get("walk_end")
        if not (isinstance(start, str) and ":" in start):
            # minutes-from-start timeline: intervals unauditable, and
            # unauditable must not mean unaudited
            return (
                "REJECTED: the route timeline has no clock times. Call "
                "optimize_route again INCLUDING start_time, re-check "
                "weather for each dog's interval, then submit again."
            )
        lat, lon = coords.get(entry["stop"], (None, None))
        if lat is None:
            continue
        walk_a, walk_b = _clock_to_hours(start), _clock_to_hours(end)
        want_cold = int(entry.get("cold_tolerance", 0))
        want_heat = int(entry.get("heat_tolerance", 0))

        def covered(w: dict) -> bool:
            """Right place, covering hours, AND this dog's tolerances:
            a delicate dog's verdict from a default-tolerance check
            would be wrong, so tolerance mismatch = not covered."""
            near = (
                abs(w.get("lat", 999) - lat) <= _NEAR_DEG
                and abs(w.get("lon", 999) - lon) <= _NEAR_DEG
            )
            # check_weather windows are whole hours [start_hour, end_hour)
            in_time = w.get("start_hour", 8) <= math.floor(walk_a) and w.get(
                "end_hour", 20
            ) >= math.ceil(walk_b)
            tolerances = (
                int(w.get("cold_tolerance", 0)) == want_cold
                and int(w.get("heat_tolerance", 0)) == want_heat
            )
            return near and in_time and tolerances

        if not any(covered(w) for w in weather_calls):
            # Prescribe the exact call, don't just name the gap
            # (vague messages livelocked a model once).
            extra = ""
            if want_cold:
                extra += f", cold_tolerance={want_cold}"
            if want_heat:
                extra += f", heat_tolerance={want_heat}"
            gaps.append(
                f"GAP: {entry['stop']} walks {start}-{end}: call "
                f"check_weather with lat={lat}, lon={lon}, "
                f"start_hour={math.floor(walk_a)}, "
                f"end_hour={math.ceil(walk_b)}{extra}."
            )
    if gaps:
        return " ".join(gaps) + " Make ALL these calls, then submit again."
    return None


# ---------------------------------------------------------------------
# 3: the loop -- as an EVENT STREAM.
#
# run_events() is the agent: a generator yielding one structured event
# per observable moment. Every consumer -- the CLI wrapper below, the
# web service's SSE endpoint, offline tests with a scripted backend --
# reads the same stream. The event vocabulary:
#
#   {"event": "start",      "model": ..., "backend": ...}
#   {"event": "plan",       "text": ...}          # the written plan
#   {"event": "call",       "round": n, "name": ..., "arguments": {...}}
#   {"event": "bounce",     "name": ..., "error": ...}   # referee refusal
#   {"event": "result",     "name": ..., "result": {...}}
#   {"event": "audit_veto", "gap": ...}           # auditor refusal
#   {"event": "nudge",      "round": n}           # prose without a finish
#   {"event": "final",      "plan": {...}}        # validated submit_plan args
#   {"event": "error",      "message": ...}       # terminal failure
#
# A stream always ends with exactly one "final" or one "error".
# ---------------------------------------------------------------------


def _brief(value: Any, limit: int = 120) -> str:
    """One-line summary for the CLI trace."""
    text = value if isinstance(value, str) else json.dumps(value)
    return text[:limit] + ("..." if len(text) > limit else "")


def _plan_text(reply: dict) -> str:
    """The plan, wherever this backend put it: OpenRouter uses
    `reasoning`, Ollama uses `thinking`, models without a reasoning
    channel narrate in `content`. First non-empty wins."""
    for key in ("reasoning", "thinking", "content"):
        if reply.get(key):
            return reply[key]
    return ""


def run_events(request: str, model: str | None = None):
    """The agent: plan, act with validation, reflect via the auditor,
    finish through submit_plan. Yields events (vocabulary above).
    `model` overrides DEFAULT_MODEL (the service validates it against
    an allowlist before it gets here)."""
    model = model or DEFAULT_MODEL
    today = datetime.now().astimezone()
    yield {"event": "start", "model": model, "backend": "openrouter"}

    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                f"You are a dog-walking planner. Today is {today:%A} "
                f"{today:%Y-%m-%d}. Call tools directly, without asking "
                "permission. Batch all addresses into ONE geocode call. "
                "Weather must be verified for each dog's ACTUAL walk "
                "interval from the route timeline, at that dog's "
                "location, passing that dog's cold_tolerance and "
                "heat_tolerance if stated. Include each dog's stated "
                "buffer_minutes and tolerances in the optimize_route "
                "stops. Weather windows are whole hours with an "
                "EXCLUSIVE end: a walk 13:29-13:59 is start_hour 13, "
                "end_hour 14. Finish by calling submit_plan exactly "
                "once. Dates are ISO YYYY-MM-DD. Be concise."
            ),
        },
        {"role": "user", "content": request},
    ]

    try:
        # ---- PLAN: one reasoning-mode round (measured: think-on plans
        # per-dog; think-off doesn't). The reply joins the state so all
        # later think-off rounds see it. If it already proposes calls,
        # the act machinery below handles them -- planning and acting
        # are allowed to overlap.
        plan_reply = chat(model, messages, think=True)
        messages.append(plan_reply["_raw"])
        if text := _plan_text(plan_reply):
            yield {"event": "plan", "text": text}
        pending = plan_reply.get("tool_calls") or []

        nudges = 0
        for round_no in range(1, MAX_ROUNDS + 1):
            # ---- ACT on whatever calls are pending
            for call in pending:
                name = call["function"]["name"]
                arguments = call["function"]["arguments"]
                yield {
                    "event": "call",
                    "round": round_no,
                    "name": name,
                    "arguments": arguments,
                }

                # the referee, BEFORE anything runs
                if error := validate_call(name, arguments):
                    yield {"event": "bounce", "name": name, "error": error}
                    messages.append(tool_feedback(call, {"error": error}))
                    continue

                if name == "submit_plan":
                    # ---- REFLECT: the deterministic auditor gets a veto
                    if gap := audit_weather_coverage(messages):
                        yield {"event": "audit_veto", "gap": gap}
                        messages.append(tool_feedback(call, {"error": gap}))
                        continue
                    yield {"event": "final", "plan": arguments}
                    return

                result = dispatch(name, arguments)
                yield {"event": "result", "name": name, "result": result}
                messages.append(tool_feedback(call, result))

            # ---- next model round (think off: plan is already in state)
            reply = chat(model, messages, think=False)
            messages.append(reply["_raw"])
            pending = reply.get("tool_calls") or []

            # prose without a finish is not an answer -- nudge, twice max
            if not pending:
                if nudges >= 2:
                    break
                nudges += 1
                yield {"event": "nudge", "round": round_no}
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Finish by calling submit_plan with the "
                            "structured walk plan."
                        ),
                    }
                )

        yield {
            "event": "error",
            "message": (
                f"no submit_plan within {MAX_ROUNDS} rounds; last message: "
                f"{_brief(messages[-1].get('content') or '', 200)}"
            ),
        }
    except Exception as e:  # noqa: BLE001 -- stream boundary: fail as an event
        yield {"event": "error", "message": f"{type(e).__name__}: {e}"}


def run(request: str, model: str | None = None, verbose: bool = True) -> dict:
    """CLI-flavoured consumer of run_events(): prints a human trace,
    returns the validated plan, raises on a terminal error."""

    def trace(text: str) -> None:
        if verbose:
            print(text)

    for ev in run_events(request, model):
        kind = ev["event"]
        if kind == "plan":
            trace(f"[plan] {_brief(ev['text'], 200)}")
        elif kind == "call":
            trace(f"[round {ev['round']}] {ev['name']}({_brief(ev['arguments'])})")
        elif kind == "bounce":
            trace(f"    bounce: {ev['error']}")
        elif kind == "audit_veto":
            trace(f"    audit: {ev['gap']}")
        elif kind == "result":
            trace(f"    -> {_brief(ev['result'])}")
        elif kind == "nudge":
            trace(f"[round {ev['round']}] prose without submit_plan; nudging")
        elif kind == "final":
            trace("    accepted.")
            return ev["plan"]
        elif kind == "error":
            raise RuntimeError(ev["message"])
    raise RuntimeError("event stream ended without final or error")


# ---------------------------------------------------------------------
# CLI harness: `uv run python -m dog_walker.agent "<request>"`
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    demo = (
        "Plan this afternoon's walks starting and ending at 21 W Chestnut St, Chicago "
        "Daisy is at Lincoln Park Zoo, Chicago (20 minutes)"
        "Ziggy is at Wrigley Field, Chicago (30 minutes). "
        "Rex is at 5218 N Clark St, Chicago (60 minutes). "
        "I leave at 9:00."
    )
    result = run(sys.argv[1] if len(sys.argv) > 1 else demo)
    print(json.dumps(result, indent=2))
