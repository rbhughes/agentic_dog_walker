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
# ling-3.0-flash won the 2026-09-14 measurement sweep outright: 35/35
# passes at ~$0.0004/plan and ~10s, cheapest and fastest of the field
# (qwen3-8b, the prior default, placed 5th of 6). See dog_walker.measure.
DEFAULT_MODEL = "inclusionai/ling-3.0-flash"

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
                "feasible": {
                    "type": "boolean",
                    "description": (
                        "false if the morning/afternoon walk windows "
                        "cannot all be arranged (optimize_route returned "
                        "feasible=false); explain in overall_advice"
                    ),
                },
                "route_summary": {"type": "string"},
                "overall_advice": {"type": "string"},
            },
            "required": ["walks", "feasible", "overall_advice"],
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
        # ask OpenRouter to return real accounting (token counts + the
        # actual dollar cost of THIS call) in the response's usage block;
        # the measurement harness sums it into cost-per-plan
        "usage": {"include": True},
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
    norm["_usage"] = resp.get("usage")  # {prompt_tokens, completion_tokens, cost}
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


def without_nulls(value):
    """Drop keys whose value is null, at every depth. Models routinely
    emit `null` for an optional field they mean to leave unset
    (max_relief_m: null); our tools already treat MISSING as the default,
    so null == absent. Stripping before validation stops a needless
    'None is not of type number' bounce loop -- measured: qwen3-8b
    livelocked 14 rounds on it -- and matches how the tools already read
    their inputs."""
    if isinstance(value, dict):
        return {k: without_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [without_nulls(v) for v in value]
    return value


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
    dict, the OpenAI dialect a JSON string. Null-stripped, same as the
    dispatch path: a model that fills an optional field with null
    (comfort_min_f: null) would otherwise reach an auditor as None and
    crash float() -- present-but-null defeats dict.get(key, default),
    which returns None, not the default. Measured: this took two
    gpt-oss-120b runs down as backend_error."""
    arguments = tool_call["function"]["arguments"]
    parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    return without_nulls(parsed)


def _clock_to_hours(hhmm: str) -> float:
    """'13:44' -> 13.73; the auditor compares hours as floats."""
    h, m = hhmm.split(":")
    return int(h) + int(m) / 60


def _latest_route_result(messages: list) -> dict | None:
    """The most recent optimize_route RESULT payload (feasible or not)."""
    found = None
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        try:
            payload = json.loads(msg.get("content") or "")
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and ("timeline" in payload or "feasible" in payload):
            found = payload
    return found


def audit_feasibility(messages: list, plan: dict) -> str | None:
    """The medications oracle: the plan's `feasible` flag must match
    what the route actually reported. Deterministic, no model.

    Both directions are enforced. Claiming success over an infeasible
    route is the fabrication we care about; claiming infeasible over a
    workable route is the loophole that would let a model skip every
    weather check by crying wolf. Neither passes.
    """
    route = _latest_route_result(messages)
    if route is None:
        return None  # no route yet; the weather auditor handles that
    route_feasible = bool(route.get("feasible", True))
    plan_feasible = plan.get("feasible")
    if not route_feasible and plan_feasible is not False:
        reason = route.get("reason", "a walk window cannot be arranged")
        return (
            f"REJECTED: the route is infeasible -- {reason}. Submit with "
            "feasible=false and say which walk window can't be met in "
            "overall_advice."
        )
    if route_feasible and plan_feasible is False:
        return (
            "REJECTED: the route IS feasible -- every walk window is "
            "arranged. Submit with feasible=true."
        )
    return None


def audit_terrain_coverage(messages: list) -> str | None:
    """The terrain oracle, parallel to weather coverage: every dog that
    declared a max_relief_m must have a check_terrain call at its own
    location carrying that exact tolerance. A terrain verdict computed
    against the wrong tolerance is the wrong verdict, so a mismatch is
    not coverage. Returns one message naming every gap, or None.
    """
    route_call, terrain_calls = None, []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if tc["function"]["name"] == "optimize_route":
                route_call = _call_args(tc)
            elif tc["function"]["name"] == "check_terrain":
                terrain_calls.append(_call_args(tc))
    if route_call is None:
        return None  # no route yet; the weather auditor handles that

    gaps = []
    for stop in route_call["stops"]:
        want = stop.get("max_relief_m")
        if want is None:
            continue
        lat, lon = stop["lat"], stop["lon"]

        def covered(c: dict) -> bool:
            return (
                abs(c.get("lat", 999) - lat) <= _NEAR_DEG
                and abs(c.get("lon", 999) - lon) <= _NEAR_DEG
                and abs(float(c.get("max_relief_m", -1)) - float(want)) < 0.5
            )

        if not any(covered(c) for c in terrain_calls):
            gaps.append(
                f"GAP: {stop['name']} has a hill tolerance: call "
                f"check_terrain with lat={lat}, lon={lon}, "
                f"max_relief_m={want:g}."
            )
    if gaps:
        return " ".join(gaps) + " Make ALL these calls, then submit again."
    return None


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
        from dog_walker.toolbox import DEFAULT_COMFORT_MAX_F, DEFAULT_COMFORT_MIN_F

        want_lo = float(entry.get("comfort_min_f", DEFAULT_COMFORT_MIN_F))
        want_hi = float(entry.get("comfort_max_f", DEFAULT_COMFORT_MAX_F))

        def covered(w: dict) -> bool:
            """Right place, covering hours, AND this dog's comfort
            band: a verdict computed against the wrong band is wrong,
            so a band mismatch = not covered."""
            near = (
                abs(w.get("lat", 999) - lat) <= _NEAR_DEG
                and abs(w.get("lon", 999) - lon) <= _NEAR_DEG
            )
            # check_weather windows are whole hours [start_hour, end_hour)
            in_time = w.get("start_hour", 8) <= math.floor(walk_a) and w.get(
                "end_hour", 20
            ) >= math.ceil(walk_b)
            band = (
                abs(float(w.get("comfort_min_f", DEFAULT_COMFORT_MIN_F)) - want_lo) < 0.5
                and abs(float(w.get("comfort_max_f", DEFAULT_COMFORT_MAX_F)) - want_hi) < 0.5
            )
            return near and in_time and band

        if not any(covered(w) for w in weather_calls):
            # Prescribe the EXACT call, band included even at the
            # default. Omitting it when the band is default let a
            # model that hallucinated a wrong band ([0,5]) livelock:
            # its check never matched and the gap never told it the
            # right band. Always state the target.
            gaps.append(
                f"GAP: {entry['stop']} walks {start}-{end}: call "
                f"check_weather with lat={lat}, lon={lon}, "
                f"start_hour={math.floor(walk_a)}, "
                f"end_hour={math.ceil(walk_b)}, "
                f"comfort_min_f={want_lo:g}, comfort_max_f={want_hi:g}."
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


def _accumulate_usage(totals: dict, usage: dict | None) -> None:
    """Fold one chat call's OpenRouter usage into the running totals."""
    if not usage:
        return
    totals["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
    totals["completion_tokens"] += usage.get("completion_tokens", 0) or 0
    totals["cost"] += usage.get("cost", 0.0) or 0.0


def run_events(request: str, model: str | None = None):
    """The agent: plan, act with validation, reflect via the auditor,
    finish through submit_plan. Yields events (vocabulary above).
    `model` overrides DEFAULT_MODEL (the service validates it against
    an allowlist before it gets here).

    The terminal event (final or error) carries `usage`
    (prompt_tokens/completion_tokens/cost, summed across every model
    round) and `rounds` (how many act rounds ran) -- the raw material
    for cost- and latency-per-plan in the measurement harness."""
    model = model or DEFAULT_MODEL
    today = datetime.now().astimezone()
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
    last_round = 0
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
                "location, passing that dog's comfort_min_f and "
                "comfort_max_f if stated. Include each dog's stated "
                "For any dog with a max_relief_m (hill tolerance), also "
                "call check_terrain at that dog's location with its "
                "max_relief_m. Include each dog's buffer_minutes, comfort "
                "band, needs_meds (boolean; adds handling time), and "
                "walk_window (any/morning/afternoon) in the optimize_route "
                "stops. If optimize_route returns feasible=false, the "
                "morning/afternoon windows cannot all be arranged: "
                "submit_plan with feasible=false and explain. Otherwise "
                "submit with feasible=true. Weather windows are whole "
                "hours with an "
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
        _accumulate_usage(usage, plan_reply.get("_usage"))
        messages.append(plan_reply["_raw"])
        if text := _plan_text(plan_reply):
            yield {"event": "plan", "text": text}
        pending = plan_reply.get("tool_calls") or []

        nudges = 0
        for round_no in range(1, MAX_ROUNDS + 1):
            last_round = round_no
            # ---- ACT on whatever calls are pending
            for call in pending:
                name = call["function"]["name"]
                # null == unset: strip before the referee and dispatch so
                # a model that fills optional fields with null isn't
                # bounced into a livelock (the _raw resent to the model is
                # untouched)
                arguments = without_nulls(call["function"]["arguments"])
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
                    # ---- REFLECT: deterministic oracles, each with a veto.
                    # Feasibility first: it settles whether the plan even
                    # claims the walks happen. An honestly-infeasible plan
                    # has no timeline to weather-check, so it stops here.
                    if gap := audit_feasibility(messages, arguments):
                        yield {"event": "audit_veto", "gap": gap}
                        messages.append(tool_feedback(call, {"error": gap}))
                        continue
                    if arguments.get("feasible") is False:
                        yield {"event": "final", "plan": arguments,
                               "usage": dict(usage), "rounds": last_round}
                        return
                    if gap := audit_weather_coverage(messages):
                        yield {"event": "audit_veto", "gap": gap}
                        messages.append(tool_feedback(call, {"error": gap}))
                        continue
                    if gap := audit_terrain_coverage(messages):
                        yield {"event": "audit_veto", "gap": gap}
                        messages.append(tool_feedback(call, {"error": gap}))
                        continue
                    yield {"event": "final", "plan": arguments,
                           "usage": dict(usage), "rounds": last_round}
                    return

                result = dispatch(name, arguments)
                yield {"event": "result", "name": name, "result": result}
                messages.append(tool_feedback(call, result))

            # ---- next model round (think off: plan is already in state)
            reply = chat(model, messages, think=False)
            _accumulate_usage(usage, reply.get("_usage"))
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
            "usage": dict(usage),
            "rounds": last_round,
        }
    except Exception as e:  # noqa: BLE001 -- stream boundary: fail as an event
        yield {"event": "error", "message": f"{type(e).__name__}: {e}",
               "usage": dict(usage), "rounds": last_round}


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

DEMOS = {
    # exercises every per-dog parameter, and stays feasible:
    # comfort band + meds + morning window (Daisy), buffer + hill
    # tolerance / terrain check (Ziggy), plain dog (Rex)
    "full": (
        "Plan this morning's walks starting and ending at 21 W Chestnut "
        "St, Chicago. "
        "Daisy is at Lincoln Park Zoo, Chicago, gets a 20 minute walk, is "
        "sensitive to cold so keep her comfortable between 45 and 95 F, "
        "needs her medication given (adds a little time), and should be "
        "walked in the morning. "
        "Ziggy is at Wrigley Field, Chicago, gets a 30 minute walk, needs "
        "15 minutes of prep time before the walk because of a slow "
        "elevator, and can't handle hills -- keep it to gentle slopes. "
        "Rex is at 5218 N Clark St, Chicago, gets a 60 minute walk."
    ),
    # too many hour-long morning walks to cluster before noon, even
    # leaving at 8:00 -> feasible=false (the walker chooses the start)
    "infeasible": (
        "Plan walks starting and ending at Millennium Park, Chicago. "
        "Every one of these needs a MORNING walk of 60 minutes: "
        "Rex at 5218 N Clark St, Chicago; "
        "Daisy at Lincoln Park Zoo, Chicago; "
        "Ziggy at Wrigley Field, Chicago; "
        "Pip at Montrose Beach, Chicago."
    ),
}

if __name__ == "__main__":
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else "full"
    request = DEMOS.get(arg, arg)  # a demo key, or a custom request string
    result = run(request)
    print(json.dumps(result, indent=2))
