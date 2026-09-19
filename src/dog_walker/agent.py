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

import http.client as http_client
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
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
# Resilience budget. The design that failed twice: a per-CALL timeout is the
# wrong knob to make aggressive, because a slow-but-working model needs a
# generous WHOLE-RUN budget while a genuinely hung endpoint must still be
# bounded. So we separate them: chat() gets a generous per-call read timeout
# AND an absolute run deadline (passed in by run_events); it retries with
# EXPONENTIAL backoff honoring OpenRouter's Retry-After on 429s, but never
# sleeps or reads past the deadline. Total run time is therefore bounded by
# AGENT_RUN_BUDGET_S no matter how many providers throttle us.
CHAT_TIMEOUT_S = 120       # per-call read timeout (room for one slow round)
# Output caps. ACT was 700 until 2026-09-17: in the overnight sweep 16 of 18
# malformed_output failures were grand-tour (6 dogs), across all five models
# that hit it, with cut-off-JSON errors -- the signature of a reply truncated
# at the cap. finish reasons are now recorded per attempt, and a live
# grand-tour check (ling-3.0-flash) proved it: 4 of 12 good act replies were
# 757-1742 tokens, and one reply hit a 2000 cap four times in a row. Output
# tokens are billed as produced, so a generous cap costs nothing unless used.
PLAN_MAX_TOKENS = 4000
ACT_MAX_TOKENS = 4000
# Values providers use for "stopped at max_tokens". Check native_finish_reason
# too: in that check OpenRouter normalized all four cut-off replies to
# "tool_calls" while the provider's native reason said "length".
_CAP_REASONS = {"length", "max_tokens", "max_output_tokens"}


def _hit_cap(a: dict) -> bool:
    return any(str(a.get(k) or "").lower() in _CAP_REASONS
               for k in ("finish_reason", "native_finish_reason"))
CHAT_ATTEMPTS = 6          # patient with 429s -- backoff clears most
CHAT_BACKOFF_BASE_S = 4    # 4, 8, 16, 32, 60... (>=2x the old 1s, exponential)
CHAT_BACKOFF_MAX_S = 60    # cap a single sleep
AGENT_RUN_BUDGET_S = 300   # absolute wall-clock per run; bounds ALL retries
# CAVEAT (verified 2026-09-16): CHAT_TIMEOUT_S bounds each socket READ, not
# the whole call. OpenRouter answers non-streaming requests with headers in
# ~1s, then whitespace keep-alive chunks while the model works, then the JSON.
# Every chunk resets the read timer, so a slow generation is never cut off by
# it, and the run deadline is only checked BETWEEN calls. A single call can
# therefore run far past both budgets. The timing diagnostics below exist to
# show where that time goes.

# ---------------------------------------------------------------------
# diagnostics: a live, append-only call log. The measurement harness sets
# CALL_LOG (and per-run CALL_LOG_CONTEXT); the web service leaves it None.
# Lines are written BEFORE each request as well as after, so a call that
# hangs is visible while it hangs, not only once it returns.
# ---------------------------------------------------------------------

CALL_LOG: Path | None = None
CALL_LOG_CONTEXT: dict = {}


def _log_call(record: dict) -> None:
    if CALL_LOG is None:
        return
    line = {"ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            **CALL_LOG_CONTEXT, **record}
    try:
        with open(CALL_LOG, "a") as fh:
            fh.write(json.dumps(line) + "\n")
    except OSError:
        pass  # diagnostics must never break a run


def _host_sleep_s(wall0: float, mono0: float) -> float:
    """Seconds the machine spent asleep since (wall0, mono0). macOS's
    monotonic clock stops during sleep while wall time keeps going, so the
    gap is sleep. Verified 2026-09-17: the overnight sweep recorded 5.48h of
    run time across 18.8h of wall time; the Mac was awake 5.45h of it."""
    return max((time.time() - wall0) - (time.monotonic() - mono0), 0.0)


class ChatFailed(RuntimeError):
    """chat() gave up. Carries the per-attempt timing log so a failed call's
    time is still attributable (throttling vs provider error vs timeout)."""

    def __init__(self, message: str, attempts_log: list[dict], throttles: int):
        super().__init__(message)
        self.attempts_log = attempts_log
        self.throttles = throttles


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


def _retry_after_seconds(err: Exception) -> float | None:
    """The server's suggested wait from a 429/503 Retry-After header, in
    seconds, or None. OpenRouter's docs say to honor it -- the upstream
    provider is telling us exactly how long it's saturated for."""
    hdrs = getattr(err, "headers", None)
    if not hdrs:
        return None
    raw = hdrs.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)                    # delta-seconds form
    except (TypeError, ValueError):
        return None                          # HTTP-date form: ignore, back off


def _backoff_seconds(attempt: int, retry_after: float | None) -> float:
    """How long to sleep before the next attempt: the server's Retry-After
    if it gave one, else exponential (4, 8, 16, 32, ...), capped."""
    if retry_after is not None:
        return min(retry_after, CHAT_BACKOFF_MAX_S)
    return min(CHAT_BACKOFF_BASE_S * (2 ** attempt), CHAT_BACKOFF_MAX_S)


def _sleep_before_retry(attempt: int, retry_after: float | None,
                        deadline: float | None) -> float:
    """Back off before the next attempt, but never sleep past the run
    deadline -- a backoff that would overrun it is pointless (the next
    read couldn't start anyway) and would blow the wall-clock budget."""
    wait = _backoff_seconds(attempt, retry_after)
    if deadline is not None:
        wait = min(wait, max(0.0, deadline - time.monotonic()))
    if wait > 0:
        time.sleep(wait)
    return round(max(wait, 0.0), 2)


def chat(model: str, messages: list, think: bool = False,
         deadline: float | None = None) -> dict:
    """One model round. Returns the normalized message:
    {"role", "content", "tool_calls": [arguments as dicts], "_raw"}.
    _raw is the wire-format original -- always resend THAT.

    Retries transient failures with EXPONENTIAL backoff (network, timeouts,
    429, 5xx, AND a 200 whose body is UNUSABLE -- no `choices`, or
    tool-call arguments that don't parse). On a 429 it honors OpenRouter's
    Retry-After header (their documented guidance: the provider is
    throttling, wait the stated time). `deadline` is an absolute
    time.monotonic() value: no read or backoff is allowed to run past it,
    so the WHOLE run stays bounded by AGENT_RUN_BUDGET_S even under heavy
    throttling -- a slow model gets its time, a hung one still can't hang.
    That unusable-200 class used to escape as a terminal error and get
    counted against the model; measured, it was a large share of the
    'backend_error' noise. A 400 gets one retry without the reasoning
    block. Other 4xx are our bug -- fail fast."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("set OPENROUTER_API_KEY in env or .env")
    body = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "temperature": 0,
        "max_tokens": PLAN_MAX_TOKENS if think else ACT_MAX_TOKENS,
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
    req = urllib.request.Request(OPENROUTER_URL, json.dumps(body).encode(), headers)
    last_error: Exception | None = None
    started = time.monotonic()
    throttles = 0   # how many 429s we rode out; surfaced for flakiness stats
    # one entry per HTTP attempt: where the wall-clock went. status is
    # ok | http_<code> | timeout | network:<Exc> | unusable:<Exc>
    attempts_log: list[dict] = []
    truncated = False      # a 200 cut off at max_tokens: retrying can't help

    def finish(a: dict, t0: float, http: Any = None) -> None:
        a["total_s"] = round(time.monotonic() - t0, 2)
        slept = _host_sleep_s(a.pop("_wall0"), t0)
        if slept > 1:
            a["host_sleep_s"] = round(slept, 1)   # our machine, not the model
        hdrs = getattr(http, "headers", None)
        if hdrs is not None and not a.get("generation_id"):
            a["generation_id"] = hdrs.get("X-Generation-Id")
        attempts_log.append(a)
        _log_call({"event": "attempt_end", "model": model, "think": think, **a})

    for attempt in range(CHAT_ATTEMPTS):
        # never start a read we can't finish before the run deadline
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 1:
            last_error = last_error or RuntimeError(
                f"run budget exhausted before attempt {attempt + 1}")
            break
        read_timeout = CHAT_TIMEOUT_S if remaining is None else min(
            CHAT_TIMEOUT_S, remaining)
        a: dict = {"attempt": attempt + 1,
                   "offset_s": round(time.monotonic() - started, 2),
                   "_wall0": time.time()}
        _log_call({"event": "attempt_start", "model": model, "think": think,
                   "attempt": attempt + 1})
        t0 = time.monotonic()
        http = None
        try:
            http = urllib.request.urlopen(req, timeout=read_timeout)
            # headers arrive fast; the body (after keep-alive whitespace)
            # arrives when the model is done. Splitting the two separates
            # OpenRouter's front door from generation.
            a["headers_s"] = round(time.monotonic() - t0, 2)
            payload = http.read()
            a["bytes"] = len(payload)
            resp = json.loads(payload)
            # recorded BEFORE parsing tool calls, so an unusable reply still
            # says why it ended: "length" = cut off at our max_tokens
            a["generation_id"] = resp.get("id")
            choice = (resp.get("choices") or [{}])[0]
            a["finish_reason"] = choice.get("finish_reason")
            a["native_finish_reason"] = choice.get("native_finish_reason")
            a["completion_tokens"] = (resp.get("usage") or {}).get("completion_tokens")
            a["provider"] = resp.get("provider")
            _msg = choice.get("message") or {}
            _args = [str((tc.get("function") or {}).get("arguments") or "")
                     for tc in (_msg.get("tool_calls") or [])]
            a["tool_calls_n"] = len(_args)
            a["args_chars"] = sum(len(x) for x in _args)
            # where the completion tokens went: measured 2026-09-17, ling's
            # think-off replies ran 1000-1770 tokens for <1900 chars of
            # arguments, so most output was prose or reasoning
            a["content_chars"] = len(str(_msg.get("content") or ""))
            a["reasoning_chars"] = len(str(_msg.get("reasoning") or ""))
            # PARSE inside the retry: a 200 whose body has no choices or
            # whose tool-call arguments don't parse is a transient
            # provider glitch, not the model's answer -- so a failure
            # here should retry, not become a terminal error.
            raw = resp["choices"][0]["message"]
            tool_calls = [
                {
                    "id": tc.get("id"),
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": json.loads(tc["function"]["arguments"]),
                    },
                }
                for tc in (raw.get("tool_calls") or [])
            ]
            norm = dict(raw)  # shallow copy; passenger fields ride through
            norm["tool_calls"] = tool_calls
            norm["_raw"] = raw
            norm["_usage"] = resp.get("usage")  # {prompt_tokens, completion_tokens, cost}
            a["status"] = "ok"
            if _hit_cap(a):
                # parsed, but cut off: tool calls after the cut are silently
                # missing. Kept as ok (the model can recover) and flagged.
                a["cap_hit"] = True
            finish(a, t0, http)
            # post-hoc call stats: latency, how many attempts it took, and
            # the upstream provider OpenRouter actually routed to (when it
            # reports it). This is how provider flakiness gets separated
            # from model competence downstream.
            norm["_meta"] = {
                "latency_s": round(time.monotonic() - started, 2),
                "attempts": attempt + 1,
                "throttles": throttles,
                "provider": resp.get("provider"),
                "generation_id": resp.get("id"),
                "attempts_log": attempts_log,
            }
            return norm
        except urllib.error.HTTPError as e:
            a["status"] = f"http_{e.code}"
            try:   # WHY it was refused; 400s were undiagnosable without this
                a["error_body"] = e.read()[:400].decode("utf-8", "replace")
            except Exception:  # noqa: BLE001 -- diagnostics only
                pass
            finish(a, t0, e)
            retry_after = None
            if e.code == 429:
                throttles += 1
                retry_after = _retry_after_seconds(e)  # honor the server
                a["retry_after"] = retry_after
                last_error = e
            elif e.code >= 500:
                last_error = e
            elif e.code == 400 and body.pop("reasoning", None) is not None:
                # some models reject the reasoning block outright;
                # drop it and retry once without
                req = urllib.request.Request(
                    OPENROUTER_URL, json.dumps(body).encode(), headers
                )
                last_error = e
            else:
                # not transient: fail fast, but keep the log and the reason
                raise ChatFailed(
                    f"request refused: HTTP {e.code}: {a.get('error_body', '')[:200]}",
                    attempts_log, throttles) from e
            a["backoff_s"] = _sleep_before_retry(attempt, retry_after, deadline)
            continue
        except (urllib.error.URLError, TimeoutError, OSError,
                http_client.HTTPException) as e:
            # HTTPException covers IncompleteRead: the connection dropped
            # mid-body. It is NOT an OSError, so until 2026-09-17 it escaped
            # this retry loop and ended the run on the first drop.
            timed_out = isinstance(e, TimeoutError) or "timed out" in str(e)
            a["status"] = "timeout" if timed_out else f"network:{type(e).__name__}"
            finish(a, t0, http)
            last_error = e
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            # 200 but the body is unusable: missing choices, or tool-call
            # arguments that were truncated / malformed. Retry like a 5xx.
            a["status"] = f"unusable:{type(e).__name__}"
            # what the broken reply looked like: a cut-off batch and a
            # runaway repetition loop read very differently at the tail
            if a.get("tool_calls_n"):
                joined = " | ".join(_args)
                a["args_head"] = joined[:300]
                a["args_tail"] = joined[-300:]
            finish(a, t0, http)
            last_error = e
            if _hit_cap(a):
                # same request, same cap, temperature 0: a retry would be
                # cut off again. Stop and say so.
                truncated = True
                break
        a["backoff_s"] = _sleep_before_retry(attempt, None, deadline)
    if truncated:
        raise ChatFailed(
            f"reply truncated at max_tokens={body['max_tokens']} "
            f"({round(time.monotonic() - started)}s): "
            f"{type(last_error).__name__}: {last_error}",
            attempts_log, throttles,
        )
    raise ChatFailed(
        f"model backend unusable after {CHAT_ATTEMPTS} attempts "
        f"({round(time.monotonic() - started)}s, {throttles} throttled): "
        f"{type(last_error).__name__}: {last_error}",
        attempts_log, throttles,
    )


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


def scrub_optionals(value):
    """Drop the values a small model uses to mean 'leave this unset',
    which the schema would otherwise bounce into a livelock, at every
    depth:
      * null anywhere -- an optional the model nulled instead of
        omitting (max_relief_m: null); our tools treat MISSING as the
        default, so null == absent. Measured: two gpt-oss-120b runs died
        on 'None is not of type number'.
      * max_relief_m <= 0 -- a dog with NO hill limit. The field's
        minimum is 1 (0 metres of tolerance is meaningless), but models
        write 0 for 'not applicable'; treat it as absent. Measured:
        qwen3-8b livelocked on 'max_relief_m: 0 is <= 0', lakeview 0/5.
    Matches how the tools already read their inputs, and runs before both
    the referee and dispatch (and inside _call_args, for the auditors)."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if v is None:
                continue
            if k == "max_relief_m" and isinstance(v, (int, float)) and v <= 0:
                continue
            out[k] = scrub_optionals(v)
        return out
    if isinstance(value, list):
        return [scrub_optionals(v) for v in value]
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
    return scrub_optionals(parsed)


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


def audit_daylight(messages: list) -> str | None:
    """The daylight oracle: no dog is walked after dark. Any roster with
    an explicit AFTERNOON dog (an 'any' dog is scheduled early, so it
    can't run late) must have called check_daylight and passed sunset_min
    to optimize_route -- otherwise the route can't know when darkness
    falls and could schedule past it. Returns one prescriptive message,
    or None."""
    route_call, called_daylight = None, False
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            name = tc["function"]["name"]
            if name == "optimize_route":
                route_call = _call_args(tc)
            elif name == "check_daylight":
                called_daylight = True
    if route_call is None:
        return None  # no route yet; other auditors handle that
    afternoon = [s for s in route_call["stops"]
                 if s.get("walk_window") == "afternoon"]
    if not afternoon:
        return None  # nothing can run late
    if not called_daylight or route_call.get("sunset_min") is None:
        names = ", ".join(s["name"] for s in afternoon)
        return (
            f"REJECTED: {names} may be walked in the afternoon and no dog "
            "is walked after dark. Call check_daylight for the walking area "
            "and date, then call optimize_route again passing sunset_min so "
            "every walk finishes before sunset."
        )
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


def audit_rain(messages: list, plan: dict) -> str | None:
    """The rain oracle: a skip_rain dog won't walk when it's wet. If such
    a dog's walk window is raining (>= RAIN_TRIGGER_MM, from its weather
    check), its submit_plan entry must be a short minimal visit, not a
    full walk. Reads precip from the weather RESULT correlated to its
    call by tool_call_id. Returns a prescriptive message, or None."""
    from dog_walker.toolbox import MINIMAL_VISIT_MIN, RAIN_TRIGGER_MM

    results_by_id: dict = {}
    for msg in messages:
        if msg.get("role") == "tool":
            try:
                results_by_id[msg.get("tool_call_id")] = json.loads(
                    msg.get("content") or "{}")
            except (ValueError, TypeError):
                pass
    route_call, weather = None, []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            name = tc["function"]["name"]
            if name == "optimize_route":
                route_call = _call_args(tc)
            elif name == "check_weather":
                weather.append((_call_args(tc), results_by_id.get(tc.get("id"))))
    if route_call is None:
        return None
    walks = {w.get("pet"): w for w in (plan.get("walks") or [])}

    gaps = []
    for stop in route_call["stops"]:
        if not stop.get("skip_rain"):
            continue
        lat, lon = stop["lat"], stop["lon"]
        precip = None
        for args, res in weather:
            if not isinstance(res, dict):
                continue
            if (abs(args.get("lat", 999) - lat) <= _NEAR_DEG
                    and abs(args.get("lon", 999) - lon) <= _NEAR_DEG):
                mp = (res.get("window") or {}).get("max_precip_mm")
                if mp is not None:
                    precip = mp if precip is None else max(precip, mp)
        if precip is None or precip < RAIN_TRIGGER_MM:
            continue  # dry (or not yet checked -- the weather oracle covers that)
        w = walks.get(stop["name"])
        if not w or "walk_start" not in w or "walk_end" not in w:
            continue  # missing walk -- feasibility/coverage auditors handle it
        try:
            dur = (_clock_to_hours(w["walk_end"])
                   - _clock_to_hours(w["walk_start"])) * 60
        except (ValueError, KeyError, AttributeError):
            continue
        if dur > MINIMAL_VISIT_MIN + 5:  # a little slack over the minimal visit
            gaps.append(
                f"{stop['name']} won't walk in the rain and it's wet: give a "
                f"minimal ~{MINIMAL_VISIT_MIN}-minute visit, not a "
                f"{round(dur)}-minute walk."
            )
    if gaps:
        return "REJECTED: " + " ".join(gaps)
    return None


def _tool_calls_and_results(messages: list) -> tuple:
    """(route_call, {tool: [(args, result)]}) -- every tool call paired
    with its result via tool_call_id. The synthesis auditors need results,
    not just call args."""
    results_by_id: dict = {}
    for msg in messages:
        if msg.get("role") == "tool":
            try:
                results_by_id[msg.get("tool_call_id")] = json.loads(
                    msg.get("content") or "{}")
            except (ValueError, TypeError):
                pass
    route_call, by_tool = None, {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            name = tc["function"]["name"]
            if name == "optimize_route":
                route_call = _call_args(tc)
            else:
                by_tool.setdefault(name, []).append(
                    (_call_args(tc), results_by_id.get(tc.get("id"))))
    return route_call, by_tool


def audit_priority(messages: list) -> str | None:
    """Priority is DERIVED, not given: the model must compute each dog's
    urgency (dog_priority) and pass it to optimize_route. Recompute it and
    veto any dog whose passed priority is missing or wrong."""
    from dog_walker.toolbox import dog_priority
    route_call, _ = _tool_calls_and_results(messages)
    if route_call is None:
        return None
    gaps = []
    for stop in route_call["stops"]:
        if int(stop.get("walk_minutes", 0) or 0) == 0:
            continue  # the depot has no dog
        want = dog_priority(stop)
        got = stop.get("priority")
        if got is None or int(got) != want:
            gaps.append(f"{stop['name']} should be {want}"
                        + (f", not {got}" if got is not None else " (missing)"))
    if gaps:
        return ("REJECTED: compute each dog's priority (+2 needs_meds, +2 "
                "comfort band <=30F wide, +1 difficulty>=3, +1 afternoon) and "
                "pass it in the optimize_route stops: " + "; ".join(gaps) + ".")
    return None


_SEVERITY = ["OK", "CAUTION", "SHORTEN", "DO_NOT_WALK"]
_TERRAIN_AS_WALK = {"OK": "OK", "CAUTION": "CAUTION", "AVOID": "DO_NOT_WALK"}


def audit_verdict(messages: list, plan: dict) -> str | None:
    """Each dog's plan verdict must be the WORST of its weather verdict and
    its terrain verdict (a hill AVOID blocks a walk as surely as a
    DO_NOT_WALK). The model must COMBINE the two tool outputs; recompute
    the worst and veto any dog whose submitted verdict disagrees."""
    route_call, by_tool = _tool_calls_and_results(messages)
    if route_call is None:
        return None
    weather = by_tool.get("check_weather", [])
    terrain = by_tool.get("check_terrain", [])
    walks = {w.get("pet"): w for w in (plan.get("walks") or [])}

    def verdict_near(calls, lat, lon):
        for args, res in calls:
            if not isinstance(res, dict):
                continue
            if (abs(args.get("lat", 999) - lat) <= _NEAR_DEG
                    and abs(args.get("lon", 999) - lon) <= _NEAR_DEG):
                return res.get("verdict")
        return None

    gaps = []
    for stop in route_call["stops"]:
        if int(stop.get("walk_minutes", 0) or 0) == 0:
            continue
        wv = verdict_near(weather, stop["lat"], stop["lon"])
        if wv is None or wv not in _SEVERITY:
            continue  # no weather yet -- the weather auditor handles that
        sev = _SEVERITY.index(wv)
        tv = verdict_near(terrain, stop["lat"], stop["lon"])
        if tv in _TERRAIN_AS_WALK:
            sev = max(sev, _SEVERITY.index(_TERRAIN_AS_WALK[tv]))
        want = _SEVERITY[sev]
        w = walks.get(stop["name"])
        if not w or "verdict" not in w:
            continue
        if w["verdict"] != want:
            gaps.append(
                f"{stop['name']} verdict should be {want} (worst of weather "
                f"{wv}" + (f" and terrain {tv}" if tv else "") + f"), not {w['verdict']}")
    if gaps:
        return "REJECTED: " + "; ".join(gaps) + "."
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
    retries = 0            # extra chat attempts beyond the first = provider friction
    throttles = 0          # 429s ridden out with backoff = throttling, specifically
    providers: set = set()  # upstream providers OpenRouter routed to
    # diagnostics: one entry per chat() call and per tool dispatch, in order,
    # so a run's wall-clock can be split into model time, throttling,
    # provider failures, and our own tools
    timing: list[dict] = []
    deadline = time.monotonic() + AGENT_RUN_BUDGET_S  # bounds every chat() retry

    def absorb(reply: dict, phase: str, round_no: int) -> None:
        nonlocal retries, throttles
        _accumulate_usage(usage, reply.get("_usage"))
        meta = reply.get("_meta") or {}
        retries += meta.get("attempts", 1) - 1
        throttles += meta.get("throttles", 0)
        if meta.get("provider"):
            providers.add(meta["provider"])
        timing.append({"kind": "chat", "phase": phase, "round": round_no,
                       "wall_s": meta.get("latency_s"),
                       "generation_id": meta.get("generation_id"),
                       "provider": meta.get("provider"),
                       "usage": reply.get("_usage"),
                       "attempts": meta.get("attempts_log", [])})

    def extras() -> dict:
        return {"usage": dict(usage), "rounds": last_round,
                "retries": retries, "throttles": throttles,
                "providers": sorted(providers), "timing": timing}
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
                "stops. For EACH dog you must also COMPUTE a priority and "
                "pass it in the optimize_route stop: +2 needs_meds, +2 if "
                "the comfort band is <=30F wide, +1 difficulty>=3, +1 "
                "afternoon window (0-6). If any dog has an afternoon window, "
                "call check_daylight for the area and pass sunset_min to "
                "optimize_route -- no dog is walked after dark. Each dog's "
                "submit_plan verdict is the WORST of its weather verdict and "
                "its terrain verdict (a terrain AVOID counts as "
                "DO_NOT_WALK). A dog with "
                "skip_rain that finds rain in its window (raining=true) "
                "gets a short ~10-minute minimal visit, not its full walk. "
                "If "
                "optimize_route returns feasible=false, the windows can't "
                "all be arranged before dark: "
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
        plan_reply = chat(model, messages, think=True, deadline=deadline)
        absorb(plan_reply, "plan", 0)
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
                arguments = scrub_optionals(call["function"]["arguments"])
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
                    # Daylight first: if the route didn't account for
                    # sunset, its feasibility can't be trusted yet, so
                    # force the sunset-aware route before judging anything.
                    if gap := audit_daylight(messages):
                        yield {"event": "audit_veto", "auditor": "daylight", "gap": gap}
                        messages.append(tool_feedback(call, {"error": gap}))
                        continue
                    if gap := audit_priority(messages):
                        yield {"event": "audit_veto", "auditor": "priority", "gap": gap}
                        messages.append(tool_feedback(call, {"error": gap}))
                        continue
                    # Feasibility next: it settles whether the plan even
                    # claims the walks happen. An honestly-infeasible plan
                    # has no timeline to weather-check, so it stops here.
                    if gap := audit_feasibility(messages, arguments):
                        yield {"event": "audit_veto", "auditor": "feasibility", "gap": gap}
                        messages.append(tool_feedback(call, {"error": gap}))
                        continue
                    if arguments.get("feasible") is False:
                        yield {"event": "final", "plan": arguments, **extras()}
                        return
                    if gap := audit_weather_coverage(messages):
                        yield {"event": "audit_veto", "auditor": "weather", "gap": gap}
                        messages.append(tool_feedback(call, {"error": gap}))
                        continue
                    if gap := audit_verdict(messages, arguments):
                        yield {"event": "audit_veto", "auditor": "verdict", "gap": gap}
                        messages.append(tool_feedback(call, {"error": gap}))
                        continue
                    if gap := audit_rain(messages, arguments):
                        yield {"event": "audit_veto", "auditor": "rain", "gap": gap}
                        messages.append(tool_feedback(call, {"error": gap}))
                        continue
                    if gap := audit_terrain_coverage(messages):
                        yield {"event": "audit_veto", "auditor": "terrain", "gap": gap}
                        messages.append(tool_feedback(call, {"error": gap}))
                        continue
                    yield {"event": "final", "plan": arguments, **extras()}
                    return

                _log_call({"event": "tool_start", "tool": name})
                t_tool = time.monotonic()
                result = dispatch(name, arguments)
                tool_s = round(time.monotonic() - t_tool, 2)
                timing.append({"kind": "tool", "name": name, "round": round_no,
                               "seconds": tool_s,
                               "error": isinstance(result, dict) and "error" in result})
                _log_call({"event": "tool_end", "tool": name, "seconds": tool_s})
                yield {"event": "result", "name": name, "result": result}
                messages.append(tool_feedback(call, result))

            # ---- next model round (think off: plan is already in state)
            reply = chat(model, messages, think=False, deadline=deadline)
            absorb(reply, "act", round_no)
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
            **extras(),
        }
    except Exception as e:  # noqa: BLE001 -- stream boundary: fail as an event
        if isinstance(e, ChatFailed):
            # the call that killed the run still spent real time: keep it
            throttles += e.throttles
            retries += max(len(e.attempts_log) - 1, 0)
            timing.append({"kind": "chat", "phase": "failed", "round": last_round,
                           "wall_s": round(sum(a.get("total_s", 0) + a.get("backoff_s", 0)
                                               for a in e.attempts_log), 2),
                           "generation_id": None, "provider": None, "usage": None,
                           "attempts": e.attempts_log})
        yield {"event": "error", "message": f"{type(e).__name__}: {e}", **extras()}


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
