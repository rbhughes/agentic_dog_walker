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
# TODO(Bryan) 1: the referee
# ---------------------------------------------------------------------


def validate_call(name: str, arguments: dict) -> str | None:
    """Return None if the call is legal, else a short error string the
    model can act on.

    Checks, in order:
      * name exists (REGISTRY or "submit_plan") -- unknown tool is an
        error string, not a KeyError
      * arguments validate against that tool's parameters schema
        (jsonschema.validate; catch jsonschema.ValidationError and
        return e.message -- it's already human/model-readable)

    Hint: the parameters schema lives at
    schema["function"]["parameters"] for each registered schema, and
    SUBMIT_PLAN_SCHEMA needs the same treatment.
    """
    raise NotImplementedError("TODO(Bryan)")


# ---------------------------------------------------------------------
# TODO(Bryan) 2: the auditor (write AFTER the loop works end-to-end)
# ---------------------------------------------------------------------


def audit_weather_coverage(messages: list) -> str | None:
    """Deterministic reflection. Read the conversation; return None if
    every dog's walk interval has a covering weather check, else ONE
    corrective sentence to inject (handle one gap per audit -- the
    loop comes back around).

    Sketch:
      * find the latest optimize_route RESULT in the tool messages
        (json.loads each; the one with a "timeline") -- no timeline
        yet means nothing to audit -> None
      * find every check_weather CALL made so far (assistant messages'
        tool_calls) with its lat/lon/start_hour/end_hour
      * for each timeline entry with walk_minutes: does some weather
        call cover it? "Covers" = hour window contains the walk
        interval; being strict about location needs the stop coords,
        which the route call's own arguments carry
      * first uncovered dog -> "GAP: <dog> walks <start>-<end> at
        (<lat>, <lon>) but no check_weather covers that interval.
        Check it before submitting."
    """
    raise NotImplementedError("TODO(Bryan)")


# ---------------------------------------------------------------------
# TODO(Bryan) 3: the loop
# ---------------------------------------------------------------------


def run(request: str, backend_name: str | None = None) -> dict:
    """The agent. Returns the validated submit_plan arguments.

    Shape to build (pseudo-code, not gospel -- deviate with reasons):

      backend = BACKENDS[backend_name or LOCAL_LLM or "openrouter"]
      messages = [system, user(request)]

      # PLAN: one think=True round, no execution expected; append the
      # reply (its _raw) so the plan text is in the state. If it tried
      # to call tools already, that's fine -- fall through to act.

      # ACT: for round in range(MAX_ROUNDS):
      #   reply = chat(messages, think=False); append _raw
      #   no tool_calls? -> nudge: inject "use submit_plan to finish"
      #     (a model that answers in prose hasn't finished)
      #   for each call:
      #     error = validate_call(...)
      #     if error: append tool_feedback(..., {"error": error}); continue
      #     if name == "submit_plan":
      #       gap = audit_weather_coverage(messages)
      #       if gap: bounce it ({"error": gap}) and keep looping
      #       else: return the arguments -- DONE
      #     append tool_feedback(..., dispatch(...))

      # falling out of the loop = failure; raise with the last state

    System prompt content that matters (from lessons 1-2): today's
    date + weekday; call tools directly without asking permission;
    batch geocoding; finish via submit_plan; be concise.
    """
    raise NotImplementedError("TODO(Bryan)")


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
