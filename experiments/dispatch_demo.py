"""The model drives the REAL toolbox in a raw dispatch loop.

toolcall_demo.py scripts fake tool results. Here the loop is honest:
    model -> tool_calls -> we EXECUTE from REGISTRY -> results back
    -> model again ... until it answers in prose.
That cycle is the skeleton of the Phase-3 agent; what's deliberately
missing (and Phase 3 adds): argument validation before dispatch, a
plan step, reflection on results, and retry discipline. Watch for
where this loop misbehaves -- its failures are the Phase-3 syllabus.

Usage:
    uv run python experiments/dispatch_demo.py            # OpenRouter
    uv run python experiments/dispatch_demo.py --think
    LOCAL_LLM=fossil uv run python experiments/dispatch_demo.py

Same backend selection and dialect notes as toolcall_demo.py.
Live network: geocoding is 1.1s/address (Nominatim politeness);
routing uses the OpenRouteService key from .env.
"""

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from dog_walker.toolbox import REGISTRY

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

# the real schemas, straight from the registry -- no hand-copies
TOOLS = [schema for _fn, schema in REGISTRY.values()]

MAX_ROUNDS = 6  # a leash: loops that can call tools can also spin


def chat(backend: dict, messages: list, think: bool) -> dict:
    """One chat round; normalized (see toolcall_demo.py docstring)."""
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
            sys.exit(f"set {backend['key_env']} in your env or .env first")
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
    """Execute one tool call against the registry.

    An unknown tool or a blown call becomes an error RESULT, not an
    exception -- the model gets to see what went wrong and try again.
    (Phase 3 formalizes this into validate-then-bounce.)
    """
    if name not in REGISTRY:
        return {"error": f"unknown tool {name!r}"}
    fn = REGISTRY[name][0]
    try:
        return fn(**arguments)
    except Exception as e:  # noqa: BLE001 -- boundary: everything becomes data
        return {"error": f"{type(e).__name__}: {e}"}


def tool_feedback(backend: dict, call: dict, result: dict) -> dict:
    """Round-trip one result in the backend's dialect."""
    msg = {"role": "tool", "content": json.dumps(result)}
    if backend["style"] == "openai":
        msg["tool_call_id"] = call["id"]
    return msg


def brief(value, limit=140) -> str:
    """One-line summary for the trace printout."""
    text = json.dumps(value) if not isinstance(value, str) else value
    return text[:limit] + ("..." if len(text) > limit else "")


####################
ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", help="override the backend's default model")
ap.add_argument("--think", action="store_true", help="enable reasoning mode")
args = ap.parse_args()
backend = BACKENDS[os.environ.get("LOCAL_LLM") or "openrouter"]
if args.model:
    backend = {**backend, "model": args.model}
print(f"--- backend: {backend['url']}  model: {backend['model']}  think: {args.think}")

today = datetime.now().astimezone()
msgs = [
    {
        "role": "system",
        "content": (
            f"You are a dog-walking planner. Today is {today:%A} "
            f"{today:%Y-%m-%d}. Call tools directly, without asking "
            "permission. Batch all addresses into ONE geocode call. "
            "Dates are ISO YYYY-MM-DD. Be concise."
        ),
    },
    {
        "role": "user",
        "content": (
            "Plan this afternoon's walks starting and ending at Wrigley "
            "Field, Chicago. Daisy is at Lincoln Park Zoo, Chicago and "
            "gets a 20 minute walk. Rex is at 5218 N Clark St, Chicago "
            "and gets 60 minutes. I leave at 13:00. Is the weather OK "
            "for them?"
        ),
    },
]
####################

for round_no in range(1, MAX_ROUNDS + 1):
    reply = chat(backend, msgs, args.think)
    calls = reply.get("tool_calls") or []
    if not calls:
        print(f"\n=== FINAL (round {round_no}) ===\n{reply.get('content')}")
        break
    msgs.append(reply["_raw"])
    for call in calls:
        name = call["function"]["name"]
        arguments = call["function"]["arguments"]
        print(f"\n[round {round_no}] {name}({brief(arguments)})")
        result = dispatch(name, arguments)
        print(f"          -> {brief(result)}")
        msgs.append(tool_feedback(backend, call, result))
else:
    print(f"\n!! hit MAX_ROUNDS={MAX_ROUNDS} without a final answer")
