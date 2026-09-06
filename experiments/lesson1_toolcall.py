"""Lesson 1 extended: one tool-call cycle against a chosen backend.

Usage:
    uv run python experiments/lesson1_toolcall.py                    # default
    uv run python experiments/lesson1_toolcall.py --think            # reasoning on
    uv run python experiments/lesson1_toolcall.py anthropic/claude-haiku-4.5

Backend selection: if LOCAL_LLM is set (env or .env, e.g.
LOCAL_LLM=fossil), that backend is used; otherwise OpenRouter.
OpenRouter needs OPENROUTER_API_KEY in the environment or in .env
(never committed; .env is gitignored).

The ecosystem speaks two wire dialects for the same idea:
  * ollama  (/api/chat):        message.tool_calls[].function.arguments is a DICT;
                                tool feedback is {"role": "tool", "content": ...}
  * openai  (/chat/completions): reply hides in choices[0].message; arguments is a
                                JSON STRING; tool feedback must carry tool_call_id
chat() normalizes both to the ollama-ish shape the rest of our code thinks in.
"""

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

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_weather",
            "description": "Get the weather forecast for a city on a date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "urban area"},
                    # JSON Schema has no "date" type -- dates are strings
                    # (optionally format: date); the model reads both hints
                    "date": {
                        "type": "string",
                        "format": "date",
                        "description": "ISO date YYYY-MM-DD",
                    },
                },
                "required": ["city", "date"],
            },
        },
    }
]


def chat(backend: dict, messages: list, think: bool = False) -> dict:
    """One chat round. Returns a normalized message:
    {"role", "content", "tool_calls": [...arguments as dict...], "_raw": ...}
    _raw is what round 2 must append -- always resend the wire-format
    original, never the normalized copy.

    `think` toggles reasoning mode symmetrically: Ollama's `think`
    field vs OpenRouter's unified `reasoning` block -- same knob,
    two dialects (like everything else on this wire)."""
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
    else:  # openai dialect
        key = os.environ.get(backend["key_env"], "")
        if not key:
            sys.exit(f"set {backend['key_env']} in your env or .env first")
        body = {
            "model": backend["model"],
            "messages": messages,
            "tools": TOOLS,
            "temperature": 0,
            "max_tokens": 2000 if think else 400,  # room for the trace
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
        # arguments arrive as a JSON string in the openai dialect
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


def tool_feedback(backend: dict, reply: dict, result: dict) -> dict:
    """Build the round-2 tool message in the backend's dialect."""
    msg = {"role": "tool", "content": json.dumps(result)}
    if backend["style"] == "openai":
        # openai dialect ties the result to the specific call it answers
        msg["tool_call_id"] = reply["tool_calls"][0]["id"]
    return msg


####################
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", help="override the backend's default model")
ap.add_argument("--think", action="store_true", help="enable reasoning mode")
args = ap.parse_args()

# LOCAL_LLM=fossil (env or .env) selects the local backend; the
# default is rented -- fossil's 3-minute thinking runs made the call.
backend = BACKENDS[os.environ.get("LOCAL_LLM") or "openrouter"]
if args.model:
    backend = {**backend, "model": args.model}
print(f"--- backend: {backend['url']}  model: {backend['model']}  think: {args.think}")

blah = "Should I walk Rex in Calgary tomorrow around 4pm?"
today = datetime.now().astimezone()

msgs = [
    {
        "role": "system",
        "content": f"Today is {today:%A} {today.isoformat()}. Pass dates as ISO YYYY-MM-DD",
    },
    {"role": "user", "content": blah},
]
####################

reply = chat(backend, msgs, think=args.think)
print(
    "FIRST REPLY:",
    json.dumps({k: v for k, v in reply.items() if k != "_raw"}, indent=2),
)

if reply.get("tool_calls"):
    # Round 2 -- feed a fake result back and watch it become prose.
    # Resend the RAW assistant message: the wire format the backend expects.
    msgs.append(reply["_raw"])
    msgs.append(
        tool_feedback(backend, reply, {"temp_c": -21, "precip_mm": 40, "wind_kph": 30})
    )
    final = chat(backend, msgs, think=args.think)
    print(
        "FINAL:", json.dumps({k: v for k, v in final.items() if k != "_raw"}, indent=2)
    )
