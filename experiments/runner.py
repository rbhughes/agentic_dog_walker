"""Phase-1 bake-off runner: fixtures x models -> scorecard.

Usage:  uv run python experiments/runner.py [model ...]
        (default: qwen2.5:7b qwen3:8b qwen2.5:3b)

What it enforces -- each check exists because raw tool-calling
experiments showed nobody else enforces it:
  * calls_tool        the model called the right tool (or, for None,
                      resisted calling one at all)
  * args_valid        arguments validate against the tool's OWN
                      parameters schema, via jsonschema -- this is the
                      enforcement layer Ollama does not provide
  * args_include      specific argument values (e.g. the resolved date)
  * prose_mentions / prose_mentions_any   fidelity of the final prose
                      to the scripted tool result

Raw transcripts land in experiments/results/<model>.json so any FAIL
can be autopsied instead of argued about.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import jsonschema

from fixtures import FIXTURES  # runner and fixtures share a directory

OLLAMA = "http://fossil:11434/api/chat"
DEFAULT_MODELS = ["qwen2.5:7b", "qwen3:8b", "qwen2.5:3b"]
RESULTS_DIR = Path(__file__).parent / "results"


def chat(model: str, messages: list, tools: list) -> tuple[dict, float]:
    """One /api/chat round. Returns (message, seconds)."""
    body = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
        # num_thread 10: measured sweet spot on fossil (12 is worse).
        # num_predict caps runaway generations at ~5 tok/s.
        "options": {"num_thread": 10, "temperature": 0, "num_predict": 300},
    }
    if model.startswith("qwen3"):
        # qwen3 "thinks" by default; that's a separate experiment. Keep
        # the bake-off apples-to-apples on plain tool calling.
        body["think"] = False
    req = urllib.request.Request(
        OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.monotonic()
    reply = json.load(urllib.request.urlopen(req, timeout=600))["message"]
    return reply, time.monotonic() - t0


def tool_schema(tools: list, name: str) -> dict:
    for t in tools:
        if t["function"]["name"] == name:
            return t["function"]["parameters"]
    raise KeyError(name)


def run_fixture(model: str, fx: dict) -> dict:
    """Run one fixture; return {id, passed, checks: [(name, ok, note)], ...}."""
    checks: list[tuple[str, bool, str]] = []
    transcript: dict = {"fixture": fx["id"]}

    reply, secs = chat(model, fx["messages"], fx["tools"])
    transcript["first_reply"] = reply
    calls = reply.get("tool_calls") or []
    want = fx["expect"]["calls_tool"]

    if want is None:
        ok = not calls
        note = "" if ok else f"called {calls[0]['function']['name']!r} anyway"
        checks.append(("no-call", ok, note))
    elif not calls:
        checks.append(("calls_tool", False, "no tool call at all"))
    else:
        got = calls[0]["function"]["name"]
        checks.append(("calls_tool", got == want, f"got {got!r}"))
        args = calls[0]["function"]["arguments"]

        if fx["expect"].get("args_valid") and got == want:
            try:
                jsonschema.validate(args, tool_schema(fx["tools"], want))
                checks.append(("args_valid", True, ""))
            except jsonschema.ValidationError as e:
                checks.append(("args_valid", False, e.message[:60]))

        for key, val in fx["expect"].get("args_include", {}).items():
            checks.append(
                (f"arg:{key}", args.get(key) == val,
                 f"got {args.get(key)!r}, want {val!r}")
            )

    # optional round 2: feed the scripted result back, grade the prose
    if "tool_result" in fx and calls:
        msgs = fx["messages"] + [
            reply,
            {"role": "tool", "content": json.dumps(fx["tool_result"])},
        ]
        final, secs2 = chat(model, msgs, fx["tools"])
        transcript["final_reply"] = final
        secs += secs2
        prose = (final.get("content") or "").lower()

        fe = fx.get("final_expect", {})
        for needle in fe.get("prose_mentions", []):
            checks.append(
                (f"mentions:{needle}", needle.lower() in prose, ""))
        if any_of := fe.get("prose_mentions_any"):
            hit = next((n for n in any_of if n.lower() in prose), None)
            checks.append(
                ("mentions-any", hit is not None,
                 f"matched {hit!r}" if hit else "no refusal phrasing found"))

    transcript["checks"] = [
        {"check": c, "ok": ok, "note": n} for c, ok, n in checks]
    return {
        "id": fx["id"],
        "passed": all(ok for _, ok, _ in checks),
        "checks": checks,
        "secs": secs,
        "transcript": transcript,
    }


def main() -> None:
    models = sys.argv[1:] or DEFAULT_MODELS
    RESULTS_DIR.mkdir(exist_ok=True)
    summary: dict[str, list] = {}

    for model in models:
        print(f"\n=== {model} " + "=" * (46 - len(model)))
        results = [run_fixture(model, fx) for fx in FIXTURES]
        summary[model] = results
        for r in results:
            mark = "PASS" if r["passed"] else "FAIL"
            print(f"  {r['id']:16s} {mark}  {r['secs']:5.1f}s")
            for name, ok, note in r["checks"]:
                if not ok:
                    print(f"      x {name}: {note}")
        (RESULTS_DIR / f"{model.replace(':', '_')}.json").write_text(
            json.dumps([r["transcript"] for r in results], indent=2))

    print("\n=== scorecard " + "=" * 40)
    print(f"{'fixture':16s}" + "".join(f"{m:>14s}" for m in models))
    for i, fx in enumerate(FIXTURES):
        row = f"{fx['id']:16s}"
        for m in models:
            row += f"{'PASS' if summary[m][i]['passed'] else 'FAIL':>14s}"
        print(row)
    row = f"{'total time':16s}"
    for m in models:
        row += f"{sum(r['secs'] for r in summary[m]):>13.0f}s"
    print(row)


if __name__ == "__main__":
    main()
