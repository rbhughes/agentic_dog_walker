"""The measurement harness: how often, and how expensively, can a
model actually finish this task?

The qualifier (qualify.py) is a GATE -- one run, pass or fail. This is
the MEASUREMENT it was always trying to become. It runs every model
across the scenario library (scenarios.py) k times each and reports:

  * a PASS RATE with a Wilson score interval, not a single-run verdict
    (5/5 and 50/50 are different confidences; the interval says so);
  * real COST and LATENCY per completed plan, summed from the
    OpenRouter usage the agent now surfaces (dollars, not token guesses);
  * a FAILURE TAXONOMY built from the event stream -- veto livelock
    (couldn't satisfy an oracle), schema thrash (kept breaking tool
    schemas), prose stall (talked instead of calling), fabricated
    feasibility (claimed an impossible roster was doable), and so on --
    with the full transcript of every failure kept for the post-mortem.

The grade is deterministic: a run passes iff it reaches a validated
submit_plan whose `feasible` flag matches the scenario's known answer.
No model judges another; the same auditor the agent must satisfy IS
the measurement. That is the whole thesis -- you can measure an agent
honestly only when the task has a checkable finish line.

Run:
  uv run python -m dog_walker.measure                     # all models x all scenarios, k=5
  uv run python -m dog_walker.measure --k 10
  uv run python -m dog_walker.measure --models qwen/qwen3-8b
  uv run python -m dog_walker.measure --scenarios morning-overbook,full-house
  uv run python -m dog_walker.measure --dry              # show the plan + cost note, run nothing

Writes measurements/measure-<timestamp>.json (git-ignored): config,
every per-run record, per-(model,scenario) and per-model aggregates,
and the failure transcripts. Timestamped from day one so the same
model measured months apart -- drift -- becomes a finding.

Latency diagnostics (added 2026-09-16), also git-ignored:
  measurements/calls-<ts>.jsonl  live call log, written BEFORE and after every
                                 HTTP attempt and tool call: `tail -f` it, and a
                                 hung call is the last attempt_start with no end
  measurements/runs-<ts>.jsonl   each run record, appended as the run finishes,
                                 so a crash or kill never loses finished runs
Every record carries `timing`: the run's wall-clock split into buckets (see
time_breakdown) plus OpenRouter's own per-generation stats (time to first
token, generation time, reasoning tokens, hidden provider failovers).
  uv run python -m dog_walker.measure --no-gen-stats    # skip the stats lookups
  uv run python -m dog_walker.measure --timing-report [archive.json | runs.jsonl ...]
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dog_walker import toolbox
from dog_walker.scenarios import SCENARIOS, scenario_prompt, seed_geocode_cache

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_FILE = REPO_ROOT / "models.json"
ARCHIVE_DIR = REPO_ROOT / "measurements"
REPORT_DATA = REPO_ROOT / "site" / "src" / "data" / "measurement.json"

# display names + notes the walker.purr.io report renders (the only
# per-model prose the charts need; everything else derives from runs)
SHORT = {
    "inclusionai/ling-3.0-flash": "ling-3.0-flash",
    "anthropic/claude-haiku-4.5": "haiku-4.5",
    "qwen/qwen3.7-flash": "qwen3.7-flash",
    "openai/gpt-oss-120b": "gpt-oss-120b",
    "qwen/qwen3-8b": "qwen3-8b",
    "inception/mercury-2.5": "mercury-2.5",
}
NOTE = {"qwen/qwen3-8b": "prior default"}

DEFAULT_K = 5           # runs per (model, scenario)
# Outer wall-clock backstop, set ABOVE agent.AGENT_RUN_BUDGET_S (300) so the
# agent's own deadline-aware retry logic terminates a run cleanly FIRST
# (with a real error event + stats) and this only fires on a pathological
# loop the agent didn't catch. The 90s value here was the bug in the last
# sweep: it guillotined slow-but-working gpt-oss/qwen3-8b runs (whose
# successes ran to ~88s) and mislabeled them provider timeouts.
RUN_DEADLINE_S = 330
Z = 1.96                # 95% Wilson interval

# Outcomes that are the PROVIDER's fault, not the model's: a hung call the
# harness had to kill (timeout) or an endpoint that returned nothing usable
# after every retry (backend_error). These are NOTED but excluded from the
# pass-rate denominator -- competence and provider weather are different
# measurements, and mixing them is exactly what produced the qwen3.7
# artifact. (malformed_output stays IN: unparseable tool JSON is the model.)
PROVIDER_FAULTS = {"timeout", "backend_error"}

# A reply needing more than this many output tokens is runaway generation,
# not a legitimate tool-call batch (measured grand-tour replies: <=1770).
GENEROUS_CAP = 4000

# Run-level sleep beyond this marks the run as the HOST's fault (see blame()).
HOST_SLEEP_TOLERANCE_S = 5

# Beyond this a run is morbid curiosity, not anything a production system
# would tolerate. Flagged per run and counted in the timing report; it does
# NOT change grading.
SLOW_RUN_S = 120

GENERATION_URL = "https://openrouter.ai/api/v1/generation?id="

# two Chicago points, used once per sweep to ask: is real-street routing
# working at all? (Nothing is graded on this pair.)
ROUTE_PROBE = [(41.8781, -87.6298), (41.9300, -87.6500)]

# Where a run's wall-clock goes. Attempt time AND the backoff sleep after it
# land in the bucket of that attempt's status:
#   model_s          successful model calls (queue at provider + generation)
#   malformed_s      200s whose tool JSON didn't parse (the model's fault)
#   throttle_s       429s and the Retry-After/backoff waits they caused
#   rejected_s       400s (the reasoning-block retry: a compatibility cost)
#   provider_fail_s  5xx, socket timeouts, network errors, empty bodies
#   tool_s           our tools: geocoding, routing, weather, terrain, daylight
#   other_s          everything else: our loop, the auditors, harness overhead
TIME_BUCKETS = ("model_s", "malformed_s", "throttle_s", "rejected_s",
                "provider_fail_s", "tool_s", "other_s")


# ---------------------------------------------------------------------
# statistics: Wilson score interval for a binomial proportion. Narrower
# than normal-approx at the extremes and never leaves [0, 1] -- which is
# exactly where small-n pass rates (5/5, 0/5) live.
# ---------------------------------------------------------------------


def wilson(passes: int, n: int, z: float = Z) -> tuple[float, float]:
    """95% confidence interval for passes/n. (0, 0) when n == 0."""
    if n == 0:
        return (0.0, 0.0)
    phat = passes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


# ---------------------------------------------------------------------
# one run: drive the agent, tally the event stream, grade against the
# scenario's known feasibility, classify any failure.
# ---------------------------------------------------------------------


def dominant_failure(bounces: int, vetoes: int, nudges: int) -> str:
    """Which struggle used up the rounds, from the event counts."""
    if vetoes >= bounces and vetoes >= nudges and vetoes > 0:
        return "veto_livelock"     # never satisfied an oracle
    if bounces >= nudges and bounces > 0:
        return "schema_thrash"     # kept breaking tool schemas
    if nudges > 0:
        return "prose_stall"       # talked instead of calling tools
    return "no_submit"             # ran out of rounds, quietly


def classify(final: dict | None, error: dict | None,
             expected_feasible: bool | None,
             bounces: int, vetoes: int, nudges: int) -> str:
    """The deterministic verdict category for one run. expected_feasible
    None means the scenario's feasibility isn't fixed -- it depends on a
    live roll (difficulty) or the date (sunset) -- so we grade on
    RELAY-FIDELITY: reaching a validated submit_plan is the pass, because
    the auditors already forced the plan to match the tools' verdicts."""
    if final is not None:
        got = final["plan"].get("feasible")
        if expected_feasible is None or got == expected_feasible:
            return "pass"
        if expected_feasible is False and got is True:
            return "fabricated_feasible"   # claimed impossible = doable
        return "false_infeasible"          # needless refusal
    msg = (error or {}).get("message", "")
    if msg.startswith("no submit_plan within"):
        return dominant_failure(bounces, vetoes, nudges)
    # a reply cut off at OUR max_tokens: checked before JSONDecodeError,
    # because the truncated JSON is a symptom of the cap, not the model
    if "reply truncated at max_tokens" in msg:
        return "truncated"
    # the 300s run budget ran out between calls (was "NoneType: None")
    if "run budget exhausted" in msg:
        return "budget_exhausted"
    if "deadline" in msg:
        return "timeout"
    # the retry gave up: a persistent JSONDecodeError is the MODEL emitting
    # unparseable tool JSON (a structured-output failure, worth separating
    # from a provider that returned nothing).
    if "JSONDecodeError" in msg:
        return "malformed_output"
    return "backend_error"


def blame(outcome: str, timing: dict, host_sleep_s: float = 0.0,
          cap_tokens: int | None = None, degraded_routes: int = 0) -> str:
    """Who a run's result is attributable to: none | model | provider |
    harness | host. Only "model" and "none" runs are graded; the rest are
    excluded from pass rates (the rule agreed 2026-09-16: slowness or failure
    counts against the model unless it is explained by OUR failure, which
    includes 429 throttling, or OpenRouter's/the provider's failure).

      host      the machine slept mid-run: connections die, clocks lie.
                Applies to passes too, so exclusions don't bias pass rates.
      harness   the reply hit a max_tokens cap below GENEROUS_CAP (at or
                above it, a cut-off reply is runaway output: model), or the
                route degraded to straight-line distances
      provider  backend_error: 5xx, 429 exhaustion, dropped connections,
                unusable bodies, refused requests
      timeout / budget_exhausted are split by where the time went: provider
                if failures+throttling outweigh model time, harness if our
                tools do, else model (a slow model is the model's problem)
      model     every competence outcome: fabricated/false feasibility,
                livelocks, schema thrash, prose stalls, malformed output
    """
    if host_sleep_s > HOST_SLEEP_TOLERANCE_S:
        return "host"
    if degraded_routes:
        # the route came from straight-line distances (ORS quota or outage),
        # so feasibility -- what the run is graded on -- rests on fake
        # numbers. Excluded whatever the outcome, passes included.
        return "harness"
    if outcome == "pass":
        return "none"
    if outcome == "truncated":
        # below GENEROUS_CAP the cap was ours to get wrong (700 was); at or
        # above it, a reply that still runs out is runaway output
        return "harness" if (cap_tokens or 0) < GENEROUS_CAP else "model"
    if outcome == "backend_error":
        return "provider"
    if outcome in ("timeout", "budget_exhausted"):
        t = timing or {}
        provider_side = sum(t.get(k, 0.0) for k in ("provider_fail_s", "throttle_s", "rejected_s"))
        model_side = t.get("model_s", 0.0) + t.get("malformed_s", 0.0)
        tools = t.get("tool_s", 0.0)
        if provider_side >= model_side and provider_side >= tools:
            return "provider"
        if tools > model_side:
            return "harness"
        return "model"
    return "model"


def trim_event(ev: dict) -> dict:
    """Shrink an event for the archive: tool results carry a big GeoJSON
    geometry blob we never re-read; drop it, keep everything else. Terminal
    events drop `timing`, which the record already keeps as `calls`."""
    if ev.get("event") == "result" and isinstance(ev.get("result"), dict):
        out = dict(ev)
        out["result"] = {k: v for k, v in ev["result"].items() if k != "geometry"}
        return out
    if "timing" in ev:
        return {k: v for k, v in ev.items() if k != "timing"}
    return ev


# ---------------------------------------------------------------------
# latency attribution: where did a run's wall-clock go?
# ---------------------------------------------------------------------


def attempt_bucket(status: str | None) -> str:
    if status == "ok":
        return "model_s"
    if status == "http_429":
        return "throttle_s"
    if status == "http_400":
        return "rejected_s"
    if status == "unusable:JSONDecodeError":
        return "malformed_s"
    return "provider_fail_s"


def time_breakdown(seconds: float, calls: list[dict]) -> dict:
    """Split one run's wall-clock into TIME_BUCKETS from the agent's timing
    list. other_s is the remainder, so the buckets always sum to `seconds`
    (a small negative remainder from rounding is clamped to zero)."""
    out = {b: 0.0 for b in TIME_BUCKETS}
    chat_calls = attempts = cap_hits = 0
    slowest_call = 0.0
    for c in calls:
        if c.get("kind") == "tool":
            out["tool_s"] += c.get("seconds") or 0.0
            continue
        chat_calls += 1
        slowest_call = max(slowest_call, c.get("wall_s") or 0.0)
        for a in c.get("attempts") or []:
            attempts += 1
            cap_hits += 1 if (a.get("cap_hit") or a.get("status") == "unusable:JSONDecodeError"
                              and str(a.get("native_finish_reason") or a.get("finish_reason") or "").lower()
                              in ("length", "max_tokens", "max_output_tokens")) else 0
            out[attempt_bucket(a.get("status"))] += (
                (a.get("total_s") or 0.0) + (a.get("backoff_s") or 0.0))
    accounted = sum(v for k, v in out.items() if k != "other_s")
    out["other_s"] = max(seconds - accounted, 0.0)
    out = {k: round(v, 2) for k, v in out.items()}
    out.update({"chat_calls": chat_calls, "http_attempts": attempts,
                "cap_hits": cap_hits,   # replies cut off at our max_tokens
                "slowest_call_s": round(slowest_call, 2),
                "slow_run": seconds > SLOW_RUN_S})
    return out


def _fetch_one_generation(gen_id: str, key: str, tries: int = 6,
                          pause_s: float = 2.0) -> dict | None:
    """OpenRouter's own stats for one generation. They lag the response by
    a moment (404 until ready), so retry briefly. Free: no tokens billed."""
    req = urllib.request.Request(
        GENERATION_URL + gen_id,
        headers={"Authorization": f"Bearer {key}",
                 "User-Agent": "dog-walker-measure"})
    for _ in range(tries):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=20))["data"]
        except urllib.error.HTTPError as e:
            if e.code in (404, 429) or e.code >= 500:
                time.sleep(pause_s)
                continue
            return None
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            time.sleep(pause_s)
            continue
        # verified 2026-09-16: generation_time INCLUDES latency (time to
        # first token); both are milliseconds measured by OpenRouter
        return {
            "first_token_s": round((d.get("latency") or 0) / 1000, 2),
            "generation_s": round((d.get("generation_time") or 0) / 1000, 2),
            "provider": d.get("provider_name"),
            "completion_tokens": d.get("native_tokens_completion") or 0,
            "reasoning_tokens": d.get("native_tokens_reasoning") or 0,
            "finish_reason": d.get("finish_reason"),
            "cancelled": d.get("cancelled"),
            # OpenRouter retries across providers INSIDE one of our calls;
            # a non-200 here is a provider failure we never saw
            "provider_attempts": [
                {"provider": r.get("provider_name"), "status": r.get("status"),
                 "latency_s": round((r.get("latency") or 0) / 1000, 2)}
                for r in d.get("provider_responses") or []],
        }
    return None


def fetch_generation_stats(gen_ids: list[str]) -> dict[str, dict]:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    ids = [g for g in dict.fromkeys(gen_ids) if g]
    if not key or not ids:
        return {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        found = pool.map(lambda g: (g, _fetch_one_generation(g, key)), ids)
        return {g: st for g, st in found if st}


def attach_generation_stats(record: dict, fetch=fetch_generation_stats) -> None:
    """Annotate each successful attempt in record['calls'] with OpenRouter's
    stats and roll them into record['timing']['openrouter']. Runs AFTER the
    run's clock stops, so lookups never inflate measured latency."""
    ok = [a for c in record.get("calls") or [] if c.get("kind") == "chat"
          for a in c.get("attempts") or [] if a.get("status") == "ok"]
    stats = fetch([a.get("generation_id") for a in ok])
    roll = {"stats_found": 0, "stats_missing": 0, "first_token_s": 0.0,
            "generation_s": 0.0, "decode_s": 0.0, "openrouter_overhead_s": 0.0,
            "completion_tokens": 0, "reasoning_tokens": 0,
            "hidden_provider_failures": 0, "slowest_generation_s": 0.0}
    for a in ok:
        st = stats.get(a.get("generation_id") or "")
        if not st:
            roll["stats_missing"] += 1
            continue
        a["openrouter"] = st
        roll["stats_found"] += 1
        roll["first_token_s"] += st["first_token_s"]
        roll["generation_s"] += st["generation_s"]
        # decode = after the first token; tokens / decode_s is throughput
        roll["decode_s"] += max(st["generation_s"] - st["first_token_s"], 0.0)
        # our wall time for the attempt minus the provider's generation time:
        # OpenRouter routing + network, not the model
        roll["openrouter_overhead_s"] += max((a.get("total_s") or 0.0) - st["generation_s"], 0.0)
        roll["completion_tokens"] += st["completion_tokens"]
        roll["reasoning_tokens"] += st["reasoning_tokens"]
        roll["hidden_provider_failures"] += sum(
            1 for r in st["provider_attempts"] if r["status"] != 200)
        roll["slowest_generation_s"] = max(roll["slowest_generation_s"], st["generation_s"])
    for k in ("first_token_s", "generation_s", "decode_s", "openrouter_overhead_s"):
        roll[k] = round(roll[k], 2)
    roll["tokens_per_s"] = (round(roll["completion_tokens"] / roll["decode_s"], 1)
                            if roll["decode_s"] > 0 else None)
    record.setdefault("timing", {})["openrouter"] = roll


def execute_run(model: str, prompt: str, expected_feasible: bool) -> dict:
    """Run the agent once; return a graded, metric-rich record."""
    from dog_walker.agent import run_events

    events: list[dict] = []
    bounces = vetoes = nudges = tool_calls = degraded_routes = 0
    bounce_errors: list[str] = []
    # round-type accounting: WHERE in the loop does the model stumble?
    calls_by_tool: dict[str, int] = {}     # every attempt, by tool
    bounces_by_tool: dict[str, int] = {}   # schema failures, by tool
    vetoes_by_auditor: dict[str, int] = {}  # oracle rejections (all on submit)
    last_ok_tool = None                    # last successful dispatch = loop depth
    final = error = None
    started = time.monotonic()
    wall_started = time.time()
    for ev in run_events(prompt, model=model):
        events.append(ev)
        kind = ev["event"]
        if kind == "call":
            tool_calls += 1
            calls_by_tool[ev["name"]] = calls_by_tool.get(ev["name"], 0) + 1
        elif kind == "bounce":
            bounces += 1
            bounce_errors.append((ev.get("error") or "")[:160])
            bounces_by_tool[ev["name"]] = bounces_by_tool.get(ev["name"], 0) + 1
        elif kind == "audit_veto":
            vetoes += 1
            a = ev.get("auditor", "?")
            vetoes_by_auditor[a] = vetoes_by_auditor.get(a, 0) + 1
        elif kind == "nudge":
            nudges += 1
        elif kind == "result":
            last_ok_tool = ev["name"]
            if ev["name"] == "optimize_route" and isinstance(ev.get("result"), dict):
                if ev["result"].get("uses_real_streets") is False:
                    degraded_routes += 1
        elif kind == "final":
            final = ev
            break
        elif kind == "error":
            error = ev
            break
        if time.monotonic() - started > RUN_DEADLINE_S:
            error = {"event": "error", "rounds": 0, "usage": {},
                     "message": f"harness deadline {RUN_DEADLINE_S}s"}
            break
    seconds = round(time.monotonic() - started, 2)
    # monotonic time stops while the Mac sleeps; wall time doesn't
    host_sleep_s = round(max((time.time() - wall_started) - (time.monotonic() - started), 0.0), 1)

    outcome = classify(final, error, expected_feasible, bounces, vetoes, nudges)
    terminal = final or error or {}
    usage = terminal.get("usage") or {}
    record = {
        "model": model,
        "passed": outcome == "pass",
        "outcome": outcome,
        "expected_feasible": expected_feasible,
        "final_feasible": final["plan"].get("feasible") if final else None,
        "seconds": seconds,
        "rounds": terminal.get("rounds", 0),
        "tool_calls": tool_calls,
        "bounces": bounces,
        # the schema-violation strings, on EVERY run (passes included) so
        # referee friction that self-healed is still auditable -- this is
        # how max_relief_m:0-class bugs surface before they livelock
        "bounce_errors": bounce_errors[:20],
        "vetoes": vetoes,
        "nudges": nudges,
        # WHERE the loop stalls: attempts and failures per round type, so
        # a bounce/veto can be traced to the tool the model was producing
        "calls_by_tool": calls_by_tool,
        "bounces_by_tool": bounces_by_tool,
        "vetoes_by_auditor": vetoes_by_auditor,
        # for a terminal failure, the last tool that DID succeed -- the
        # depth the model reached before it broke
        "fail_after": None if outcome == "pass" else last_ok_tool,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cost": round(usage.get("cost", 0.0) or 0.0, 6),
        "error": error.get("message") if error else None,
        # provider flakiness, captured post-hoc from the terminal event:
        # extra chat attempts beyond the first (retries), the upstream
        # providers OpenRouter routed to, and whether this run is a provider
        # fault to be excluded from the competence denominator.
        "retries": terminal.get("retries", 0),
        "throttles": terminal.get("throttles", 0),
        "providers": terminal.get("providers", []),
        # latency attribution: per-call/attempt/tool timings, and the run's
        # wall-clock split into TIME_BUCKETS
        "calls": terminal.get("timing") or [],
        "timing": time_breakdown(seconds, terminal.get("timing") or []),
        "host_sleep_s": host_sleep_s,
        "degraded_routes": degraded_routes,
        "started_at": datetime.fromtimestamp(wall_started).astimezone().isoformat(timespec="seconds"),
    }
    cap = None
    if outcome == "truncated":
        found = re.search(r"max_tokens=(\d+)", record["error"] or "")
        cap = int(found.group(1)) if found else None
    record["blame"] = blame(outcome, record["timing"], host_sleep_s, cap_tokens=cap,
                            degraded_routes=degraded_routes)
    # field name kept for archive/site compatibility; it now means "not
    # attributable to the model", which covers provider, harness and host
    record["provider_excluded"] = record["blame"] not in ("model", "none")
    if outcome != "pass":
        # failures are the content: keep the whole transcript
        record["transcript"] = [trim_event(e) for e in events]
    return record


# ---------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------


def summarize(records: list[dict]) -> dict:
    """Roll per-run records up into pass rates (with intervals), median
    cost/latency, and a failure histogram, for a group of runs.

    Competence and provider weather are graded apart: provider-fault runs
    (PROVIDER_FAULTS) are dropped from the pass-rate denominator and the
    cost/latency medians, then reported separately as `provider_excluded`.
    Flakiness -- the retries a run needed to succeed at all -- is counted
    across EVERY attempted run, passes included, because a model that only
    passes after three retries is a different reliability story than one
    that passes clean."""
    excluded = [r for r in records if r.get("provider_excluded")]
    scored = [r for r in records if not r.get("provider_excluded")]
    n = len(scored)
    passes = sum(1 for r in scored if r["passed"])
    lo, hi = wilson(passes, n)
    costs = [r["cost"] for r in scored]
    secs = [r["seconds"] for r in scored]
    rounds = [r["rounds"] for r in scored]
    failures: dict[str, int] = {}
    for r in scored:
        if not r["passed"]:
            failures[r["outcome"]] = failures.get(r["outcome"], 0) + 1
    total_retries = sum(r.get("retries", 0) for r in records)
    total_throttles = sum(r.get("throttles", 0) for r in records)
    flaky_runs = sum(1 for r in records if r.get("retries", 0) > 0)
    return {
        "n": n,
        "passes": passes,
        "pass_rate": round(passes / n, 3) if n else 0.0,
        "wilson_lo": round(lo, 3),
        "wilson_hi": round(hi, 3),
        "median_cost": round(statistics.median(costs), 6) if costs else 0.0,
        "total_cost": round(sum(costs), 6),
        "median_seconds": round(statistics.median(secs), 1) if secs else 0.0,
        "median_rounds": statistics.median(rounds) if rounds else 0,
        "failures": failures,
        # provider weather, kept apart from competence
        "attempted": len(records),
        "provider_excluded": len(excluded),
        "total_retries": total_retries,
        "total_throttles": total_throttles,
        "flaky_runs": flaky_runs,
    }


def by_key(records: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r[key], []).append(r)
    return groups


# ---------------------------------------------------------------------
# model list + CLI
# ---------------------------------------------------------------------


def default_models() -> list[str]:
    """The picker allowlist if we have one, else the pinned pair."""
    try:
        data = json.loads(MODELS_FILE.read_text())
        ids = [m["id"] for m in data.get("models", [])]
        if ids:
            return ids
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        pass
    from dog_walker.qualify import PINNED
    return list(PINNED)


def parse_args(argv: list[str]) -> dict:
    opts = {"k": DEFAULT_K, "models": None, "scenarios": None, "dry": False,
            "gen_stats": True, "allow_degraded_routes": False}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--dry":
            opts["dry"] = True
        elif a == "--no-gen-stats":
            opts["gen_stats"] = False
        elif a == "--allow-degraded-routes":
            opts["allow_degraded_routes"] = True
        elif a == "--k" and i + 1 < len(argv):
            opts["k"] = int(argv[i + 1]); i += 1
        elif a == "--models" and i + 1 < len(argv):
            opts["models"] = argv[i + 1].split(","); i += 1
        elif a == "--scenarios" and i + 1 < len(argv):
            opts["scenarios"] = argv[i + 1].split(","); i += 1
        i += 1
    return opts


def print_report(models: list[str], scenario_ids: list[str],
                 records: list[dict]) -> None:
    """A human-readable summary to stdout; the JSON archive has the rest."""
    per_model = by_key(records, "model")
    print("\n" + "=" * 72)
    print("MEASUREMENT SUMMARY")
    print("=" * 72)
    for model in models:
        mine = per_model.get(model, [])
        if not mine:
            continue
        agg = summarize(mine)
        print(f"\n{model}")
        print(f"  overall: {agg['passes']}/{agg['n']} "
              f"({agg['pass_rate']:.0%}, 95% CI "
              f"{agg['wilson_lo']:.0%}-{agg['wilson_hi']:.0%})  "
              f"${agg['total_cost']:.4f} total, "
              f"${agg['median_cost']:.4f}/run median, "
              f"{agg['median_seconds']:.0f}s median")
        prov = []
        if agg["provider_excluded"]:
            prov.append(f"{agg['provider_excluded']}/{agg['attempted']} "
                        f"excluded (provider fault)")
        if agg["flaky_runs"]:
            prov.append(f"{agg['flaky_runs']} runs needed retries "
                        f"({agg['total_retries']} total, "
                        f"{agg['total_throttles']} were 429 throttles)")
        if prov:
            print("  provider: " + "; ".join(prov))
        scn_groups: dict[str, list[dict]] = {}
        for r in mine:
            scn_groups.setdefault(r["scenario"], []).append(r)
        for sid in scenario_ids:
            g = scn_groups.get(sid)
            if not g:
                continue
            s = summarize(g)
            fail = ", ".join(f"{k}:{v}" for k, v in s["failures"].items())
            print(f"    {sid:20s} {s['passes']}/{s['n']} "
                  f"[{s['wilson_lo']:.0%}-{s['wilson_hi']:.0%}]  "
                  f"${s['median_cost']:.4f}  {s['median_seconds']:.0f}s"
                  + (f"  ({fail})" if fail else ""))
    # global failure taxonomy
    taxonomy: dict[str, int] = {}
    for r in records:
        if not r["passed"]:
            taxonomy[r["outcome"]] = taxonomy.get(r["outcome"], 0) + 1
    if taxonomy:
        print("\nfailure taxonomy (all models):")
        for cat, count in sorted(taxonomy.items(), key=count_desc):
            print(f"  {cat:22s} {count}")
    print()


def count_desc(item: tuple) -> int:
    return -item[1]


# ---------------------------------------------------------------------
# report data: merge one or more archives into the compact JSON the
# walker.purr.io report imports, so a re-run refreshes the charts by
# rebuild, not by hand-editing. Later archives WIN per model, so a
# targeted re-run (e.g. gpt-oss-120b under a bug fix) overrides that
# model's rows from the full sweep while leaving the others untouched.
# ---------------------------------------------------------------------


def _rank_key(m: dict) -> tuple:
    return (-m["pass"], m["cost"])


def _tax_key(t: dict) -> int:
    return -t["n"]


def build_report_data(archive_paths: list[str], out_path: Path = REPORT_DATA) -> dict:
    """Merge archives (in order; later wins per model) into the report's
    data file: per-model pass rate + Wilson CI + median cost/latency +
    per-scenario pass counts, plus the failure taxonomy and totals."""
    model_runs: dict[str, list[dict]] = {}
    meta = {}
    for path in archive_paths:
        d = json.loads(Path(path).read_text())
        meta = {"generated_at": d.get("generated_at", ""),
                "k": d.get("config", {}).get("k")}
        for model, runs in by_key(d["runs"], "model").items():
            model_runs[model] = runs      # later archive overrides

    scenarios = list(SCENARIOS)
    models, all_runs = [], []
    for model, runs in model_runs.items():
        all_runs.extend(runs)
        agg = summarize(runs)
        cells_by_scn = by_key(runs, "scenario")
        cells = [sum(1 for r in cells_by_scn.get(s, []) if r["passed"])
                 for s in scenarios]
        lo, hi = wilson(agg["passes"], agg["n"])
        models.append({
            "id": model, "short": SHORT.get(model, model.split("/")[-1]),
            "pass": agg["passes"], "n": agg["n"],
            "ci": [round(lo, 2), round(hi, 2)],
            "cost": agg["median_cost"], "sec": round(agg["median_seconds"]),
            "cells": cells, "note": NOTE.get(model),
        })
    models.sort(key=_rank_key)

    tax: dict[str, int] = {}
    for r in all_runs:
        if not r["passed"]:
            tax[r["outcome"]] = tax.get(r["outcome"], 0) + 1
    taxonomy = sorted(({"name": k, "n": v} for k, v in tax.items()), key=_tax_key)

    data = {
        "generated": (meta.get("generated_at") or "")[:10],
        "totals": {"runs": len(all_runs),
                   "cost": round(sum(r["cost"] for r in all_runs), 4),
                   "models": len(models), "scenarios": len(scenarios),
                   "k": meta.get("k")},
        "scenarios": scenarios,
        "models": models,
        "taxonomy": taxonomy,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"wrote {out_path}: {len(models)} models, {len(all_runs)} runs, "
          f"${data['totals']['cost']}")
    return data


# ---------------------------------------------------------------------
# round-type analysis: WHERE in the loop does a model stumble? Ordered
# easy -> hard a priori (a lookup emits a few params and relays a verdict;
# optimize_route transcribes every dog's attributes; submit_plan must
# integrate ALL prior results and judge feasibility). If corrections
# climb with that order, the difficulty is structurally predictable.
# ---------------------------------------------------------------------

ROUND_ORDER = [
    "geocode_addresses",  # easy: one address list
    "check_weather",      # easy: params -> relay a verdict
    "check_terrain",      # easy
    "check_daylight",     # easy
    "optimize_route",     # medium: every dog's attributes, one synthesis (sunset)
    "submit_plan",        # hard: integrate all results + judge feasibility
]


def _sum_counters(records: list[dict], field: str) -> dict:
    total: dict = {}
    for r in records:
        for k, v in (r.get(field) or {}).items():
            total[k] = total.get(k, 0) + v
    return total


def round_breakdown(records: list[dict]) -> dict:
    """Per round type: attempts and corrections (schema bounces, plus for
    submit_plan the auditor vetoes), the correction rate, the veto
    breakdown, and where terminal failures landed."""
    calls = _sum_counters(records, "calls_by_tool")
    bounces = _sum_counters(records, "bounces_by_tool")
    vetoes = _sum_counters(records, "vetoes_by_auditor")
    total_vetoes = sum(vetoes.values())
    rows = []
    for tool in ROUND_ORDER:
        attempts = calls.get(tool, 0)
        corr = bounces.get(tool, 0)
        if tool == "submit_plan":
            corr += total_vetoes  # every veto is a rejected submit_plan
        rows.append({
            "round": tool, "attempts": attempts, "corrections": corr,
            "rate": round(corr / attempts, 3) if attempts else 0.0,
        })
    fail_after: dict = {}
    for r in records:
        if not r.get("passed"):
            fa = r.get("fail_after")
            if fa:
                fail_after[fa] = fail_after.get(fa, 0) + 1
    return {
        "rows": rows,
        "vetoes_by_auditor": vetoes,
        "malformed_output": sum(1 for r in records
                                if r.get("outcome") == "malformed_output"),
        "fail_after": fail_after,
        "runs": len(records),
    }


def print_round_report(records: list[dict]) -> None:
    per_model = by_key(records, "model")
    print("\n" + "=" * 72)
    print("ROUND-TYPE BREAKDOWN  (a priori easy -> hard)")
    print("where corrections land per round type; does the rate climb?")
    print("=" * 72)
    for model, rs in per_model.items():
        b = round_breakdown(rs)
        print(f"\n{model}  ({b['runs']} runs)")
        print(f"  {'round type':20s} {'attempts':>8} {'corrections':>12} {'rate':>7}")
        for row in b["rows"]:
            print(f"  {row['round']:20s} {row['attempts']:>8} "
                  f"{row['corrections']:>12} {row['rate']:>6.0%}")
        if b["vetoes_by_auditor"]:
            vs = ", ".join(f"{k}:{v}" for k, v in b["vetoes_by_auditor"].items())
            print(f"    submit_plan vetoes by auditor: {vs}")
        if b["malformed_output"]:
            fa = ", ".join(f"after {k}:{v}" for k, v in b["fail_after"].items())
            print(f"    malformed_output: {b['malformed_output']}  ({fa})")
    print()


def _load_runs(archive_paths: list[str]) -> list[dict]:
    """Runs from measure-*.json archives or incremental runs-*.jsonl logs."""
    runs: list[dict] = []
    for p in archive_paths:
        text = Path(p).read_text()
        if p.endswith(".jsonl"):
            runs.extend(json.loads(line) for line in text.splitlines() if line.strip())
        else:
            runs.extend(json.loads(text).get("runs", []))
    return runs


# ---------------------------------------------------------------------
# timing report: where did the time go, per model?
# ---------------------------------------------------------------------


def timing_summary(records: list[dict]) -> dict:
    """Bucket totals and shares for a group of runs, plus OpenRouter's
    view. Runs without timing (archives older than 2026-09-16) are skipped."""
    timed = [r for r in records if isinstance(r.get("timing"), dict)
             and "model_s" in r["timing"]]
    total = sum(r["seconds"] for r in timed)
    buckets = {b: round(sum(r["timing"].get(b, 0.0) for r in timed), 1)
               for b in TIME_BUCKETS}
    ors = [r["timing"]["openrouter"] for r in timed if "openrouter" in r["timing"]]

    def osum(k: str) -> float:
        return round(sum(o.get(k) or 0 for o in ors), 1)

    decode = osum("decode_s")
    blames: dict[str, int] = {}
    for r in timed:
        b = r.get("blame", "?")
        blames[b] = blames.get(b, 0) + 1
    return {
        "blame": blames,
        "host_sleep_s": round(sum(r.get("host_sleep_s") or 0.0 for r in timed), 1),
        "runs": len(timed),
        "slow_runs": sum(1 for r in timed if r["seconds"] > SLOW_RUN_S),
        "total_s": round(total, 1),
        "median_s": round(statistics.median([r["seconds"] for r in timed]), 1) if timed else 0.0,
        "buckets": buckets,
        "shares": {b: (round(v / total, 3) if total else 0.0) for b, v in buckets.items()},
        "openrouter": {
            "runs_with_stats": len(ors),
            "stats_missing": int(osum("stats_missing")),
            "first_token_s": osum("first_token_s"),
            "generation_s": osum("generation_s"),
            "openrouter_overhead_s": osum("openrouter_overhead_s"),
            "hidden_provider_failures": int(osum("hidden_provider_failures")),
            "median_reasoning_tokens": (statistics.median(
                [o.get("reasoning_tokens") or 0 for o in ors]) if ors else 0),
            "tokens_per_s": round(osum("completion_tokens") / decode, 1) if decode else None,
        },
    }


def print_timing_report(records: list[dict], slowest: int = 25) -> None:
    per_model = by_key(records, "model")
    print("\n" + "=" * 72)
    print(f"TIMING  (where the wall-clock went; slow = over {SLOW_RUN_S}s)")
    print("buckets: model | malformed | throttle(429) | rejected(400) | "
          "provider_fail | tools | other")
    print("=" * 72)
    for model, rs in per_model.items():
        t = timing_summary(rs)
        if not t["runs"]:
            continue
        sh = t["shares"]
        print(f"\n{model}  ({t['runs']} runs, {t['slow_runs']} slow, "
              f"median {t['median_s']:.0f}s, total {t['total_s']:.0f}s)")
        deg = sum(1 for r in rs if r.get("degraded_routes"))
        print("  blame: " + ", ".join(f"{k} {v}" for k, v in sorted(t["blame"].items()))
              + (f"  ({deg} runs on straight-line distances)" if deg else "")
              + (f"  (host slept {t['host_sleep_s']:.0f}s during runs)" if t["host_sleep_s"] else ""))
        print("  share: " + "  ".join(
            f"{b[:-2]} {sh[b]:.0%}" for b in TIME_BUCKETS if sh[b] >= 0.005))
        o = t["openrouter"]
        if o["runs_with_stats"]:
            print(f"  openrouter: first-token {o['first_token_s']:.0f}s, "
                  f"generation {o['generation_s']:.0f}s, "
                  f"routing+network {o['openrouter_overhead_s']:.0f}s, "
                  f"{o['tokens_per_s']} tok/s, median reasoning tokens/run "
                  f"{o['median_reasoning_tokens']:.0f}, hidden provider failovers "
                  f"{o['hidden_provider_failures']}"
                  + (f", stats missing {o['stats_missing']}" if o["stats_missing"] else ""))
    slow = sorted((r for r in records if isinstance(r.get("timing"), dict)
                   and r["seconds"] > SLOW_RUN_S),
                  key=lambda r: -r["seconds"])[:slowest]
    if slow:
        print(f"\nslowest runs (top {len(slow)}):")
        for r in slow:
            tm = r["timing"]
            top = sorted(((b, tm.get(b, 0.0)) for b in TIME_BUCKETS), key=lambda x: -x[1])[:2]
            o = tm.get("openrouter") or {}
            print(f"  {r['seconds']:5.0f}s  {r['model']:28s} {r.get('scenario', ''):18s} "
                  f"{r['outcome']:16s} " + ", ".join(f"{b[:-2]} {v:.0f}s" for b, v in top)
                  + f"  calls {tm.get('chat_calls')}, slowest {tm.get('slowest_call_s', 0):.0f}s"
                  + (f", reasoning tok {o.get('reasoning_tokens')}" if o else ""))
    print()


def main(argv: list[str]) -> None:
    if argv and argv[0] == "--build-report":
        paths = argv[1:] or sorted(str(p) for p in ARCHIVE_DIR.glob("measure-*.json"))
        build_report_data(paths)
        return
    if argv and argv[0] == "--timing-report":
        paths = argv[1:] or sorted(str(p) for p in ARCHIVE_DIR.glob("measure-*.json"))
        print_timing_report(_load_runs(paths))
        return
    if argv and argv[0] == "--round-report":
        paths = argv[1:] or sorted(str(p) for p in ARCHIVE_DIR.glob("measure-*.json"))
        print_round_report(_load_runs(paths))
        return
    opts = parse_args(argv)
    seed_geocode_cache()
    models = opts["models"] or default_models()
    scenario_ids = opts["scenarios"] or list(SCENARIOS)
    bad = [s for s in scenario_ids if s not in SCENARIOS]
    if bad:
        raise SystemExit(f"unknown scenarios: {bad}; have {list(SCENARIOS)}")
    k = opts["k"]
    total = len(models) * len(scenario_ids) * k

    print(f"models:    {len(models)}  {models}")
    print(f"scenarios: {len(scenario_ids)}  {scenario_ids}")
    print(f"k={k}  ->  {total} live runs (each is a real, paid OpenRouter call)")
    if opts["dry"]:
        print("\n--dry: nothing run.")
        return

    # INTERLEAVE models round-robin, not model-at-a-time. Running one model
    # to completion before the next confounds model quality with the
    # provider conditions of that time slice -- which is exactly how a
    # single provider's bad half-hour became a "qwen3.7 is slow" finding.
    # Order: pass -> scenario -> model, so within each pass every model hits
    # the same scenario back-to-back (~same minute), and each model's k runs
    # are spread across the whole session.
    from dog_walker import agent as agent_mod

    # Hold the Mac awake for the life of this process. The 2026-09-16 sweep
    # took 18.8h of wall time for 5.5h of work because the Mac idle-slept,
    # waking ~45s every 15 min; sleep-interrupted runs failed at 7x the rate.
    # -i blocks idle sleep, -s blocks system sleep on AC, -w ends with us.
    # A closed lid on battery still sleeps; host_sleep_s catches that.
    if shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])
        print("caffeinate: holding the Mac awake for this sweep")
    else:
        print("WARNING: no caffeinate; if this machine sleeps, runs are marked blame=host")

    # the harness draws no map: skip the ORS geometry request per route
    toolbox.WANT_GEOMETRY = False
    # and refuse to grade models on straight-line distances
    _, real_streets = toolbox._walking_matrix(ROUTE_PROBE)
    if not real_streets:
        why = ", ".join(f"{k} x{v}" for k, v in toolbox.ROUTE_FALLBACKS.items()) or "unknown"
        print("\n" + "!" * 72)
        print("ROUTING DEGRADED: OpenRouteService is not answering "
              f"({why}).")
        print("Routes would use straight-line distances, so feasibility -- what")
        print("every run is graded on -- would be fiction. The free quota resets")
        print("daily; cached routes (data/route_cache.json) are reused when they")
        print("exist. Re-run later, or pass --allow-degraded-routes to proceed")
        print("with every affected run marked blame=harness and left ungraded.")
        print("!" * 72 + "\n", flush=True)
        if not opts["allow_degraded_routes"]:
            raise SystemExit(2)
    else:
        print("routing: real street distances available")

    ARCHIVE_DIR.mkdir(exist_ok=True)
    tag = f"{datetime.now().astimezone():%Y-%m-%dT%H-%M-%S}"
    agent_mod.CALL_LOG = ARCHIVE_DIR / f"calls-{tag}.jsonl"
    runs_log = ARCHIVE_DIR / f"runs-{tag}.jsonl"
    print(f"live call log: {agent_mod.CALL_LOG}")
    print(f"runs saved as they finish: {runs_log}", flush=True)

    records: list[dict] = []
    running_cost = 0.0
    done = 0
    for run_i in range(k):
        for sid in scenario_ids:
            scenario = SCENARIOS[sid]
            prompt = scenario_prompt(scenario)
            expected = scenario["expected_feasible"]
            for model in models:
                done += 1
                agent_mod.CALL_LOG_CONTEXT = {"n": done, "scenario": sid, "run": run_i}
                rec = execute_run(model, prompt, expected)
                rec["scenario"] = sid
                rec["run"] = run_i
                if opts["gen_stats"]:
                    attach_generation_stats(rec)   # after the clock stops
                records.append(rec)
                with open(runs_log, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                running_cost += rec["cost"]
                flag = "ok " if rec["passed"] else (
                    "?? " if rec["provider_excluded"] else "XX ")
                tm = rec["timing"]
                where = " ".join(f"{b[:-2]}={tm[b]:.0f}" for b in TIME_BUCKETS if tm[b] >= 1)
                who = "" if rec["blame"] in ("model", "none") else f" blame={rec['blame']}"
                print(f"[{done:3d}/{total}] {flag}{model:28s} {sid:20s} "
                      f"{rec['outcome']:18s} {rec['seconds']:5.0f}s "
                      f"${rec['cost']:.4f}  (${running_cost:.3f} so far)"
                      f"  [{where}]{who}",
                      flush=True)

    write_archive(models, scenario_ids, k, records)
    print_report(models, scenario_ids, records)
    print_round_report(records)
    print_timing_report(records)


def write_archive(models: list[str], scenario_ids: list[str], k: int,
                  records: list[dict]) -> Path:
    ARCHIVE_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().astimezone()
    path = ARCHIVE_DIR / f"measure-{stamp:%Y-%m-%dT%H-%M-%S}.json"
    per_model = {m: summarize(rs) for m, rs in by_key(records, "model").items()}
    per_cell = {}
    for r in records:
        per_cell.setdefault(f"{r['model']}::{r['scenario']}", []).append(r)
    cell_aggs = {key: summarize(rs) for key, rs in per_cell.items()}
    path.write_text(json.dumps({
        "generated_at": stamp.isoformat(),
        "config": {"models": models, "scenarios": scenario_ids, "k": k},
        "per_model": per_model,
        "per_cell": cell_aggs,
        "runs": records,
    }, indent=2))
    print(f"\nwrote {path}")
    return path


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
