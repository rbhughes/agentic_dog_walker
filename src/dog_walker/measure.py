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
"""

from __future__ import annotations

import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

from dog_walker.scenarios import SCENARIOS, scenario_prompt, seed_geocode_cache

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_FILE = REPO_ROOT / "models.json"
ARCHIVE_DIR = REPO_ROOT / "measurements"

DEFAULT_K = 5           # runs per (model, scenario)
RUN_DEADLINE_S = 240    # wall-clock guard per single run
Z = 1.96                # 95% Wilson interval


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


def classify(final: dict | None, error: dict | None, expected_feasible: bool,
             bounces: int, vetoes: int, nudges: int) -> str:
    """The deterministic verdict category for one run."""
    if final is not None:
        got = final["plan"].get("feasible")
        if got == expected_feasible:
            return "pass"
        if expected_feasible is False and got is True:
            return "fabricated_feasible"   # claimed impossible = doable
        return "false_infeasible"          # needless refusal
    msg = (error or {}).get("message", "")
    if msg.startswith("no submit_plan within"):
        return dominant_failure(bounces, vetoes, nudges)
    if "deadline" in msg:
        return "timeout"
    return "backend_error"


def trim_event(ev: dict) -> dict:
    """Shrink an event for the archive: tool results carry a big GeoJSON
    geometry blob we never re-read; drop it, keep everything else."""
    if ev.get("event") == "result" and isinstance(ev.get("result"), dict):
        out = dict(ev)
        out["result"] = {k: v for k, v in ev["result"].items() if k != "geometry"}
        return out
    return ev


def execute_run(model: str, prompt: str, expected_feasible: bool) -> dict:
    """Run the agent once; return a graded, metric-rich record."""
    from dog_walker.agent import run_events

    events: list[dict] = []
    bounces = vetoes = nudges = tool_calls = 0
    final = error = None
    started = time.monotonic()
    for ev in run_events(prompt, model=model):
        events.append(ev)
        kind = ev["event"]
        if kind == "call":
            tool_calls += 1
        elif kind == "bounce":
            bounces += 1
        elif kind == "audit_veto":
            vetoes += 1
        elif kind == "nudge":
            nudges += 1
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
        "vetoes": vetoes,
        "nudges": nudges,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cost": round(usage.get("cost", 0.0) or 0.0, 6),
        "error": error.get("message") if error else None,
    }
    if outcome != "pass":
        # failures are the content: keep the whole transcript
        record["transcript"] = [trim_event(e) for e in events]
    return record


# ---------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------


def summarize(records: list[dict]) -> dict:
    """Roll per-run records up into pass rates (with intervals), median
    cost/latency, and a failure histogram, for a group of runs."""
    n = len(records)
    passes = sum(1 for r in records if r["passed"])
    lo, hi = wilson(passes, n)
    costs = [r["cost"] for r in records]
    secs = [r["seconds"] for r in records]
    rounds = [r["rounds"] for r in records]
    failures: dict[str, int] = {}
    for r in records:
        if not r["passed"]:
            failures[r["outcome"]] = failures.get(r["outcome"], 0) + 1
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
    opts = {"k": DEFAULT_K, "models": None, "scenarios": None, "dry": False}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--dry":
            opts["dry"] = True
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


def main(argv: list[str]) -> None:
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

    records: list[dict] = []
    running_cost = 0.0
    done = 0
    for model in models:
        for sid in scenario_ids:
            scenario = SCENARIOS[sid]
            prompt = scenario_prompt(scenario)
            expected = scenario["expected_feasible"]
            for run_i in range(k):
                done += 1
                rec = execute_run(model, prompt, expected)
                rec["scenario"] = sid
                rec["run"] = run_i
                records.append(rec)
                running_cost += rec["cost"]
                flag = "ok " if rec["passed"] else "XX "
                print(f"[{done:3d}/{total}] {flag}{model:28s} {sid:20s} "
                      f"{rec['outcome']:18s} {rec['seconds']:5.0f}s "
                      f"${rec['cost']:.4f}  (${running_cost:.3f} so far)",
                      flush=True)

    write_archive(models, scenario_ids, k, records)
    print_report(models, scenario_ids, records)


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
