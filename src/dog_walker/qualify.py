"""Discover and QUALIFY models for the picker.

The picker's allowlist is generated, not curated: candidates come
from OpenRouter's public catalog filtered by hard criteria, and every
candidate must then PASS a live end-to-end run before it may appear.
"Capable" is measured on our task, not read off a leaderboard --
the gate is exactly the failure observed live from
gemini-2.5-flash-lite (geocode, then submit fabricated times and
verdicts): a passing model must reach a validated submit_plan with a
real clock-timed route, within bounded interventions and time.

Criteria (owner's spec, 2026-09-11):
  * must support tools AND reasoning (catalog supported_parameters)
  * price comparable to the default qwen3-8b (caps below)
  * pinned exceptions bypass the price cap (the frontier contrast)

Run:    uv run python -m dog_walker.qualify            # full run
        uv run python -m dog_walker.qualify --discover # list, no gate

Writes models.json at the repo root: {generated_at, models, rejected}.
The service loads `models` as the picker allowlist at startup and
falls back to the pinned pair if the file is missing.
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

CATALOG_URL = "https://openrouter.ai/api/v1/models"
MODELS_FILE = Path(__file__).resolve().parents[2] / "models.json"

# price caps, USD per MILLION tokens -- "comparable to qwen3-8b"
# (~$0.05-0.15/M as of 2026-09) with headroom, not 20x it
PROMPT_CAP_PER_M = 0.30
COMPLETION_CAP_PER_M = 1.20

# always in the list, price cap notwithstanding
PINNED: dict[str, str] = {
    "qwen/qwen3-8b": "Qwen3 8B — the bake-off winner (default)",
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5 — frontier contrast (pinned)",
}

MAX_CANDIDATES = 10   # qualification is live and costs pennies; bound it
KEEP = 4              # discovered slots in the final list (plus PINNED)
MAX_VETOES = 2        # a pass may need auditor help, not a tug-of-war
MAX_SECONDS = 120     # wall clock per qualification run


def fetch_catalog() -> list[dict]:
    with urllib.request.urlopen(CATALOG_URL, timeout=30) as resp:
        return json.load(resp)["data"]


def discover(catalog: list[dict]) -> list[dict]:
    """Filter the catalog by the hard criteria; cheapest first."""
    out = []
    for m in catalog:
        mid = m.get("id", "")
        params = m.get("supported_parameters") or []
        pricing = m.get("pricing") or {}
        try:
            prompt_m = float(pricing.get("prompt", "inf")) * 1e6
            completion_m = float(pricing.get("completion", "inf")) * 1e6
        except (TypeError, ValueError):
            continue
        if (
            mid in PINNED
            or ":" in mid.split("/", 1)[-1]    # :free/:batch/:extended variants
            or mid.startswith("openrouter/")   # router pseudo-models
            or "tools" not in params
            or "reasoning" not in params
            or prompt_m <= 0                   # 0 = free alias, <0 = "variable"
            or prompt_m > PROMPT_CAP_PER_M
            or completion_m > COMPLETION_CAP_PER_M
        ):
            continue
        out.append(
            {
                "id": mid,
                "name": m.get("name", mid),
                "prompt_per_m": round(prompt_m, 3),
                "completion_per_m": round(completion_m, 3),
            }
        )
    out.sort(key=price_key)
    return out[:MAX_CANDIDATES]


def price_key(candidate: dict) -> tuple:
    return (candidate["prompt_per_m"], candidate["completion_per_m"])


def qualify(model_id: str) -> dict:
    """The gate: one real end-to-end preset run. Pass = a validated
    submit_plan backed by a clock-timed route, within the veto and
    time budgets. Every failure mode observed in the wild maps to a
    reason string here."""
    from dog_walker.agent import run_events
    from dog_walker.presets import PRESETS, build_request, seed_geocode_cache

    seed_geocode_cache()
    p = PRESETS["loop-lunch-hour"]
    prompt = build_request(p["start_address"], p["start_time"], p["pets"])

    started = time.monotonic()
    vetoes = bounces = 0
    route_ok = False
    for ev in run_events(prompt, model=model_id):
        kind = ev["event"]
        if kind == "audit_veto":
            vetoes += 1
        elif kind == "bounce":
            bounces += 1
        elif kind == "result" and isinstance(ev.get("result"), dict):
            if "timeline" in ev["result"]:
                route_ok = True
        elif kind == "final":
            seconds = round(time.monotonic() - started, 1)
            if not route_ok:
                return {"passed": False, "reason": "finished without a route"}
            if vetoes > MAX_VETOES:
                return {"passed": False,
                        "reason": f"{vetoes} auditor vetoes (budget {MAX_VETOES})"}
            if seconds > MAX_SECONDS:
                return {"passed": False, "reason": f"too slow: {seconds}s"}
            return {"passed": True, "seconds": seconds,
                    "vetoes": vetoes, "bounces": bounces}
        elif kind == "error":
            return {"passed": False, "reason": ev["message"][:160]}
        if time.monotonic() - started > MAX_SECONDS:
            return {"passed": False, "reason": f"deadline {MAX_SECONDS}s"}
    return {"passed": False, "reason": "stream ended without final"}


def label_for(candidate: dict, verdict: dict) -> str:
    return (
        f"{candidate['name']} — ${candidate['prompt_per_m']:.2f}/M in, "
        f"qualified {verdict['seconds']:.0f}s"
    )


def main(discover_only: bool = False) -> None:
    catalog = fetch_catalog()
    candidates = discover(catalog)
    print(f"catalog: {len(catalog)} models; candidates after criteria: "
          f"{len(candidates)}")
    for c in candidates:
        print(f"  {c['id']:48s} ${c['prompt_per_m']:.3f}/M in  "
              f"${c['completion_per_m']:.3f}/M out")
    if discover_only:
        return

    passed, rejected = [], []
    for c in candidates:
        print(f"qualifying {c['id']} ...", flush=True)
        verdict = qualify(c["id"])
        if verdict["passed"]:
            print(f"  PASS  {verdict['seconds']}s, vetoes={verdict['vetoes']}")
            passed.append((c, verdict))
        else:
            print(f"  FAIL  {verdict['reason']}")
            rejected.append({"id": c["id"], "reason": verdict["reason"]})
        if len(passed) >= KEEP:
            break

    models = [{"id": mid, "label": label} for mid, label in PINNED.items()]
    models += [{"id": c["id"], "label": label_for(c, v)} for c, v in passed]
    MODELS_FILE.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(),
                "criteria": {
                    "prompt_cap_per_m": PROMPT_CAP_PER_M,
                    "completion_cap_per_m": COMPLETION_CAP_PER_M,
                    "requires": ["tools", "reasoning"],
                    "gate": "end-to-end preset run: validated submit_plan "
                            "with clock-timed route",
                },
                "models": models,
                "rejected": rejected,
            },
            indent=2,
        )
    )
    print(f"\nwrote {MODELS_FILE} with {len(models)} models "
          f"({len(rejected)} rejected)")


if __name__ == "__main__":
    import sys

    main(discover_only="--discover" in sys.argv)
