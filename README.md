# Agentic Dog Walker 🐕

**Live demo: [walker.purr.io](https://walker.purr.io)** — watch a small language
model plan dog-walking routes: every tool call, every validation bounce, every
audit veto, streamed as it happens.

You give it a starting point and a few dogs — each with an address, a walk
length (20, 30, or 60 minutes), optional prep time, and a comfortable
temperature range. It looks up the addresses, solves the visiting order over
real street distances, checks the weather for each dog's actual walk window at
that dog's location, and returns a schedule with a per-dog safety verdict.

## The point

The model is the least trusted component in the system. Everything interesting
here is the machinery that keeps it honest:

- **A referee.** Every tool call the model proposes is checked against the
  tool's schema before anything runs. Bad calls bounce back as errors the model
  can read and correct. Nothing else in the stack enforces schemas — we
  measured.
- **An auditor.** Plain code — no AI — refuses to accept a finished plan unless
  every dog's walk interval has a weather check at that dog's location, with
  that dog's temperature band. When it refuses, it prescribes the exact missing
  call. This is the reason a model can't hand you confident, fabricated
  verdicts (one tried; it's in the commit history).
- **A structured finish line.** The model ends by calling `submit_plan`, a tool
  with no implementation whose schema-validated arguments *are* the answer. No
  prose parsing, ever.
- **A qualification gate for models.** The picker on the site is generated, not
  curated: candidates come from the OpenRouter catalog filtered by price and
  capability, and each must complete a real end-to-end run — route, per-dog
  weather, accepted plan — before it may appear. `models.json` is the current
  ledger, rejects and reasons included.

Weather verdicts come from threshold tables in code, not model judgment. The
route order comes from OR-Tools, not vibes. The model's job is deciding what to
call next and narrating the results.

## Architecture, briefly

Hand-rolled agent loop (no framework) over three tools — Nominatim geocoding,
Open-Meteo weather with deterministic verdicts, OpenRouteService + OR-Tools
routing — speaking OpenRouter's API. A thin [MCP](https://modelcontextprotocol.io)
server exposes the same tools to any MCP host. A FastAPI service streams the
agent's events over SSE to a static Astro site; the service runs on a retired
laptop behind Tailscale Funnel for about a dollar a month, with inference
rented per-token.

Details, decisions, and the rebuild changelog: [CLAUDE.md](CLAUDE.md).
API contract and abuse armor: [docs/SERVICE.md](docs/SERVICE.md).
Ops runbook for the laptop: [FOSSIL.md](FOSSIL.md).

## Run it

```bash
uv sync
uv run --with pytest python -m pytest          # 73 offline tests, no network
uv run python -m dog_walker.agent "<request>"  # CLI agent run
uv run uvicorn dog_walker.service:app --port 8010   # the service
uv run python -m dog_walker.qualify            # regenerate the model ledger
```

Secrets go in `.env` (gitignored): `OPENROUTER_API_KEY`, and optionally
`OPENROUTESERVICE_API_KEY` for real street routing.

## History

This repo began in 2024 as a LangChain/ReAct app with a Streamlit UI and a local
Ollama model — a competent tutorial-shaped version of the idea. The 2026 rebuild
replaced every one of those pieces: the framework gave way to a hand-rolled loop,
the ReAct string-parsing to native tool calling, prose output to schema-validated
`submit_plan`, and local inference to rented per-token models. The original is
frozen under [`legacy/`](legacy/) for comparison; nothing current imports it.

## Where it's going

The qualification gate is quietly becoming the point. Turning "which cheap models
can do this?" into a real **measurement** — pass rates with confidence intervals
instead of single-run gates, a taxonomy of *how* models fail, real cost and
latency per completed plan, and a timestamped archive of the model ledger so the
same model drifting over time becomes a finding. The dog-walking task stays, but
grows a library of scenarios (forced weather refusals, infeasible schedules,
constraint conflicts) so a pass rate measures competence, not luck. The thesis
worth publishing: you can only measure an agent honestly when the task has a
deterministic, checkable finish line — which is exactly what the auditor is.