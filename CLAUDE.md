# agentic_dog_walker — Agent guide (READ FIRST)

## 1. What this project is

A dog-walking route planner that demonstrates modern LLM tool use, headed for
public hosting at **walker.purr.io**. Given a start point and a set of pets
(each with an address and a 20/30/60-minute walk), it geocodes, solves the
visiting order over real street distances, checks weather per dog's actual
walk interval, and returns a structured plan with per-dog safety verdicts.

The 2024 original was a LangChain/ReAct application: prompt-format tool calls
regex-parsed from model prose, single-string tool inputs, judgment embedded in
prompts, Folium maps rendered server-side, Streamlit UI, qwen2.5:14b via local
Ollama. It worked, brittlely. The 2026 rebuild replaces every one of those
mechanisms; the old implementation is frozen under `legacy/` for reference and
nothing imports it.

## 2. Architecture (current)

**LLM sandwich:** language at the boundaries, deterministic code in the middle.
Everything with an exact answer (route order, safety thresholds, schema
validation) is plain code; the model translates intent and narrates results.

- **Toolbox** (`src/dog_walker/toolbox.py`) — three typed functions returning
  JSON-able dicts, each with a JSON Schema, joined in `REGISTRY`:
  - `check_weather` — Open-Meteo hourly forecast (Fahrenheit, served by the
    API via `temperature_unit`), assessed over the requested walk window only.
    Verdicts (`OK/CAUTION/SHORTEN/DO_NOT_WALK`) come from threshold ladders +
    combo escalations (e.g. wet-cold bump) in code — the model relays
    verdicts, never derives them. Ladder rungs are policy; changing one trips
    tests by design. (Citations for the chosen thresholds: still TODO.)
  - `geocode_addresses` — Nominatim, batched (one call per roster), cached,
    rate-spaced 1.1 s with identifying User-Agent per usage policy, 12-address
    cap. Failures return per-address error slots, never raise.
  - `optimize_route` — OpenRouteService walking-distance matrix + OR-Tools
    Traveling-Salesman solve (round trip from stop 0) + street geometry as
    GeoJSON for the browser. Haversine fallback with an honest
    `uses_real_streets` flag. Returns a timeline with each dog's walk
    interval — the input to per-stop weather checks.
- **Agent** (`src/dog_walker/agent.py`) — a hand-rolled loop, no framework:
  - `run_events(request)` generator yields one event per observable moment
    (`start, plan, call, bounce, result, audit_veto, nudge, final, error`;
    the stream ends with exactly one `final` or `error`). `run()` is a thin
    CLI consumer of the same stream.
  - **plan**: one reasoning-mode round; the reply (wherever the backend puts
    it — `reasoning`/`thinking`/`content`) joins the message state.
  - **act**: every proposed call is validated against its REGISTRY schema
    (`validate_call`, jsonschema) BEFORE dispatch; violations bounce back as
    tool-result errors written as prompts the model can act on. Execution
    failures likewise become error results (`dispatch`).
  - **reflect**: `audit_weather_coverage` — deterministic code that reads the
    conversation, extracts each dog's walk interval from the route timeline,
    and vetoes `submit_plan` unless a weather check covers that interval
    within ~1 km (`_NEAR_DEG = 0.01`), PRESCRIBING the exact missing call
    (vague veto messages livelocked the model; prescriptive ones heal in one
    round).
  - **finish**: the model ends by calling `submit_plan`, a tool with no
    implementation whose validated arguments ARE the structured answer.
    Downstream consumers render from `final.plan`, never from model prose.
  - `chat()` normalizes two wire dialects (Ollama `/api/chat` vs
    OpenAI-compatible `/chat/completions`: dict vs JSON-string arguments,
    `tool_call_id`, `choices[0]` wrapping) and retries transient failures
    (3 attempts, backoff; 429/5xx/network retried, 4xx raised).
- **MCP facade** (`src/dog_walker/mcp_server.py`) — ~12 lines exposing the
  same REGISTRY over the Model Context Protocol (stdio) so external hosts can
  plug the tools in. The agent itself dispatches in-process — protocol at the
  interop boundary only. SDK note: mcp 2.x renamed FastMCP → MCPServer.
- **Service** (`src/dog_walker/service.py`) — FastAPI: `GET /presets`,
  `POST /plan` (SSE stream of the event vocabulary), `GET /healthz`.
  Armor: structured input only (pydantic caps: ≤6 pets, field lengths,
  walk-minutes enum, preset XOR custom), per-IP rate limit (6/h),
  single-flight queue (+3 waiting, then 429), 180 s run deadline, CORS pinned
  to the site, per-run transcripts in `runs/` (gitignored). Full contract and
  the honest limits of each layer: **`docs/SERVICE.md`** — keep it in sync.
- **Presets** (`src/dog_walker/presets.py`) — anonymous visitors get curated
  rosters with coordinates frozen from live lookups and seeded into the
  geocode cache at startup: preset runs cost zero Nominatim calls. Custom
  mode is structured-form only; free text never reaches the model from the
  network. `build_request()` renders validated fields into the prompt.

**Model backends** (`BACKENDS` in agent.py): default is OpenRouter
(`qwen/qwen3-8b`, `OPENROUTER_API_KEY` in `.env`); `LOCAL_LLM=fossil` selects
the local Ollama box. qwen3:8b was selected by a five-fixture bake-off (5/5;
qwen2.5:3b fails relative dates, qwen2.5:7b asks permission instead of
calling); the fixtures in `experiments/` remain a ~2-minute regression suite
to rerun after any system-prompt change and against any candidate model.

### fossil (the local inference box)

Dell Latitude 5430 (i5-1245U, 16 GB single-channel DDR4-3200, 256 GB NVMe),
headless Debian 13, hostname `fossil`, tailnet address 100.71.229.15.
Provisioned 2026-09 as the project's inference server; **effectively retired
2026-09-10** (OpenRouter default: rented does more in seconds than local
thinking does in minutes, and electricity likely exceeds token cost) — but
the plumbing and this documentation stay intact for model experiments.

- Setup: passwordless sudo (user `bryan`), lid-switch ignored + sleep/
  suspend/hibernate targets masked (closes like a laptop, runs like a
  server), unattended security upgrades. SSH keys in place from the other
  tailnet machines.
- Ollama bound **tailnet-only** (`OLLAMA_HOST=100.71.229.15` via systemd
  override) — nothing listens on localhost, so even on-box CLI use needs
  `OLLAMA_HOST=100.71.229.15`. Models pulled: qwen2.5:7b, qwen2.5:3b,
  qwen3:8b.
- Measured performance (qwen2.5:7b Q4, 2026-09-04): **~5.2 tok/s
  generation** — memory-bandwidth-bound, so thread count barely moves it —
  and prompt ingestion **~33 tok/s at `num_thread: 10`**, the measured sweet
  spot (12 threads is *worse*: hyperthread contention past the 10 physical
  cores). That measurement is why `chat()` pins `num_thread: 10` for the
  Ollama dialect. No thermal throttling under sustained load.
- Known cheap upgrade if inference is ever revived: the second SODIMM slot
  is empty; 16 GB more (~$30) doubles memory bandwidth ≈ doubles tok/s.
- **Un-retired for SERVICE duty 2026-09-11**: fossil hosts the FastAPI
  service behind Tailscale Funnel (idle orchestrator wattage ~$1/mo beats
  any VPS). Inference stays on OpenRouter; Ollama stays parked.

## 3. How it got here (rebuild changelog, 2026-09)

1. **Native tool calling replaced ReAct.** Measured findings that shaped
   everything after: schemas steer but nothing enforces them (models invent
   and omit arguments); a required field the context can't fill produces
   confident garbage (models have no clock — inject today's date per
   request); numeric fidelity tool→prose is good; severity judgment from raw
   numbers is brittle. Hence: validation layer, date injection, judgment
   moved into tools as deterministic flags.
2. **Model bake-off** on five scripted fixtures (call structure, relative
   dates, over-eager-call trap, tool choice, safety-flag relay) →
   qwen3:8b. The first eval run mostly found instrument bugs; prose substring
   checks are triage-only (see fixtures.py docstring) — durable grading uses
   structured outputs.
3. **Toolbox port** from `legacy/dog_walker_2024/tools/`: typed signatures
   replaced single-string inputs; prose outputs became structured dicts;
   whole-day weather averaging became windowed assessment; verdict tiers and
   escalations added; Folium dropped for GeoJSON; walk durations became the
   20/30/60 enum with the timeline model (each dog's walk is a solo loop from
   its own home, so visiting order is pure geography and durations shape the
   schedule; no group walks).
4. **Multi-backend chat** with dialect normalization; OpenRouter made the
   default (local thinking runs took minutes; rented does more in seconds;
   electricity > token cost).
5. **Agent loop** with plan/act/reflect as above. Two bugs found by tests and
   live runs, kept as comments where they happened: the auditor's location
   tolerance (a start-point check "covered" a dog 3 km away) and the
   vague-veto livelock.
6. **MCP facade** after deciding against agent-as-MCP-client (owning both
   ends of a local wire makes protocol plumbing cargo-culting; the facade
   serves the actual interop case).
7. **Event-stream refactor + service armor + presets** (Phase 4), for the
   public site.

## 4. Environment & commands

- uv project, Python 3.12. `uv sync`.
- Tests: `uv run --with pytest python -m pytest` — 53 offline tests, no
  network, no model (scripted-backend fixtures prove the event loop).
  (Bare `uv run pytest` can fail to spawn yet exit 0 — never trust it in a
  `&&` chain.)
- Agent CLI: `uv run python -m dog_walker.agent "<request>"`.
- Service: `uv run uvicorn dog_walker.service:app --port 8010`.
- Model-compat check: `uv run python experiments/runner.py <model>`.
- Style: no lambdas — named functions with docstrings.
- Secrets in `.env` (gitignored): `OPENROUTER_API_KEY`,
  `OPENROUTESERVICE_API_KEY`. Never in code or committed files.

## 5. Roadmap / open questions

- **Hosting: DONE 2026-09-11 — fossil + Tailscale Funnel.** The service
  (not the model) runs on fossil as `dogwalker.service`, public at
  **https://fossil.taild72aca.ts.net** (Funnel → 127.0.0.1:8010).
  Verified: unattended reboot to healthy public API in ~30 s. Ops runbook:
  **FOSSIL.md**. Inference stays on OpenRouter. Frontend must call that
  URL; CORS already pins walker.purr.io.
- **Frontend**: Astro page on Cloudflare Pages at walker.purr.io (Route 53
  CNAME; custom-domain step happens in the Cloudflare dashboard), purr.io
  family style; renders the SSE trace live and the route from `final.plan` +
  GeoJSON.
- **README/essay pass**: the site explains its own architecture, including
  the facade rationale and the sandwich boundary.
- **Candidate experiment**: code-execution vs schema-calling bake-off — same
  toolbox, two idioms (validated per-round JSON calls vs one model-written
  program run in a sandbox), measured on rounds/tokens/wall-clock/failure
  modes at qwen3:8b and a frontier model. Expected: the reflect problem
  dissolves in code (a program derives weather windows from the timeline
  itself); small models fail wholesale where the loop fails one bounce at a
  time; token savings modest at 3 tools. Needs a sandbox decision before
  anything public.
- Threshold citations for the weather ladders.
