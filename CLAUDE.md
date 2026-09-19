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
    **2026-09-17 window fix**: a `morning` walk must FINISH by noon, not
    merely start by noon. The old arrival-based bound scheduled 11:35-12:35
    "morning" walks and returned feasible for morning-overbook (a scenario
    whose whole point is infeasibility), so every model that faithfully
    relayed the tool was graded `fabricated_feasible` — 19 runs in the
    2026-09-17 sweep, the largest failure bucket in it.
    **Routing quota**: ORS free quota is per day and a sweep replays the same
    dozen rosters hundreds of times (318 route calls = ~636 ORS requests on
    2026-09-17, which exhausted it: 76 of 81 routes in failure transcripts
    silently fell back to straight-line distances). So real-street answers are
    cached on disk (`data/route_cache.json`, gitignored, keyed by coordinates
    to 5 decimals; fallbacks are never cached), `toolbox.WANT_GEOMETRY=False`
    skips the map-geometry request for anything that draws no map (the
    harness sets it), and every fallback is counted in
    `toolbox.ROUTE_FALLBACKS`. measure.py probes routing once at sweep start
    and REFUSES to run when it is degraded (`--allow-degraded-routes` to
    override); any run whose route fell back is `blame=harness` and ungraded,
    passes included.
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
  - `chat()` speaks ONE dialect: OpenRouter's OpenAI-compatible API (the
    Ollama dialect was retired 2026-09-11 with fossil's inference role;
    if local inference returns, Ollama serves this same dialect at
    /v1/chat/completions). Retries transient failures with EXPONENTIAL
    backoff bounded by a per-run DEADLINE: CHAT_ATTEMPTS=6 at
    CHAT_TIMEOUT_S=120s, backoff 4→8→16→32→60s (CHAT_BACKOFF_BASE_S=4,
    _MAX_S=60), honoring OpenRouter's `Retry-After` header on 429s (their
    documented guidance). run_events passes an absolute deadline
    (AGENT_RUN_BUDGET_S=300) into every chat() call; no read or sleep runs
    past it, so a slow-but-working model gets its full time while a
    chronically-throttled one still can't overrun the budget. CAVEAT
    (verified 2026-09-16): the timeout bounds each socket READ, and
    OpenRouter sends whitespace keep-alive chunks before the JSON body, so
    one slow call is never cut off and can run far past both budgets (the
    deadline is only checked between calls). Latency diagnostics: chat()
    times every HTTP attempt (status, headers_s, total_s, backoff_s,
    generation id); run_events adds tool timings and returns them as
    `timing` on the terminal event; setting agent.CALL_LOG writes a live
    JSONL line before and after each attempt and tool call. The harness
    (measure.py) buckets each run's wall-clock (model, malformed, 429
    throttle, 400 rejected, provider failure, tools, other), pulls
    OpenRouter's free /generation stats after the clock stops (time to
    first token, generation time, reasoning tokens, hidden provider
    failovers), appends each run to measurements/runs-<ts>.jsonl as it
    finishes, and prints a timing report (`--timing-report`, runs >120 s
    flagged). Attribution fixes after the 2026-09-16 overnight sweep:
    (1) the Mac idle-slept once Claude released its sleep assertion, so
    18.8 h of wall time held 5.5 h of work and sleep-interrupted runs hit
    backend_error at 7x the awake rate; measure.py now runs `caffeinate -i -s
    -w <pid>` and records host_sleep_s (wall minus monotonic) per run and
    per attempt. (2) ACT max_tokens was 700; 16/18 malformed_output were
    grand-tour cut-offs. Caps are now 4000 (PLAN_MAX_TOKENS/ACT_MAX_TOKENS),
    truncation is detected from finish_reason OR native_finish_reason
    (OpenRouter normalized cut-offs to "tool_calls"), is not retried, and is
    its own outcome. (3) IncompleteRead (dropped connection) escaped the
    retry loop; now retried. (4) Refused requests keep the HTTP error body.
    (5) Every run gets `blame` (none/model/provider/harness/host, see
    measure.blame); pass rates grade only none+model runs, and
    `provider_excluded` now means "not the model's fault". This
    separates the two knobs the earlier designs conflated: per-CALL timeout
    (bounds one hang) vs whole-RUN budget (gives slow models room). 429/
    5xx/network/unusable-200 retried; a 400 gets one retry without the
    reasoning block; other 4xx raise. Every reply carries `_meta`
    (latency_s, attempts, throttles=429s ridden out, upstream provider);
    run_events accumulates `retries`+`throttles`+`providers` onto the
    terminal event so the harness separates provider throttling/flakiness
    from model competence.
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

**Models**: everything runs through OpenRouter (`OPENROUTER_API_KEY` in
`.env`); `DEFAULT_MODEL` in agent.py. The picker allowlist is GENERATED by
the qualifier (`src/dog_walker/qualify.py`, `python -m dog_walker.qualify`):
OpenRouter catalog filtered by hard criteria (tools + reasoning support,
price caps comparable to the default; router pseudo-models, :variants, and
zero/negative price sentinels excluded), then each candidate must pass a
LIVE end-to-end gate — a preset run reaching validated submit_plan with a
clock-timed route within veto/time budgets. Output: models.json (committed);
the service loads it at startup, falling back to the PINNED pair (default +
Claude Haiku 4.5, the price-cap-exempt frontier contrast). Measured
history: the original five-fixture bake-off chose qwen3:8b; live comparison
2026-09-11 — haiku/llama-70b/gemini-2.5-flash textbook, gemini-2.5-flash-
LITE fabricated verdicts (now structurally impossible), mistral-small died
on upstream 429s.

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
- Tests: `uv run --with pytest python -m pytest` — 121 offline tests, no
  network, no model (scripted-backend fixtures prove the event loop).
  (Bare `uv run pytest` can fail to spawn yet exit 0 — never trust it in a
  `&&` chain.)
- Agent CLI: `uv run python -m dog_walker.agent "<request>"`.
- Service: `uv run uvicorn dog_walker.service:app --port 8010`.
- Measurement harness: `uv run python -m dog_walker.measure [--k N]
  [--models a,b] [--scenarios x,y] [--dry]` — see roadmap below.
- Model-compat check: `uv run python experiments/runner.py <model>`.
- Style: no lambdas — named functions with docstrings.
- Secrets in `.env` (gitignored): `OPENROUTER_API_KEY`,
  `OPENROUTESERVICE_API_KEY`. Never in code or committed files.

## 5. Roadmap / open questions

- **Harder constraints for model differentiation: DONE 2026-09-15.**
  Motivated by the k=2 sweep finding that a clean harness makes the six
  models near-indistinguishable (all ~100%) -- to see capability drift you
  need a longer horizon (METR 2503.14499) and reliability scoring (tau-bench
  pass^k, 2406.12045). Added three constraints, each the usual attribute ->
  effect -> oracle:
  - **Daylight.** Workday is now 7am-7pm (WORKDAY_START/END_MIN; morning
    starts 7:00). New tool `check_daylight` (Open-Meteo daily sunset).
    optimize_route takes `sunset_min` and HARD-caps every walk to end by
    min(7pm, sunset) -> feasible:false on short winter days. `audit_daylight`
    forces the model to call check_daylight + pass sunset when any dog has an
    afternoon window ('any' dogs get scheduled early, so they can't run late).
  - **Difficulty (0-5).** Each point = 5 min; ~50% of walks the dog acts up
    and runs difficulty*5 longer, ROLLED in optimize_route (like weather:
    realized, not planned; `random.random()`), threaded into the solver
    service AND the timeline AND the daylight bound. Can push a later walk
    past a window/sunset -> infeasible. Stochastic, so harness runs aren't
    bit-reproducible.
  - **skip_rain.** check_weather returns `raining` (>= RAIN_TRIGGER_MM 0.2);
    a skip_rain dog in rain must get a MINIMAL_VISIT_MIN (10) visit not a
    walk -- `audit_rain` enforces it from the weather RESULT correlated to
    its call by tool_call_id. Relaxes the schedule (shorter), so it's a
    weather->plan branch, not an infeasibility driver.
  - MAX_STOPS 10->13, service MAX_PETS 6->12. Site form: difficulty slider
    (default 0, self-explaining label) + skips-rain checkbox. build_request
    + Pet model carry both. New `grand-tour` scenario (6 dogs, every
    constraint) with expected_feasible=None -> harness grades on
    RELAY-FIDELITY (a validated final IS the pass, since the auditors already
    forced correctness). Live: grand-tour is a 16+ tool-call chain ling
    completes ~2 of 3 -- the run-to-run variance we were after.
  - STILL OPEN: pass^k metric in the report; a rain/difficulty live check
    (need a wet day / a bad roll); a full high-k re-sweep once the task
    actually differentiates.
- **Measurement harness: DONE 2026-09-14 — the actual deliverable.**
  `src/dog_walker/measure.py` + `scenarios.py`. Evolves the single-run
  gate (qualify.py) into a real measurement: every model × the scenario
  library × k runs. Reports a PASS RATE with a **Wilson score interval**
  (5/5 ≠ certainty), real **cost/latency/rounds/tokens per plan** (summed
  from OpenRouter usage — `chat()` now sends `usage:{include:true}` and
  `run_events` emits `usage`+`rounds` on the terminal event), and a
  deterministic **failure taxonomy** from the event stream (veto_livelock,
  schema_thrash, prose_stall, fabricated_feasible, false_infeasible,
  backend_error, timeout) with the FULL transcript of every failure kept.
  Grade is deterministic: pass iff a validated submit_plan's `feasible`
  matches the scenario's known answer (7 scenarios, all ground-truth-
  checked against the solver). No LLM judge — the auditor IS the
  measurement. Archives to `measurements/measure-<ts>.json` (gitignored),
  timestamped from day one so model drift becomes a finding. Scenarios
  reuse frozen preset coords → zero live geocoding; only inference varies.
  FIRST FINDING that fed back into the system: qwen3-8b sets optional
  fields to `null` and livelocked 14 rounds on `None is not of type
  number`; fixed with `without_nulls()` in the referee (null == unset,
  matches how tools already read inputs) — single-easy 0/2 → 2/2, ~4×
  cheaper. **Provider-vs-competence split (2026-09-16):** a k=5 sweep was
  contaminated by one provider serving qwen3.7 hung ($0-billed) calls; the
  old 4×180s retry amplified each stall to 400-700s. Hardened: chat()
  timeout 180→60s, attempts 4→3, RUN_DEADLINE 240→90s; every reply now
  carries `_meta` and the harness records per-run `retries`/`providers`.
  `PROVIDER_FAULTS={timeout,backend_error}` are EXCLUDED from the pass-rate
  denominator and cost/latency medians (reported separately as
  `provider_excluded`); `malformed_output` stays IN (that's the model).
  Flakiness (`total_retries`/`flaky_runs`/`total_throttles`) counted across
  all runs, passes included. The sweep loop now INTERLEAVES models
  round-robin (pass→scenario→model) instead of model-at-a-time, so a bad
  provider window can't be misread as one model's weakness.
  **CORRECTION (2026-09-16, first k=5 sweep):** the "provider fault"
  label was over-broad. Of 63 excluded runs, ~23 were HTTP 429 throttling
  (real, external, concentrated on qwen3.7) but ~37 were my own 90s
  RUN_DEADLINE guillotining slow-but-WORKING gpt-oss/qwen3-8b runs (their
  successes ran to ~88s) -- a harness artifact, the same "confound model
  speed with provider health" error inverted. Fix: RUN_DEADLINE 90→330s
  (above AGENT_RUN_BUDGET 300), chat() exponential backoff + Retry-After
  (see §2). That first sweep's numbers are NOT publishable: qwen3.7 (n=16),
  qwen3-8b (n=19), gpt-oss (n=23) had denominators thinned to
  incomparability. What survived the confound (within-model, robust):
  grand-tour breaks everyone (malformed_output on the long chain);
  morning-overbook (infeasible) draws fabricated_feasible across models,
  haiku worst at 0/5; corrections concentrate at submit_plan (29-60%),
  ~0% on lookups. Still open: the fault-tolerant re-sweep to publish;
  repeat at 2-3 times of day to characterize flakiness; re-qualify
  models.json.
- **Hosting: DONE 2026-09-11 — fossil + Tailscale Funnel.** The service
  (not the model) runs on fossil as `dogwalker.service`, public at
  **https://fossil.taild72aca.ts.net** (Funnel → 127.0.0.1:8010).
  Verified: unattended reboot to healthy public API in ~30 s. Ops runbook:
  **FOSSIL.md**. Inference stays on OpenRouter. Frontend must call that
  URL; CORS already pins walker.purr.io.
- **Frontend: LIVE at https://walker.purr.io (2026-09-11).** Astro on
  Cloudflare Pages (project agentic-dog-walker; classic Pages — the
  Workers-flavored tooling once injected a server adapter, reverted),
  Route 53 CNAME + custom domain attached, TLS verified, CORS confirmed
  for the final origin. Known quirk: on tailnet machines Chrome's Local
  Network Access blocks the page's API fetches (MagicDNS resolves fossil
  privately) — public visitors unaffected; see FOSSIL.md troubleshooting.
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
- **Buffer time: DONE 2026-09-11.** Per-pet `buffer_minutes` (0-60): prep
  spent BEFORE the walk — delays walk_start, lengthens the schedule, does
  not extend the dog's outdoor interval. Flows Pet → build_request →
  optimize_route stops → timeline (`buffer_minutes` on the entry).
- **Dog hardiness: COMFORT BANDS (2026-09-12, superseding the two-score
  tolerances).** Per-dog `comfort_min_f`/`comfort_max_f` — the dog's
  comfortable feels-like range, a dual-thumb slider on a blue→red
  gradient in the UI (default [20, 84], bounds [-20, 110], ≥10F wide;
  the tool forgives inverted/narrow bands, the front door 422s them).
  Verdict = distance beyond either edge, one rung per BEYOND_STEP_F=10F
  (this replaced the separate cold/heat ladders entirely; wind/precip
  ladders and the ABSOLUTE wet-cold trigger remain). Band position =
  husky-vs-iggy axis, band width = hardy-vs-bulldog axis — expresses the
  narrow-band bulldog the single-axis design couldn't. Bands echo through
  the timeline and the auditor requires each dog's weather check to carry
  that dog's band. Preset exemplars: Daisy [45,95], Wilbur [-10,70].
  Verified live: same ~75F afternoon, Wilbur CAUTION while others OK.
- **Meds + terrain + walk windows: DONE 2026-09-14.** Three more per-dog
  attributes, each attribute → schedule/verdict effect → oracle rule:
  - `needs_meds` (boolean): meds are given on arrival, adding
    `MED_HANDLING_MIN=10` handling time before the walk. Not a timed
    deadline — walkers don't administer meds at a set hour (design
    corrected 2026-09-13).
  - `max_relief_m` (hill tolerance): a labeled slider on the site
    (flat only / gentle slopes / hilly OK / any terrain). Set → requires
    a `check_terrain` call at that dog's location (Open-Meteo elevation
    on a 3×3 grid, relief = max−min); `audit_terrain_coverage` vetoes
    submit_plan without it. Advisory (walks are abstract loops, no path).
  - `walk_window` (any / morning / afternoon): how walkers really
    schedule. **The walker chooses the departure** — start times are
    flexible, so `optimize_route` solves for a departure that clusters
    every walk into ONE outing across the noon boundary (OR-Tools time
    dimension, absolute minutes, free start cumul, slack 0 = no
    loitering, finalizer minimizes departure). Morning walk starts
    8:00am–noon (`MORNING_START_MIN`), afternoon at/after noon
    (`NOON_MIN`). A lone afternoon dog starts at noon; one AM + one PM
    become a single 11-ish-to-1-ish trip. A caller `start_time` is
    IGNORED once any window is set; the solved start comes back in the
    result. No (departure, order) that fits every window → `feasible:
    false` (real infeasibility, e.g. 4× 60-min morning walks across
    Chicago). `audit_feasibility` vetoes BOTH directions (rosy-over-
    infeasible and false-alarm). Superseded the earlier fixed-start
    model that computed forward from a required start_time.
- **Model picker (DONE 2026-09-11)**: service MODELS allowlist (cost
  armor — browser picks from the list, never names arbitrary models),
  optional `model` on /plan threaded through run_events, /info returns the
  list, site renders a selector. Lineup: qwen3-8b (default), Claude Haiku
  4.5, Gemini 2.5 Flash Lite, Llama 3.3 70B, Mistral Small 3.2 — verify
  slugs/prices on OpenRouter when touching. chat() retries a 400 once
  without the reasoning block (some models reject it).
