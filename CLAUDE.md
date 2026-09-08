# agentic_dog_walker — Agent guide (READ FIRST)

## 0. How we work here — the collaboration contract (MOST IMPORTANT)

**This is a LEARNING project. The deliverable is Bryan's understanding of modern LLM
tool-use design — custom agent loops, native tool calling, and MCP — NOT finished code
shipped fast.** This overrides the default "autonomous implementer" mode.

- **Explain before doing.** Before writing or changing code, explain the concept, the
  protocol shape (what JSON actually crosses the wire), and the design trade-off in play.
- **One phase at a time.** Do a single concept/phase, then stop and check in. Bryan
  drives the pace.
- **Bryan writes the code that carries the learning** — the agent loop (plan/act/reflect),
  tool-call parsing/validation, the MCP wiring. I scaffold, explain, and review. When I do
  write code, walk through it rather than handing over a finished block.
- **Best practices are an explicit goal** — surface and explain structure/typing/testing
  decisions as we hit them.
- **Teacher's register, permanently (Bryan, 2026-09-06, after repeated corrections).**
  Every acronym/term of art gets a plain-language definition at first use — every time,
  no "obvious" exceptions. Plain concept BEFORE the term (the name is the footnote).
  Concrete example before abstraction. If a sentence needs a glossary, rewrite it.
  The temptation doubles when defending a position — that's exactly when to write plainer.
- Bryan's background: strong data engineering, solid Python; did the original LangChain
  version of this repo (2024) and the well-spacing PyTorch project (2026). Wants the
  modern replacement for what LangChain hid from him.

## 1. What this project is becoming (decided 2026-09-04)

Rebuild of the 2024 LangChain dog-walker as a **public demonstration of modern LLM
tool use**, served at **walker.purr.io**, powered by a local model on Bryan's own
hardware — no subscription APIs anywhere.

Architecture (three pieces):

- **fossil** (Dell Latitude 5430, headless Debian 13, on the tailnet at 100.71.229.15,
  hostname `fossil`): runs Ollama (bound to 100.71.229.15:11434, tailnet-only —
  deliberate; nothing listens on localhost) and will run the agent as a FastAPI service.
  Passwordless sudo for user bryan; lid-ignore + sleep masked; unattended-upgrades on.
- **Tailscale Funnel** exposes ONLY the agent's API publicly (structured plan-walk
  requests only — never free-form prompts; rate-limited; single-flight queue; ≤6 pets).
- **walker.purr.io**: static Astro page on Cloudflare Pages (Route 53 CNAME, Bryan
  clicks the custom-domain step), purr.io family style. Streams the agent's
  plan/act/reflect trace over SSE; renders the route with MapLibre from GeoJSON.

The agent itself: **hand-rolled loop** (no LangChain) doing plan → act → reflect, using
Ollama's native tool calling (`/api/chat` with `tools`).

**Tool architecture: FACADE design (decided 2026-09-05, after reviewing MCP-in-production
criticism).** Tools (geocode/Nominatim, weather/Open-Meteo with deterministic safety
flags, TSP routing/OR-Tools + OpenRouteService) live as a plain Python module — typed
functions + JSON schemas — that the agent dispatches IN-PROCESS (we own both ends of the
wire; MCP-client plumbing for our own local tools would be cargo-culting). A thin **MCP
server facade** (official SDK / FastMCP) wraps the SAME functions as a separate entry
point, so any MCP host (Claude Desktop etc.) can plug in — the interop boundary is what
MCP is actually for, and the site writeup says exactly that. Folium is dropped; the
route tool returns GeoJSON for the browser.

## 2. Model facts (measured on fossil, 2026-09-04)

- qwen2.5:7b Q4: **~5.2 tok/s generation** (memory-bandwidth-bound; single-channel
  DDR4 — a second 16GB SODIMM would roughly double it), prompt eval **~33 tok/s at
  `num_thread: 10`** (the sweet spot — 12 threads is worse; always pass num_thread 10).
- No thermal throttling under sustained load.
- Design consequences: terse system prompts, lean tool schemas, short outputs, rely on
  Ollama KV cache. Phase 1 bake-off vs qwen3:8b (thinking mode) decides the model.

## 3. Phases

1. ✅ Model bake-off DONE 2026-09-05 (commit 48d7be2): **qwen3:8b selected, 5/5**
   (qwen2.5:3b 4/5 fails relative dates; qwen2.5:7b 3/5 defers/asks permission).
   Fixtures in experiments/ double as a regression suite (~2 min) — rerun after any
   system-prompt change. Prose checks are triage-only (see fixtures.py docstring).
2. ✅ Tools DONE 2026-09-08: toolbox.py (check_weather with Bryan's verdict
   ladders + escalations, geocode_addresses batched/cached/polite, optimize_route
   with timeline + GeoJSON) + mcp_server.py facade (mcp 2.x SDK: MCPServer, the
   class tutorials still call FastMCP; verified over stdio with a real client).
   28 offline tests. Threshold citations still TODO(Bryan).
   **Walk-duration model (Bryan, 2026-09-07):** each dog's walk is 20/30/60 min
   (WALK_DURATIONS enum), taken as a solo loop from its own home — no group walks,
   ever solo. So visiting order stays pure geography (TSP over transit), and
   durations drive the TIMELINE (optimize_route returns per-dog walk intervals).
   ROADMAP: per-stop weather checks against those intervals (agent-side, Phase 3);
   pet time-window constraints via OR-Tools time dimension (future).
3. ✅ Agent loop DONE 2026-09-08 (src/dog_walker/agent.py). Mode change: Bryan
   chose to STUDY a completed implementation and annotate it rather than write it
   ("I would rather study a completed example and annotate it than guess") — his
   annotation pass is the remaining learning step. Live-verified full cycle:
   plan (think-on) → act with validate-before-dispatch bounces → submit_plan
   vetoed by the deterministic auditor → model made the PRESCRIBED missing
   weather call → accepted, structured per-dog verdicts out. Two bugs found by
   tests/live-run and fixed with comments telling the story: _NEAR_DEG 0.03→0.01
   (start-location check "covered" a dog 3 km away) and vague audit messages
   livelock (now prescribe exact call args).
   **Phase-3 syllabus, extracted from live lesson-2 traces (2026-09-07/08):**
   (a) validate-before-dispatch with bounce-and-retry (jsonschema against REGISTRY
   schemas; lesson 2 dispatches raw). (b) The reflect loop-back: walk intervals
   only exist AFTER optimize_route returns the timeline, so weather must be
   re-checked per dog's actual interval/location afterward — an information-
   dependency no upfront planning fixes (think-on run checked per-dog locations
   spontaneously but still used the 13:00 departure hour for a 16:09 walk).
   (c) Reasoning mode measurably improves orchestration (think-off: one global
   weather check; think-on: per-dog checks, parallel calls in one round) → use
   think for the plan step, not for mechanical calls. (d) UI takes timelines from
   tool output, never from the model's retelling (it drops details).
4. FastAPI + SSE + hardening (queue, rate limit, input caps).
5. Funnel + Astro frontend + walker.purr.io CNAME.
6. README/essay pass; the site explains its own architecture (including the
   facade rationale: native calling where we own both ends, MCP at the boundary).

## 4. Legacy code (the 2024 LangChain version)

`src/dog_walker/` is the old implementation — keep as reference; port tool logic out of
`tools/` (geocoding.py, weather.py, route_optimizer.py). agent.py (ReAct via
`OllamaLLM`) and mapping.py (Folium) are being replaced outright. Old commands in git
history if needed. OPENROUTESERVICE_API_KEY still comes from `.env` (free key).

## 5. Environment

- uv project, Python 3.12. `uv sync`. Tests: `uv run --with pytest python -m pytest`
  (bare `uv run pytest` can fail to spawn yet exit 0 — never trust it in a && chain).
- Dev happens on ichabod (Mac) or pepper (Arch); deploy target is fossil via
  `ssh bryan@fossil` (keys in place from both).
