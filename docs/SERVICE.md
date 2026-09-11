# The service: API, event stream, and armor

The FastAPI app in `src/dog_walker/service.py` is the public face of
the agent. This document is the contract: what the API accepts, what
it streams, what protects it, and what each protection does and does
not promise. If you change a limit in code, change it here.

Run locally:

```bash
uv run uvicorn dog_walker.service:app --port 8010
```

Secrets: `OPENROUTER_API_KEY` in the environment or `.env`. Optional
`OPENROUTESERVICE_API_KEY` for real street routing (haversine
fallback without it, honestly flagged). All inference goes through
OpenRouter; the picker allowlist comes from `models.json`, generated
by the qualifier (`python -m dog_walker.qualify` — catalog criteria
plus a live end-to-end gate; see its docstring).

## Endpoints

### `GET /presets`

The demo rosters: `[{id, title, description, pets: [names]}]`.
Anonymous visitors are expected to use these.

### `POST /plan` → SSE stream

Body is **either** a preset **or** a custom roster, never both:

```json
{"preset": "lakeview-classic"}

{"start_address": "Wrigley Field, Chicago",
 "start_time": "13:00",
 "pets": [{"name": "Rex", "address": "5218 N Clark St, Chicago",
           "walk_minutes": 60,
           "buffer_minutes": 10,      // optional, 0-60: pre-walk prep
           "cold_tolerance": 0,       // optional, -3..+3 (husky = +3)
           "heat_tolerance": 0}]}     // optional, -3..+3
```

The response is a server-sent-event stream (`text/event-stream`),
one JSON object per `data:` line. First event is always
`{"event": "accepted", "run_id": ...}`; the agent's own events
follow (vocabulary below); the stream always ends with exactly one
`final` or one `error`.

### `GET /healthz`

`{"ok": true}` liveness probe.

### `GET /info`

`{"model": ..., "backend": ..., "models": [{id, label}]}` — the
default model, plus the allowlist the site's picker renders. Model
identity is never hard-coded in the site.

`POST /plan` accepts an optional `"model"` field validated against
the allowlist (422 otherwise) — the browser can pick, never name an
arbitrary model (cost armor: an open passthrough would spend our
OpenRouter credits on anyone's favorite frontier model).

## The event vocabulary

Produced by `agent.run_events()`; the service relays them verbatim.

| event        | payload                          | meaning |
|--------------|----------------------------------|---------|
| `accepted`   | `run_id`                         | request admitted (service-level) |
| `start`      | `model`, `backend`               | run begins |
| `plan`       | `text`                           | the written plan (from the model's reasoning channel, wherever the backend puts it) |
| `call`       | `round`, `name`, `arguments`     | the model proposes a tool call |
| `bounce`     | `name`, `error`                  | the referee refused it (schema violation); the error went back to the model |
| `result`     | `name`, `result`                 | tool executed; full structured result |
| `audit_veto` | `gap`                            | submit_plan refused: a dog's walk interval lacks a covering weather check |
| `nudge`      | `round`                          | model answered in prose; loop reminded it to finish via submit_plan |
| `final`      | `plan`                           | the validated submit_plan arguments — THE answer |
| `error`      | `message`                        | terminal failure (deadline, round cap, backend down) |

UI rule of thumb: render the trace from `call`/`result`/`bounce`/
`audit_veto` as they arrive; render the answer ONLY from `final.plan`
(never from model prose — it drops details).

## The armor, layer by layer

Each layer states what it defends against and its honest limits.

**1. Structured input only.** Pydantic models reject anything but the
schema: ≤ 6 pets, name ≤ 40 chars, address 4–120 chars, walk minutes
∈ {20, 30, 60}, buffer 0–60, tolerances −3..+3, time `HH:MM`,
preset XOR custom. Free-form text from
the network never reaches the model — the prompt is rendered by
`presets.build_request()` from validated fields, one dull sentence
per fact. This is the main defense against prompt injection and
against the endpoint being farmed as a general chatbot. *Limit:* the
`address` field is still ultimately text a model reads; 120
characters caps how much mischief fits, it doesn't make mischief
impossible.

**2. Per-IP rate limit.** 6 runs per rolling hour per IP, in-memory,
HTTP 429 beyond that. *Limits:* resets on restart; counts IPs, so a
NAT full of students shares one bucket and a botnet has many; if the
service ever runs on >1 process, the buckets fragment. All fine for
a demo — revisit before real multi-user service.

**3. Single-flight queue.** One agent run at a time (semaphore), at
most 3 more waiting; everyone else gets an immediate 429 ("queue
full"). An honest busy-signal beats a mystery hang, and one-at-a-time
keeps worst-case OpenRouter spend and tool-API usage linear in time.
*Limit:* a client that disconnects while queued can leak its waiting
slot until process restart (rare; acceptable for now).

**4. Per-run deadline.** 180 s wall clock, enforced between events;
a stream that exceeds it ends with an `error` event. Individual model
calls are separately bounded by `chat()`'s own 180 s timeout and
3-attempt retry (backoff; retries network faults, 429s, 5xxs — never
4xxs, which are our bugs and should surface). *Limit:* the deadline
fires between events, so the true ceiling is deadline + one call
timeout.

**5. CORS.** Browsers may call only from `https://walker.purr.io`
and the local Astro dev origin. *Limit:* CORS constrains browsers,
not curl; the rate limit and input caps are what constrain curl.

**6. Transcript logging.** Every run writes
`runs/<run_id>.json`: timestamp, caller IP, rendered prompt, every
event, duration. Gitignored. This is the audit trail for abuse
investigation, the debugging record, and raw material for the site
essay. *Privacy note:* custom-mode addresses land in these files;
if the service ever takes real user data, decide retention then.

**7. Upstream quota safety.** Preset runs cost zero Nominatim calls
(coordinates are frozen in `presets.py` and seeded into the geocode
cache at startup). Custom-mode geocoding is bounded by the rate
limit × 12-address batch cap, comfortably inside Nominatim's
1 req/s policy given single-flight. OpenRouteService free tier
(~2k directions/day) exceeds any plausible demo traffic; OpenRouter
spend is capped by prepaid credits (~$0.001/run).

## What is deliberately NOT here yet

* Authentication / API keys — everything is anonymous; the armor
  assumes demo stakes. A "real service" needs keyed access and
  per-key quotas before relaxing any cap above.
* True run cancellation — a disconnected client stops receiving
  events, but the in-flight agent run completes server-side (bounded
  by the deadline). Fine at these durations.
* Multi-process scaling — the limiter and queue are process-local by
  design; scaling out means moving both to shared state (or, more
  likely, deciding the demo doesn't need to scale out).
* Hosting — undecided (fossil is retired from model duty; the
  service itself is lightweight and could run anywhere Python runs).
  See CLAUDE.md Phase 5.
