// walker.purr.io front end.
//
// Two rendering rules, both earned upstream:
//   * the TRACE renders from events as they stream
//   * the ANSWER renders only from structured data: final.plan for
//     verdicts/advice, and the optimize_route call+result (captured
//     from the stream) for stops, timeline, and map geometry --
//     never from model prose, which drops details.

import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

const API = "https://fossil.taild72aca.ts.net";

// On machines that are ON the tailnet (the author's, mostly), MagicDNS
// resolves the API host to a private (CGNAT) address, and Chrome's
// Local Network Access policy blocks public-site fetches to private
// addresses unless the fetch opts in -- which triggers a one-time
// permission prompt. Plain visitors resolve the public Funnel ingress
// and never hit this. So: try the normal fetch; on failure, retry
// with the opt-in (unknown options are ignored by other browsers).
async function apiFetch(path, opts = {}) {
  try {
    return await fetch(`${API}${path}`, opts);
  } catch (err) {
    return await fetch(`${API}${path}`, {
      ...opts,
      targetAddressSpace: "private",
    });
  }
}

const el = {
  presets: document.getElementById("preset-cards"),
  form: document.getElementById("custom-form"),
  petRows: document.getElementById("pet-rows"),
  addPet: document.getElementById("add-pet"),
  run: document.getElementById("run"),
  runTitle: document.getElementById("run-title"),
  trace: document.getElementById("trace"),
  result: document.getElementById("result"),
  verdicts: document.getElementById("verdicts"),
  timeline: document.getElementById("timeline"),
  advice: document.getElementById("advice"),
  pill: document.getElementById("status-pill"),
  reset: document.getElementById("reset"),
  modelSelect: document.getElementById("model-select"),
};

function chosenModel() {
  return el.modelSelect.value || undefined;
}

// ---------------------------------------------------------------------
// reset: clear the run and result, back to the launcher
// ---------------------------------------------------------------------

function resetAll() {
  el.run.hidden = true;
  el.result.hidden = true;
  el.reset.hidden = true;
  el.trace.innerHTML = "";
  el.verdicts.innerHTML = "";
  el.timeline.innerHTML = "";
  el.advice.textContent = "";
  routeCall = routeResult = null;
  if (map) {
    map.remove();
    map = null;
  }
  document.getElementById("launcher").scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}

el.reset.addEventListener("click", resetAll);

// ---------------------------------------------------------------------
// presets
// ---------------------------------------------------------------------

async function loadPresets() {
  try {
    const list = await apiFetch("/presets").then((r) => r.json());
    el.presets.innerHTML = "";
    for (const p of list) {
      const card = document.createElement("div");
      card.className = "card";
      card.innerHTML = `
        <h3>${p.title}</h3>
        <p>${p.description}</p>
        <p class="muted small">dogs: ${p.pets.join(", ")}</p>
        <button data-preset="${p.id}">Run this demo</button>`;
      card.querySelector("button").addEventListener("click", startPreset);
      el.presets.appendChild(card);
    }
  } catch {
    el.presets.innerHTML =
      '<p class="muted">The planning service is unreachable right now ' +
      "(it lives on a laptop in a closet; even closets have bad days). " +
      "Try again in a minute.</p>";
  }
}

function startPreset(e) {
  runPlan({ preset: e.target.dataset.preset, model: chosenModel() });
}

// ---------------------------------------------------------------------
// custom form (structured only -- mirrors the service's schema)
// ---------------------------------------------------------------------

function petRow() {
  const row = document.createElement("div");
  row.className = "petrow";
  row.innerHTML = `
    <label>Dog <input name="pet_name" required maxlength="40" placeholder="Rex" /></label>
    <label>Address <input name="pet_address" required minlength="4" maxlength="120"
      placeholder="5218 N Clark St, Chicago" size="28" /></label>
    <label>Walk
      <select name="pet_minutes">
        <option value="20">20 min</option>
        <option value="30" selected>30 min</option>
        <option value="60">60 min</option>
      </select>
    </label>
    <button type="button" title="remove">&times;</button>`;
  row.querySelector("button").addEventListener("click", removePet);
  return row;
}

function removePet(e) {
  e.target.closest(".petrow").remove();
}

function addPet() {
  if (el.petRows.children.length < 6) el.petRows.appendChild(petRow());
}

el.addPet.addEventListener("click", addPet);
addPet(); // start with one row

el.form.addEventListener("submit", submitCustom);

function submitCustom(e) {
  e.preventDefault();
  const data = new FormData(el.form);
  const pets = [];
  const names = data.getAll("pet_name");
  const addresses = data.getAll("pet_address");
  const minutes = data.getAll("pet_minutes");
  for (let i = 0; i < names.length; i++) {
    pets.push({
      name: names[i],
      address: addresses[i],
      walk_minutes: parseInt(minutes[i], 10),
    });
  }
  runPlan({
    start_address: data.get("start_address"),
    start_time: data.get("start_time"),
    pets,
    model: chosenModel(),
  });
}

// ---------------------------------------------------------------------
// the run: POST + hand-parsed SSE (EventSource is GET-only)
// ---------------------------------------------------------------------

let routeCall = null;   // optimize_route arguments (stop coords)
let routeResult = null; // optimize_route result (timeline, geometry)

async function runPlan(body) {
  document.querySelectorAll("#launcher button").forEach((b) => (b.disabled = true));
  el.run.hidden = false;
  el.result.hidden = true;
  el.trace.innerHTML = "";
  el.runTitle.textContent = "Planning…";
  routeCall = routeResult = null;
  el.run.scrollIntoView({ behavior: "smooth", block: "start" });

  try {
    const resp = await apiFetch("/plan", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (resp.status === 429) {
      const detail = (await resp.json()).detail || "busy";
      traceLine("t-error", `The service says: ${detail}`);
      return;
    }
    if (!resp.ok) {
      traceLine("t-error", `Request rejected (${resp.status}).`);
      return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sep;
      while ((sep = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        for (const line of chunk.split("\n")) {
          if (line.startsWith("data: ")) handleEvent(JSON.parse(line.slice(6)));
        }
      }
    }
  } catch (err) {
    traceLine("t-error", `Stream broke: ${err.message}`);
  } finally {
    document.querySelectorAll("#launcher button").forEach((b) => (b.disabled = false));
    el.reset.hidden = false;
  }
}

function traceLine(cls, html) {
  const div = document.createElement("div");
  div.className = cls;
  div.innerHTML = html;
  el.trace.appendChild(div);
  el.trace.scrollTop = el.trace.scrollHeight;
}

const FRIENDLY = {
  geocode_addresses: "looking up addresses",
  optimize_route: "optimizing the route",
  check_weather: "checking weather",
  submit_plan: "submitting the plan",
};

function handleEvent(ev) {
  switch (ev.event) {
    case "accepted":
      traceLine("t-note", `run ${ev.run_id} accepted`);
      break;
    case "start":
      el.pill.textContent = `${ev.model} · live`;
      break;
    case "plan":
      traceLine("t-note", "the model wrote a plan:");
      traceLine("t-plan", escapeHtml(ev.text));
      break;
    case "call": {
      const label = FRIENDLY[ev.name] || ev.name;
      if (ev.name === "optimize_route") routeCall = ev.arguments;
      traceLine(
        "t-call",
        `<span class="name">${label}</span> ${escapeHtml(brief(ev.arguments))}`
      );
      break;
    }
    case "result":
      if (ev.name === "optimize_route" && ev.result && ev.result.timeline) {
        routeResult = ev.result;
      }
      traceLine("t-result", `→ ${escapeHtml(brief(ev.result))}`);
      break;
    case "bounce":
      traceLine("t-bounce", `referee bounced it: ${escapeHtml(ev.error)}`);
      break;
    case "audit_veto":
      traceLine("t-veto", `auditor veto: ${escapeHtml(ev.gap)}`);
      break;
    case "nudge":
      traceLine("t-note", "(nudged: finish with submit_plan)");
      break;
    case "final":
      el.runTitle.textContent = "Done";
      traceLine("t-note", "plan accepted ✓");
      renderResult(ev.plan);
      break;
    case "error":
      el.runTitle.textContent = "Failed";
      traceLine("t-error", escapeHtml(ev.message));
      break;
  }
}

function brief(value, limit = 110) {
  const text = JSON.stringify(value);
  return text.length > limit ? text.slice(0, limit) + "…" : text;
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s == null ? "" : String(s);
  return div.innerHTML;
}

// ---------------------------------------------------------------------
// the answer: structured data only
// ---------------------------------------------------------------------

const VERDICT_COLORS = {
  OK: "#2f7d33",
  CAUTION: "#b26a00",
  SHORTEN: "#d7301f",
  DO_NOT_WALK: "#990000",
};

function visitSequence() {
  // stop name -> visit number (1-based), from the route tool's order;
  // index 0 is the start and gets no number
  const seq = {};
  (routeResult?.order || []).forEach(assign);
  function assign(name, i) {
    if (i > 0) seq[name] = i;
  }
  return seq;
}

function renderResult(plan) {
  el.result.hidden = false;
  const seq = visitSequence();

  el.verdicts.innerHTML = "";
  for (const w of plan.walks || []) {
    const card = document.createElement("div");
    card.className = `card verdict-card bl-${w.verdict}`;
    const n = seq[w.pet];
    card.innerHTML = `
      <h3>${n ? `<span class="mk mk-inline" style="background:${
        VERDICT_COLORS[w.verdict] || "#256abf"}">${n}</span> ` : ""}${escapeHtml(w.pet)}</h3>
      <p><span class="v v-${w.verdict}">${w.verdict.replaceAll("_", " ")}</span>
         &nbsp;${escapeHtml(w.walk_start)}–${escapeHtml(w.walk_end)}</p>
      ${w.notes ? `<p class="small">${escapeHtml(w.notes)}</p>` : ""}`;
    el.verdicts.appendChild(card);
  }

  el.timeline.innerHTML = "";
  for (const entry of routeResult?.timeline || []) {
    const li = document.createElement("li");
    li.textContent = entry.walk_minutes
      ? `${entry.arrive} — ${entry.stop}: ${entry.walk_minutes} min walk (until ${entry.walk_end})`
      : `${entry.arrive} — back at ${entry.stop}`;
    el.timeline.appendChild(li);
  }

  el.advice.textContent = plan.overall_advice || "";
  drawMap(plan, seq);
  el.result.scrollIntoView({ behavior: "smooth", block: "start" });
}

let map = null;

function drawMap(plan, seq) {
  const stops = routeCall?.stops || [];
  const verdictByPet = {};
  for (const w of plan?.walks || []) verdictByPet[w.pet] = w.verdict;
  const geometry = routeResult?.geometry || null;
  if (!stops.length) {
    document.getElementById("map").style.display = "none";
    return;
  }

  if (map) map.remove();
  const bounds = new maplibregl.LngLatBounds();
  for (const s of stops) bounds.extend([s.lon, s.lat]);

  map = new maplibregl.Map({
    container: "map",
    style: "https://tiles.openfreemap.org/styles/positron",
    // bounds in the constructor: fitting after construction can fail
    // silently if the just-unhidden container hasn't laid out yet
    bounds,
    fitBoundsOptions: { padding: 60, maxZoom: 15 },
    attributionControl: { compact: true },
    cooperativeGestures: true,
  });
  window._map = map;

  map.on("load", addMapLayers);

  function addMapLayers() {
    // belt and suspenders: by load time the container has real
    // dimensions, so re-measure and re-fit
    map.resize();
    map.fitBounds(bounds, { padding: 60, maxZoom: 15, duration: 0 });
    if (geometry) {
      map.addSource("route", {
        type: "geojson",
        data: { type: "Feature", geometry, properties: {} },
      });
      map.addLayer({
        id: "route",
        type: "line",
        source: "route",
        paint: { "line-color": "#256abf", "line-width": 4, "line-opacity": 0.8 },
      });
    } else if (routeResult) {
      // haversine fallback: straight lines between stops, honestly dashed
      const order = routeResult.order || stops.map((s) => s.name);
      const byName = Object.fromEntries(stops.map((s) => [s.name, s]));
      const coords = order.concat(order[0]).map((n) => [byName[n].lon, byName[n].lat]);
      map.addSource("route", {
        type: "geojson",
        data: { type: "Feature", properties: {},
          geometry: { type: "LineString", coordinates: coords } },
      });
      map.addLayer({
        id: "route",
        type: "line",
        source: "route",
        paint: { "line-color": "#256abf", "line-width": 3,
                 "line-dasharray": [2, 2], "line-opacity": 0.7 },
      });
    }
    stops.forEach(addStopMarker);

    function addStopMarker(s, i) {
      // numbered by visit order, colored by that dog's verdict;
      // the start stop gets an orange "S"
      const n = seq?.[s.name];
      const dot = document.createElement("div");
      dot.className = "mk";
      dot.textContent = n ? String(n) : "S";
      dot.style.background = n
        ? VERDICT_COLORS[verdictByPet[s.name]] || "#256abf"
        : "#d95f00";
      new maplibregl.Marker({ element: dot })
        .setLngLat([s.lon, s.lat])
        .setPopup(new maplibregl.Popup({ offset: 18 }).setText(s.name))
        .addTo(map);
    }
  }
}

async function loadInfo() {
  try {
    const info = await apiFetch("/info").then((r) => r.json());
    el.pill.textContent =
      `${info.model} · ${info.backend === "openrouter" ? "rented" : "local"}` +
      " inference · served from a laptop in a closet";
    const name = document.getElementById("model-name");
    if (name) name.textContent = info.model;
    for (const m of info.models || []) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.label;
      if (m.id === info.model) opt.selected = true;
      el.modelSelect.appendChild(opt);
    }
  } catch {
    el.pill.textContent = "served from a laptop in a closet";
  }
}

loadPresets();
loadInfo();
