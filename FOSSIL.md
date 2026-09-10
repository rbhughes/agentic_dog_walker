# FOSSIL.md — the box that serves walker.purr.io

Runbook for `fossil`, the Dell Latitude 5430 in the closet. Written so
the whole setup can be re-understood (or rebuilt) cold.

## Quick facts

| | |
|---|---|
| Hardware | Dell Latitude 5430 — i5-1245U, 16 GB (single-channel), 256 GB NVMe |
| OS | Debian 13, headless (no desktop) |
| Hostname / tailnet | `fossil` / `100.71.229.15` (tailnet `taild72aca`) |
| Public URL | **https://fossil.taild72aca.ts.net** (Tailscale Funnel → service on :8010) |
| Service | `dogwalker.service` (systemd) → uvicorn on 127.0.0.1:8010 |
| Code | `/home/bryan/agentic_dog_walker` (git clone of the repo) |
| Secrets | `/home/bryan/agentic_dog_walker/.env` (OpenRouter + OpenRouteService keys) |
| Role | Serves the FastAPI/SSE agent service. Inference is OpenRouter; local Ollama is parked. |
| Verified | Unattended reboot → public API healthy in ~30 s (2026-09-11) |

**Tailscale account note: the tailnet is authenticated with Sign in with
APPLE** — to administer (admin console, ACLs, Funnel approvals, adding
machines), go to https://login.tailscale.com and pick "Sign in with Apple",
not Google/GitHub/email.

## Getting in

```bash
ssh bryan@fossil            # MagicDNS name, works from any tailnet machine
ssh bryan@100.71.229.15     # stable tailnet IP, if MagicDNS is being weird
ssh bryan@192.168.4.36      # home-LAN IP (DHCP -- may drift), tailnet-less fallback
```

SSH keys are installed from the Mac (ichabod) and the Arch laptop
(pepper). Passwordless sudo is enabled for `bryan`
(`/etc/sudoers.d/010-bryan-nopasswd`).

## The service (dogwalker)

```bash
systemctl status dogwalker              # is it up
sudo systemctl restart dogwalker        # restart after a deploy
sudo journalctl -u dogwalker -f         # live logs
sudo journalctl -u dogwalker --since "1 hour ago"
curl localhost:8010/healthz             # on-box check
curl https://fossil.taild72aca.ts.net/healthz    # the world's view
```

Unit file: `/etc/systemd/system/dogwalker.service` — runs
`uv run uvicorn dog_walker.service:app` as user `bryan`, working dir the
repo (that's how `.env` gets found), `Restart=always`.

### Deploying an update

```bash
ssh bryan@fossil
cd ~/agentic_dog_walker
git pull
~/.local/bin/uv sync
sudo systemctl restart dogwalker
curl localhost:8010/healthz
```

Run transcripts (one JSON per visitor run) accumulate in
`~/agentic_dog_walker/runs/` — gitignored, safe to delete when large.

## Tailscale / Funnel

```bash
tailscale status                        # who's on the tailnet
tailscale ip -4                         # this box's tailnet address
sudo tailscale funnel status            # what's exposed publicly
sudo tailscale funnel --bg 8010         # (re)expose the service port
sudo tailscale funnel --https=443 off   # stop public exposure
```

Facts worth remembering:

- **Funnel config persists across reboots** (proven) — no cron or unit
  needed for it.
- Funnel required a ONE-TIME tailnet approval (an admin-console click via
  a `login.tailscale.com/f/funnel?...` link). That's done; a rebuilt or
  replacement node would need it again.
- Only port 8010 is proxied publicly. Ollama (11434) is bound to the
  tailnet address and is NOT funneled — the world cannot reach it.
- Watch for **node key expiry**: by default Tailscale keys expire after
  ~180 days and the box silently drops off the tailnet. Either re-auth
  with `sudo tailscale up` when it happens, or (better) disable expiry
  for fossil in the admin console: Machines → fossil → ... →
  "Disable key expiry".

## Ollama (parked, not removed)

```bash
OLLAMA_HOST=100.71.229.15 ollama list           # models on disk
OLLAMA_HOST=100.71.229.15 ollama run qwen3:8b   # interactive chat
systemctl status ollama
```

Bound tailnet-only via systemd override
(`systemctl edit ollama` → `OLLAMA_HOST=100.71.229.15`), so even on-box
CLI needs the env var. Idle cost is ~zero (models load on demand and
unload after `keep_alive`). NOTE: the agent no longer speaks Ollama's
native dialect (retired 2026-09-11) — if local inference ever returns,
point the agent at Ollama's OpenAI-compatible endpoint
(`http://100.71.229.15:11434/v1/chat/completions`) instead.

## Debian upkeep

Security patches apply automatically (`unattended-upgrades`). Manual
maintenance, occasionally:

```bash
sudo apt update && sudo apt full-upgrade    # bring everything current
df -h /                                     # disk (256 GB, mostly empty)
free -h                                     # memory
uptime
sudo reboot                                 # safe: everything recovers
                                            # unattended in ~30 s
```

Power/lid behavior (why a closed laptop stays up): lid switches ignored in
`/etc/systemd/logind.conf`; `sleep.target suspend.target hibernate.target
hybrid-sleep.target` are masked. Undo with `systemctl unmask` if the box
ever goes back to being a laptop.

## Troubleshooting the public site

Site can't reach the API — check in this order:

1. `curl https://fossil.taild72aca.ts.net/healthz` from anywhere.
2. On fossil: `systemctl status dogwalker` (service died? `journalctl -u
   dogwalker -n 50`).
3. `sudo tailscale funnel status` (funnel off? re-run `sudo tailscale
   funnel --bg 8010`).
4. `tailscale status` (node offline? key expired? `sudo tailscale up`).
5. Visitors seeing HTTP 429: that's the armor working (rate limit or
   queue full), not an outage — see docs/SERVICE.md.
6. **The site fails ONLY on tailnet machines** (yours): MagicDNS
   resolves the API host to its private tailnet address, and Chrome's
   Local Network Access policy blocks public-page fetches to private
   addresses (page-load fetches have no user gesture, so the
   permission is auto-denied). Off-tailnet visitors resolve the
   public Funnel ingress and are unaffected — verified end-to-end.
   To test from a tailnet machine: use the Astro dev server
   (`npm run dev` in site/, localhost origin is exempt), click a
   button (a user gesture may surface Chrome's permission prompt),
   or point Chrome's Secure DNS at a public resolver.

## If fossil dies

Nothing on it is precious: the repo is on GitHub, secrets are two API
keys re-copyable from the Mac's `.env`, transcripts in `runs/` are
disposable. Rebuild = install headless Debian, then repeat: sudoers
file, lid/sleep config, unattended-upgrades, tailscale (+ funnel
re-approval), uv, clone, `.env`, the systemd unit. Every step is in this
file or CLAUDE.md.
