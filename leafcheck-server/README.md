# LeafCheck Expert Server

Runs Qwen3-VL on Atlas and serves a phone-friendly web page. Open the page on a
phone, take a photo of a **leaf** or **soil**, and get a diagnosis plus advice.

```
Phone browser ──photo──▶ Atlas :8080 (this server) ──▶ Ollama :11434 (qwen3-vl:30b)
              ◀─result──                              ◀── JSON diagnosis
```

The model only describes what it sees. The "what to do" text comes from
`advice.json`, so advice stays vetted and never includes made-up chemicals or doses.

## 1. Get the model (one time)

Atlas already runs Ollama. Qwen3-VL needs Ollama **0.12.7 or newer**.

```bash
ollama --version            # update if older than 0.12.7
ollama pull qwen3-vl:30b    # about 20 GB download
```

Quick check that it works:

```bash
ollama run qwen3-vl:30b "Describe this image" ./some-leaf.jpg
```

## 2. Start the server

```bash
cd leafcheck-server
chmod +x run.sh
./run.sh
```

It prints the address to open, for example `http://192.168.1.50:8080`.

No GPU nearby? `BACKEND=mock ./run.sh` gives fake answers so you can work on the page.

## Phone app from anywhere (Cloudflare link)

The quickest way to use this from any phone, on Wi-Fi or mobile data:

```bash
# terminal 1: start the server WITH an access code (required, since the link is public)
ACCESS_CODE=pick-a-code DEFAULT_PLACE="Dunbar, WV" MODEL=qwen3-vl:32b-instruct ./run.sh

# terminal 2: create the link
./share.sh
```

`share.sh` installs Cloudflare's free `cloudflared` tool the first time (asks for your
password), then prints a link like `https://some-random-words.trycloudflare.com`.
On the phone: open the link, enter the access code once, then tap **Install on this
phone** (Android) or **Share → Add to Home Screen** (iPhone Safari). It opens
full-screen with its own icon, and GPS works because the link is https.

- Atlas must stay on with both terminals running. The AI runs on Atlas, so this app
  needs internet (unlike the offline Field Check / Soil Check apps).
- The link **changes every time** you run `share.sh`; send the new one to your team.
  A permanent address needs a free Cloudflare account and a domain name (a "named tunnel").
- `share.sh` refuses to run if the server has no access code.

## 3. Open it on a phone

**Same Wi-Fi (easiest):** open the printed address in the phone's browser.
If it won't load, open the port in Ubuntu's firewall:

```bash
sudo ufw allow 8080/tcp
```

**From anywhere (recommended for the demo): Tailscale.** It's free, and gives
you a real `https://` address without opening your router.

1. Install Tailscale on Atlas (`curl -fsSL https://tailscale.com/install.sh | sh`, then `sudo tailscale up`) and the Tailscale app on your phone. Sign in with the same account on both.
2. On Atlas: `sudo tailscale serve --bg 8080`
3. It prints an address like `https://atlas.your-tailnet.ts.net`. Open that on the phone.

Teammates' phones need Tailscale too, or you can share the machine with them in the Tailscale admin page.

**Why https matters:** the camera works on plain `http://`, but the phone only
shares GPS location (used for the outbreak map) over `https://`. Over plain
Wi-Fi the page works but scans won't have a location.

### Lock it down

Anyone who can reach the address can use your GPU. Set a shared code:

```bash
ACCESS_CODE=pick-a-code ./run.sh
```

The phone asks for the code once and remembers it.

## 4. Start automatically on boot (optional)

Edit `User=` and the paths in `leafcheck.service`, then:

```bash
sudo cp leafcheck.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now leafcheck
journalctl -u leafcheck -f      # watch logs
```

## Settings

| Variable | Default | What it does |
|---|---|---|
| `BACKEND` | `ollama` | `ollama`, `openai` (vLLM or any OpenAI-compatible server), or `mock` |
| `MODEL` | `qwen3-vl:30b` | Ollama tag, or the vLLM model name |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Where Ollama listens |
| `OPENAI_URL` | `http://127.0.0.1:8000/v1` | Where vLLM listens |
| `ACCESS_CODE` | empty | Require a code on phones |
| `NOT_SURE_THRESHOLD` | `0.6` | Below this confidence, ask for a retake |
| `MAX_SIDE` | `1280` | Photos are shrunk to this many pixels before sending |

### Using vLLM instead of Ollama

vLLM is faster when several people scan at once. In one terminal:

```bash
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct --port 8000 --max-model-len 16384
```

Then start this server with:

```bash
BACKEND=openai MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct ./run.sh
```

## Soil pH and Ask tabs

**Soil pH:** enter a reading from a soil test kit or lab report, the crop, soil type
and area. It says whether the pH suits the crop and, if not, estimates how much
ground limestone (to raise it) or elemental sulfur (to lower it) to use, with how-to
and safety notes. The numbers come from extension tables in `soil_rules.json`, not
from the AI, and every result lists its sources:
- Crop pH ranges: Oklahoma Cooperative Extension, HLA-6036
- Lime: Iowa State University Extension (via Ask Extension)
- Sulfur: Ohio State University Extension, AGF-507

They're starting estimates; the page always says to confirm with a lab soil test.
Below pH 5.0 or above 7.5 it tells you to get a lab test instead of guessing.

**Ask:** a chatbot running on your local Qwen. It looks up the weather for your
town or GPS position (Open-Meteo, free, no account) and includes it, plus your
last Soil pH result, in every question. It is told not to give pesticide or
fertilizer doses. Weather needs Atlas to have internet; the model itself is local.

Settings:

| Variable | Default | What it does |
|---|---|---|
| `CHAT_MODEL` | same as `MODEL` | Model for the Ask tab, e.g. a text-only model like `qwen3.5:122b` |
| `DEFAULT_PLACE` | empty | Town used when the phone sends no location, e.g. `"Dunbar, WV"` |

## API (for the LeafCheck phone app to call during sync)

| Method | Path | Body / result |
|---|---|---|
| `GET` | `/api/health` | `{ ok, backend, model, access_code_required }` |
| `POST` | `/api/analyze` | multipart: `photo` (file), `mode` (`plant` or `soil`), optional `lat`, `lng`. Header `X-Access-Code` if set. Returns `{ id, mode, seconds, model, advice, thumb }` |
| `GET` | `/api/history` | Last 50 scans, newest first |
| `DELETE` | `/api/history/{id}` | Delete one scan |
| `GET` | `/api/soil-ph/options` | Crops and soil types for the form |
| `POST` | `/api/soil-ph` | JSON `{ph, crop, texture, area, area_unit, organic_matter?, moisture?}` → advice cards and sources |
| `GET` | `/api/weather` | `?place=Town, ST` or `?lat=&lng=` → now + 3-day forecast |
| `POST` | `/api/chat` | JSON `{messages:[{role,content}], place? or lat/lng, soil?}` → `{reply, weather}` |

`advice.status` is one of `ok`, `not_sure`, `not_target` (not a leaf / not soil),
or `unvetted` (a crop or disease outside our 8 labels; the page shows the model's
description and says to contact an extension office).

## Files

| File | What it is |
|---|---|
| `app.py` | The server: image handling, model calls, advice lookup, history (SQLite) |
| `advice.json` | Advice cards for the 8 plant labels and 9 soil issues. **Fill in `source` with the .edu/USDA page for each before the demo.** |
| `static/index.html` | The phone page (Scan, Soil pH, Ask, History) |
| `farm_tools.py` | Soil pH Analyzer, weather and Ask chatbot |
| `soil_rules.json` | Crop pH ranges and lime/sulfur tables, with sources |
| `run.sh` | Start script |
| `share.sh` | Public https link for phones (Cloudflare quick tunnel) |
| `static/manifest.json`, `static/sw.js`, `static/icons/` | Make the page installable as a phone app |
| `leafcheck.service` | systemd unit for auto-start |
| `scans.db` | Created on first scan; saved history |

## Known limits

- Soil pH and nutrients can't be judged from a photo. Put a soil test-kit strip or color chart in the photo and the model reads it.
- First scan after the model has been idle 30 minutes is slow (about 10-30 s) while Ollama reloads it.
- This is the base model, not fine-tuned yet. Measure it against the test/ folder before trusting the numbers.
