# LeafCheck team bundle

Everything we built for the agriculture track, in three folders. **Most teammates only
need the phone links at the bottom.** Only the person running Atlas (the GPU PC) needs
to run the server or train models.

| Folder | What it is | Who needs it |
|---|---|---|
| `leafcheck-server/` | **LeafCheck Expert**: a phone web app running on Atlas with Qwen3-VL. Tabs: **Scan** (leaf or soil photo → AI diagnosis + vetted advice), **Soil pH** (test-kit reading → lime/sulfur estimate from extension tables), **Ask** (farm chatbot that knows the local weather), **History** | Whoever runs Atlas |
| `field-vision/` | Trains the small **offline** models (plant disease, soil texture) and turns them into installable phone apps that work in airplane mode | ML teammate / whoever runs Atlas |
| `styles.css` | Stylesheet for the original LeafCheck PWA from the team plan (big buttons, sunlight contrast, severity colors) | UI teammate |

Each folder has its own `README.md` with full steps.

---

## Two kinds of app

| | Offline apps (Soil Check, Field Check) | LeafCheck Expert |
|---|---|---|
| Runs on | The phone itself | Atlas (needs internet) |
| Works in airplane mode | **Yes** | No |
| Speed | Under a second | 5–30 seconds |
| Knows | Only the classes it was trained on | Almost any plant; explains what it sees |
| Extras | — | Soil pH analyzer, weather-aware chatbot |

The pitch: **instant offline answer in the field, expert AI review when back online.**

---

## Running LeafCheck Expert (Atlas only)

```bash
cd leafcheck-server
# terminal 1: pick your own access code
ACCESS_CODE=your-code DEFAULT_PLACE="Dunbar, WV" MODEL=qwen3-vl:32b-instruct ./run.sh
# terminal 2: public https link for phones
./share.sh
```

`share.sh` prints a `https://….trycloudflare.com` link. **The link changes every time**,
so post the new one in the team chat along with the access code. Atlas has to stay on.

## Offline Soil Check app

Already published: **https://sislam2-gif.github.io/soil-check/**
Open it once with internet, tap Install (Android) or Share → Add to Home Screen
(iPhone Safari), then it works in airplane mode. Point straight down at bare soil from
about a hand's length away.

The trained model files live on Atlas (`field-vision/runs/`, `field-vision/exports/`).
They aren't in this bundle because they're large; `field-vision/README.md` shows how to
rebuild or update them.

---

## Honest limits (say these if judges ask)

- **Soil Check** (loam / sand / clay): 99.9% on the dataset's own held-out test photos
  (field photos from one research project). On our own phone photos it got 4 of 4 right
  using grid mode, but that's only 4 photos: collect more before quoting a number.
- **Plant model v1**: trained on PlantVillage lab photos (8 classes); 99.9% on similar
  photos, weaker on real field photos. Plant v2 (with PlantDoc field photos and a
  "not a plant" class) is the next step.
- **pH and nutrients can't be measured from a photo.** The Soil pH tab uses a test-kit
  or lab reading, and its amounts are estimates from Iowa State and Ohio State extension
  tables. Always confirm with a lab soil test.
- **Advice cards** in `leafcheck-server/advice.json` are starter text. Check each one
  against a .edu or USDA extension page and fill in its `source` before relying on it.
- The AI never gives pesticide or fertilizer doses; it points to the product label and
  the county extension office.

## Data credits

PlantVillage (spMohanty/PlantVillage-Dataset), PlantDoc (pratikkayal/PlantDoc-Dataset),
Imagenette (fast.ai), Soil texture 92k patches (Zenodo 10.5281/zenodo.20404766, CC BY 4.0),
Open-Meteo weather, extension guides from Oklahoma State, Iowa State, Ohio State, Cornell
and UC ANR (listed in `leafcheck-server/soil_rules.json`). Check each dataset's license
before publishing a model trained on it.
