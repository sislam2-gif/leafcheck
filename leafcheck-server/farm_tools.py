"""
Soil pH Analyzer and Ask (farm chatbot) for the LeafCheck Expert server.

  POST /api/soil-ph   rule-based advice from a soil test reading (soil_rules.json)
  GET  /api/weather   current weather + 3-day forecast (Open-Meteo, free, no key)
  POST /api/chat      local Qwen in Ollama, with the weather and soil results as context

The pH advice never comes from the language model: amounts are interpolated from
extension-service tables, and every result lists its sources. The chatbot is told
not to give pesticide or fertilizer doses and to point to a soil test or the
county extension office instead.
"""
import json
import os
import re
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
RULES = json.loads((BASE / "soil_rules.json").read_text(encoding="utf-8"))

BACKEND = os.getenv("BACKEND", "ollama").lower()
CHAT_MODEL = os.getenv("CHAT_MODEL") or os.getenv("MODEL", "qwen3-vl:30b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OPENAI_URL = os.getenv("OPENAI_URL", "http://127.0.0.1:8000/v1")
ACCESS_CODE = os.getenv("ACCESS_CODE", "")
DEFAULT_PLACE = os.getenv("DEFAULT_PLACE", "")          # e.g. "Dunbar, WV"
WEATHER_URL = os.getenv("WEATHER_URL", "https://api.open-meteo.com/v1/forecast")
GEOCODE_URL = os.getenv("GEOCODE_URL", "https://geocoding-api.open-meteo.com/v1/search")

router = APIRouter()


def check_code(code):
    if ACCESS_CODE and code != ACCESS_CODE:
        raise HTTPException(401, "Wrong access code.")

# ================================================================ soil pH

class SoilInput(BaseModel):
    ph: float = Field(ge=3.0, le=10.0)
    crop: str
    texture: str = "loam"                       # sand | loam | silt | clay
    area: float = Field(default=100, gt=0)
    area_unit: str = "sqft"                     # sqft | acre
    organic_matter: float | None = Field(default=None, ge=0, le=100)
    moisture: str | None = None                 # dry | moist | wet


def interpolate(rows, ph, col):
    """Linear interpolation along the 'ph' column. Returns None outside the table."""
    rows = sorted(rows, key=lambda r: r["ph"])
    if ph < rows[0]["ph"] or ph > rows[-1]["ph"]:
        return None
    for a, b in zip(rows, rows[1:]):
        if a["ph"] <= ph <= b["ph"]:
            t = (ph - a["ph"]) / (b["ph"] - a["ph"])
            return a[col] + t * (b[col] - a[col])
    return rows[-1][col]


def analyze_soil(s: SoilInput):
    crop = RULES["crops"].get(s.crop)
    tex = RULES["texture_columns"].get(s.texture)
    if not crop:
        raise HTTPException(400, "Unknown crop.")
    if not tex:
        raise HTTPException(400, "Texture must be sand, loam, silt or clay.")
    sqft = s.area * (43560 if s.area_unit == "acre" else 1)
    lo, hi = crop["min"], crop["max"]
    src = RULES["sources"]
    out = {"crop": crop["name"], "ph": s.ph, "range": [lo, hi], "texture": tex["name"],
           "area_sqft": round(sqft), "cards": [], "sources": [src["crop_ph"]]}

    if lo <= s.ph <= hi:
        out["status"] = "ok"
        out["headline"] = f"pH {s.ph:g} is good for {crop['name'].lower()} ({lo}–{hi})."
        out["cards"].append({"title": "No lime or sulfur needed", "severity": "none",
                             "items": ["Retest every 2–3 years, or sooner if plants struggle."]})

    elif s.ph < lo:
        out["status"] = "low"
        out["headline"] = f"pH {s.ph:g} is too acidic for {crop['name'].lower()} (best {lo}–{hi})."
        per100 = interpolate(RULES["lime_to_6_5"]["rows"], s.ph, tex["lime"])
        if per100 is None:
            out["cards"].append({"title": "Get a lab soil test first", "severity": "high",
                                 "items": [f"pH {s.ph:g} is below the range of the lime table (5.0–6.5). "
                                           "Very acid soil needs a lab recommendation; a guess could be far off."]})
        else:
            total = per100 * sqft / 100
            out["cards"].append({
                "title": "Add ground limestone (estimate)", "severity": "medium",
                "amount": {"product": "ground limestone", "per_100_sqft_lb": round(per100, 1),
                           "total_lb": round(total, 1), "target_ph": 6.5},
                "items": [f"About {per100:.1f} lb per 100 sq ft → about {fmt_lb(total)} for "
                          f"{fmt_area(sqft)} ({tex['name'].lower()} soil, to pH 6.5)."] + RULES["lime_how"],
            })
            out["sources"] += [src["lime"], src["lime_how"]]

    else:
        out["status"] = "high"
        out["headline"] = f"pH {s.ph:g} is too alkaline for {crop['name'].lower()} (best {lo}–{hi})."
        per_acre = interpolate(RULES["sulfur_to_6_5_per_acre"]["rows"], s.ph, tex["sulfur"])
        if per_acre is None or s.ph > 7.5:
            out["cards"].append({"title": "Get a lab soil test first", "severity": "high",
                                 "items": ["Above pH 7.5 the soil may contain free lime, which makes sulfur "
                                           "estimates unreliable. Ask the lab to check for carbonates."]})
        if per_acre is not None:
            per100 = per_acre / 435.6
            total = per100 * sqft / 100
            out["cards"].append({
                "title": "Add elemental sulfur (estimate)", "severity": "medium",
                "amount": {"product": "elemental sulfur", "per_100_sqft_lb": round(per100, 2),
                           "total_lb": round(total, 1), "target_ph": 6.5},
                "items": [f"About {per100:.2f} lb per 100 sq ft → about {fmt_lb(total)} for "
                          f"{fmt_area(sqft)} ({tex['name'].lower()} soil, to pH 6.5)."] + RULES["sulfur_how"],
            })
            out["sources"].append(src["sulfur"])

    # general soil care from the other inputs (no amounts)
    care = []
    if s.organic_matter is not None and s.organic_matter < 3:
        care.append(f"Organic matter {s.organic_matter:g}% is low. Add compost or well-rotted manure "
                    "and keep the soil covered with mulch or a cover crop.")
    if s.moisture == "wet":
        care.append("The soil is wet: don't till or spread amendments until it crumbles in your hand.")
    elif s.moisture == "dry":
        care.append("The soil is dry: water deeply and less often, and mulch to hold moisture. "
                    "Lime and sulfur work slowly in dry soil.")
    if s.texture == "sand":
        care.append("Sandy soil loses water and nutrients fast; organic matter helps it hold both.")
    if s.texture == "clay":
        care.append("Clay soil drains slowly; avoid working it wet and add compost to loosen it.")
    if care:
        out["cards"].append({"title": "Soil care", "severity": "low", "items": care})

    if out["status"] != "ok":
        out["cards"].append({"title": "Safety", "severity": "low", "items": RULES["safety"]})
    out["confirm"] = ("These are starting estimates from extension tables. For exact amounts, send a "
                      "sample to a soil testing lab; your county extension office can help.")
    return out


def fmt_lb(lb):
    return f"{lb:,.0f} lb" if lb >= 10 else f"{lb:.1f} lb"


def fmt_area(sqft):
    return f"{sqft / 43560:,.2f} acres" if sqft >= 43560 / 4 else f"{sqft:,.0f} sq ft"


@router.get("/api/soil-ph/options")
async def soil_options():
    return {"crops": [{"id": k, "name": v["name"], "min": v["min"], "max": v["max"]}
                      for k, v in sorted(RULES["crops"].items(), key=lambda kv: kv[1]["name"])],
            "textures": [{"id": k, "name": v["name"]} for k, v in RULES["texture_columns"].items()]}


@router.post("/api/soil-ph")
async def soil_ph(body: SoilInput, x_access_code: str = Header("")):
    check_code(x_access_code)
    return analyze_soil(body)

# ================================================================ weather

_cache: dict = {}
CODES = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
         51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain",
         65: "heavy rain", 66: "freezing rain", 67: "freezing rain", 71: "light snow", 73: "snow",
         75: "heavy snow", 77: "snow grains", 80: "rain showers", 81: "rain showers",
         82: "heavy rain showers", 85: "snow showers", 86: "snow showers", 95: "thunderstorm",
         96: "thunderstorm with hail", 99: "thunderstorm with hail"}


async def geocode(place):
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(GEOCODE_URL, params={"name": place.split(",")[0].strip(), "count": 10,
                                            "language": "en", "format": "json"})
        r.raise_for_status()
        results = r.json().get("results") or []
    if not results:
        return None
    # "Dunbar, WV": prefer a result whose state/country matches the part after the comma
    hint = place.split(",", 1)[1].strip().lower() if "," in place else ""
    pick = results[0]
    if hint:
        for res in results:
            fields = " ".join(str(res.get(k, "")) for k in ("admin1", "country", "country_code")).lower()
            if hint in fields or (len(hint) == 2 and STATE_ABBR.get(hint.upper(), "").lower() in fields):
                pick = res
                break
    name = ", ".join(x for x in (pick.get("name"), pick.get("admin1"), pick.get("country_code")) if x)
    return pick["latitude"], pick["longitude"], name


async def get_weather(lat=None, lng=None, place=None):
    place = place or DEFAULT_PLACE
    label = None
    if (lat is None or lng is None) and place:
        g = await geocode(place)
        if not g:
            return {"error": f"Couldn't find '{place}'."}
        lat, lng, label = g
    if lat is None or lng is None:
        return None
    key = (round(lat, 2), round(lng, 2))
    if key in _cache and time.time() - _cache[key][0] < 900:
        return _cache[key][1]
    params = {
        "latitude": lat, "longitude": lng, "timezone": "auto", "forecast_days": 3,
        "temperature_unit": "fahrenheit", "precipitation_unit": "inch", "wind_speed_unit": "mph",
        "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weather_code",
    }
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(WEATHER_URL, params=params)
        r.raise_for_status()
        d = r.json()
    cur, day = d.get("current", {}), d.get("daily", {})
    w = {
        "place": label or f"{lat:.2f}, {lng:.2f}",
        "now": {"temp_f": cur.get("temperature_2m"), "humidity": cur.get("relative_humidity_2m"),
                "precip_in": cur.get("precipitation"), "wind_mph": cur.get("wind_speed_10m"),
                "sky": CODES.get(cur.get("weather_code"), "")},
        "days": [{"date": day["time"][i], "high_f": day["temperature_2m_max"][i],
                  "low_f": day["temperature_2m_min"][i], "rain_in": day["precipitation_sum"][i],
                  "rain_chance": (day.get("precipitation_probability_max") or [None] * 3)[i],
                  "sky": CODES.get(day["weather_code"][i], "")}
                 for i in range(len(day.get("time", [])))],
    }
    _cache[key] = (time.time(), w)
    return w


def weather_text(w):
    if not w or "error" in w:
        return "Weather: not available."
    n = w["now"]
    lines = [f"Location: {w['place']}.",
             f"Now: {n['temp_f']}°F, {n['sky']}, humidity {n['humidity']}%, wind {n['wind_mph']} mph, "
             f"rain this hour {n['precip_in']} in."]
    for i, dday in enumerate(w["days"]):
        when = ["Today", "Tomorrow", "Day after"][i] if i < 3 else dday["date"]
        chance = f", {dday['rain_chance']}% chance of rain" if dday.get("rain_chance") is not None else ""
        lines.append(f"{when}: {dday['sky']}, high {dday['high_f']}°F, low {dday['low_f']}°F, "
                     f"rain {dday['rain_in']} in{chance}.")
    return "\n".join(lines)


@router.get("/api/weather")
async def weather(lat: float | None = None, lng: float | None = None, place: str | None = None,
                  x_access_code: str = Header("")):
    check_code(x_access_code)
    try:
        w = await get_weather(lat, lng, place)
    except httpx.HTTPError:
        raise HTTPException(503, "Weather service unreachable.")
    if not w:
        raise HTTPException(400, "Share your location or type a town.")
    if "error" in w:
        raise HTTPException(404, w["error"])
    return w

# ================================================================ chat

CHAT_SYSTEM = """You are a friendly, practical farm and garden assistant for small farmers.
Answer in plain words, short paragraphs or a few bullet points, using US units.
Use the weather and soil information you are given when it matters (for example, don't
advise watering or spraying right before heavy rain, warn about frost).
Rules:
- Never give pesticide, herbicide, fungicide or fertilizer product names with doses or rates.
  Say what kind of treatment helps and tell them to follow the product label and their
  county extension office.
- For lime or sulfur amounts, refer to the Soil pH tab's estimate or a lab soil test; don't make up amounts.
- If a problem could spread fast (like late blight), say so and suggest contacting the extension office.
- If you're not sure, say so. Don't invent facts, statistics or sources.
- You can't see photos in this chat; suggest the Scan tab for photos."""


class ChatMsg(BaseModel):
    role: str
    content: str = Field(max_length=4000)


class ChatInput(BaseModel):
    messages: list[ChatMsg] = Field(min_length=1, max_length=30)
    lat: float | None = None
    lng: float | None = None
    place: str | None = None
    crop: str | None = None
    soil: dict | None = None           # last Soil pH result, if the user ran one


def clean_reply(text):
    """Remove reasoning: complete <think>...</think> blocks, and an unfinished <think> at the end."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return re.sub(r"<think>.*\Z", "", text, flags=re.S).strip()


async def ask_chat(messages):
    # thinking models can spend the whole budget reasoning: retry with more room if the answer is empty
    for budget in (1500, 6000):
        reply = await _ask_chat(messages, budget)
        if reply:
            return reply
    return ""


async def _ask_chat(messages, budget):
    if BACKEND == "mock":
        last = messages[-1]["content"]
        ctx = messages[0]["content"]
        has_w = "Now:" in ctx
        return (f"(mock reply) You asked: \"{last[:80]}\". "
                + ("I can see the weather for your area. " if has_w else "I don't have your weather. ")
                + "Water at the base of plants in the morning.")
    async with httpx.AsyncClient(timeout=180) as c:
        if BACKEND == "openai":
            r = await c.post(f"{OPENAI_URL}/chat/completions", headers={"Authorization": "Bearer local"},
                             json={"model": CHAT_MODEL, "messages": messages, "temperature": 0.4, "max_tokens": budget})
            r.raise_for_status()
            return clean_reply(r.json()["choices"][0]["message"]["content"])
        payload = {"model": CHAT_MODEL, "messages": messages, "stream": False, "think": False,
                   "keep_alive": "30m", "options": {"temperature": 0.4, "num_predict": budget}}
        r = await c.post(f"{OLLAMA_URL}/api/chat", json=payload)
        if r.status_code == 400 and "think" in r.text.lower():
            payload.pop("think")
            r = await c.post(f"{OLLAMA_URL}/api/chat", json=payload)
        r.raise_for_status()
        msg = r.json().get("message", {})
        return clean_reply(msg.get("content") or msg.get("thinking"))


@router.post("/api/chat")
async def chat(body: ChatInput, x_access_code: str = Header("")):
    check_code(x_access_code)
    try:
        w = await get_weather(body.lat, body.lng, body.place)
    except httpx.HTTPError:
        w = None
    context = [weather_text(w)]
    if body.crop and body.crop in RULES["crops"]:
        context.append(f"Main crop: {RULES['crops'][body.crop]['name']}.")
    if body.soil:
        s = body.soil
        context.append(f"Latest soil test the user entered: pH {s.get('ph')}, {s.get('texture', '')} soil, "
                       f"crop {s.get('crop', '')}. Analyzer result: {s.get('headline', '')}")
    messages = [{"role": "system", "content": CHAT_SYSTEM + "\n\nContext:\n" + "\n".join(context)}]
    messages += [{"role": m.role if m.role in ("user", "assistant") else "user", "content": m.content}
                 for m in body.messages[-12:]]
    try:
        reply = await ask_chat(messages)
    except httpx.ConnectError:
        raise HTTPException(503, "The model server isn't running on Atlas.")
    except Exception as e:
        print("chat error:", repr(e))
        raise HTTPException(502, "The model didn't answer. Try again.")
    return {"reply": reply or "Sorry, I didn't get an answer. Try asking again.",
            "weather": w if w and "error" not in w else None}


STATE_ABBR = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}
