"""
LeafCheck Expert Server
Runs on Atlas. A phone opens the web page, takes a photo of a leaf or soil,
and this server asks Qwen3-VL what it sees, then attaches vetted advice.

Backends (set BACKEND env var):
  ollama  - default. Uses Ollama's /api/chat (model: qwen3-vl:30b)
  openai  - any OpenAI-compatible server, e.g. vLLM
  mock    - fake answers, no GPU needed (for building the UI)
"""
import base64
import io
import json
import os
import random
import re
import sqlite3
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

BASE = Path(__file__).parent
BACKEND = os.getenv("BACKEND", "ollama").lower()
MODEL = os.getenv("MODEL", "qwen3-vl:30b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OPENAI_URL = os.getenv("OPENAI_URL", "http://127.0.0.1:8000/v1")
ACCESS_CODE = os.getenv("ACCESS_CODE", "")          # optional shared code
MAX_SIDE = int(os.getenv("MAX_SIDE", "1280"))        # resize before sending
NOT_SURE = float(os.getenv("NOT_SURE_THRESHOLD", "0.6"))
MAX_UPLOAD = 15 * 1024 * 1024
DB_PATH = BASE / "scans.db"

ADVICE = json.loads((BASE / "advice.json").read_text(encoding="utf-8"))
PLANT_LABELS = list(ADVICE["plants"].keys())
SOIL_ISSUES = list(ADVICE["soil"].keys())

# ---------------------------------------------------------------- prompts

SYSTEM = (
    "You are an agricultural field assistant that inspects photos taken by "
    "farmers on a phone. Describe only what is visible. Never recommend "
    "specific chemicals or doses. Answer with JSON only, matching the schema."
)

PLANT_PROMPT = f"""Inspect this photo of a plant leaf.
1. Is it a plant leaf at all? If not, set is_plant false.
2. Identify the crop.
3. If the photo matches one of these known labels, use it exactly:
   {", ".join(PLANT_LABELS)}
   Otherwise use "other" and describe the crop and condition in words.
4. confidence is how sure you are of the label, from 0 to 1. Be honest; a blurry,
   distant or cluttered photo should get low confidence.
5. signs: 1-4 short phrases describing what you can actually see
   (spots, color, pattern, location on the leaf).
6. image_quality "poor" if blurry, dark, too far, or several leaves overlap,
   and give a one-line retake_tip."""

SOIL_PROMPT = f"""Inspect this photo of soil.
1. Is it soil at all? If not, set is_soil false.
2. Estimate texture (sandy, loamy, clay, silty) from grain size, clumping and cracks.
3. Estimate moisture and an organic matter hint from color (darker is usually higher).
4. issues: list only problems you can clearly see, using these exact words:
   {", ".join(i for i in SOIL_ISSUES if not i.startswith("ph_"))}
5. If a soil test kit, test strip or color chart is in the photo, read it:
   fill test_kit with pH and N/P/K levels as shown. If there is no kit, set
   present=false, ph=0 and each nutrient to "unreadable".
   Never guess pH or nutrients without a kit in the photo.
6. confidence from 0 to 1. observations: 1-4 short phrases of what you see."""

PLANT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_plant": {"type": "boolean"},
        "crop": {"type": "string"},
        "label": {"type": "string", "enum": PLANT_LABELS + ["other"]},
        "condition": {"type": "string"},
        "confidence": {"type": "number"},
        "signs": {"type": "array", "items": {"type": "string"}},
        "image_quality": {"type": "string", "enum": ["good", "poor"]},
        "retake_tip": {"type": "string"},
    },
    "required": ["is_plant", "crop", "label", "condition", "confidence",
                 "signs", "image_quality", "retake_tip"],
}

# Plain types only: Ollama's grammar engine handles these reliably.
LEVEL = {"type": "string", "enum": ["low", "medium", "high", "unreadable"]}
SOIL_SCHEMA = {
    "type": "object",
    "properties": {
        "is_soil": {"type": "boolean"},
        "texture": {"type": "string",
                    "enum": ["sandy", "loamy", "clay", "silty", "unknown"]},
        "moisture": {"type": "string",
                     "enum": ["dry", "moist", "wet", "waterlogged", "unknown"]},
        "color": {"type": "string"},
        "organic_matter_hint": {"type": "string",
                                "enum": ["low", "medium", "high", "unknown"]},
        "issues": {"type": "array",
                   "items": {"type": "string",
                             "enum": [i for i in SOIL_ISSUES if not i.startswith("ph_")]}},
        "test_kit": {
            "type": "object",
            "properties": {
                "present": {"type": "boolean"},
                "ph": {"type": "number"},
                "nitrogen": LEVEL, "phosphorus": LEVEL, "potassium": LEVEL,
            },
            "required": ["present", "ph", "nitrogen", "phosphorus", "potassium"],
        },
        "confidence": {"type": "number"},
        "observations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["is_soil", "texture", "moisture", "color", "organic_matter_hint",
                 "issues", "test_kit", "confidence", "observations"],
}

# ---------------------------------------------------------------- storage

def db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS scans (
        id TEXT PRIMARY KEY, ts REAL, mode TEXT, result TEXT, thumb TEXT,
        lat REAL, lng REAL)""")
    return con

# ---------------------------------------------------------------- images

def prepare_image(raw: bytes):
    """Fix rotation, convert to JPEG (Qwen3-VL in Ollama can't read WebP),
    shrink to MAX_SIDE. Returns (jpeg_b64, thumb_data_url)."""
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        raise HTTPException(400, "That file isn't a photo I can read.")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    full = base64.b64encode(buf.getvalue()).decode()

    thumb = img.copy()
    thumb.thumbnail((160, 160))
    tbuf = io.BytesIO()
    thumb.save(tbuf, "JPEG", quality=70)
    thumb_url = "data:image/jpeg;base64," + base64.b64encode(tbuf.getvalue()).decode()
    return full, thumb_url

# ---------------------------------------------------------------- model calls

def strip_think(text: str) -> str:
    """Remove reasoning: complete <think>...</think> blocks, and an unfinished <think> at the end."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return re.sub(r"<think>.*\Z", "", text, flags=re.S).strip()


def parse_json(text: str) -> dict:
    text = strip_think(text)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


async def ask_ollama(prompt, schema, img_b64, budget=2048):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt, "images": [img_b64]},
        ],
        "stream": False,
        "think": False,          # thinking can eat the whole output budget
        "format": schema,        # Ollama structured output
        "keep_alive": "30m",     # keep the model loaded between scans
        "options": {"temperature": 0.2, "num_predict": budget},
    }
    async with httpx.AsyncClient(timeout=300) as c:
        r = await c.post(f"{OLLAMA_URL}/api/chat", json=payload)
        if r.status_code == 400 and "think" in r.text.lower():
            payload.pop("think")          # model tag without thinking support
            r = await c.post(f"{OLLAMA_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message", {})
        # Some Qwen3-VL tags put the answer in "thinking" and leave "content" empty.
        text = msg.get("content") or msg.get("thinking") or ""
        try:
            return parse_json(text)
        except Exception:
            print(f"bad model output (done_reason={data.get('done_reason')}): {text[:400]!r}")
            raise


async def ask_openai(prompt, schema, img_b64, budget=2048):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ]},
        ],
        "temperature": 0.2,
        "max_tokens": budget,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "result", "schema": schema}},
    }
    async with httpx.AsyncClient(timeout=300) as c:
        r = await c.post(f"{OPENAI_URL}/chat/completions", json=payload,
                         headers={"Authorization": "Bearer local"})
        r.raise_for_status()
        return parse_json(r.json()["choices"][0]["message"]["content"])


def ask_mock(mode):
    time.sleep(1.0)
    if mode == "plant":
        label = random.choice(PLANT_LABELS)
        card = ADVICE["plants"][label]
        return {"is_plant": True, "crop": card["crop"], "label": label,
                "condition": card["name"], "confidence": round(random.uniform(0.4, 0.98), 2),
                "signs": ["(mock) brown spots with rings", "(mock) yellowing edges"],
                "image_quality": "good", "retake_tip": ""}
    issues = random.sample([i for i in SOIL_ISSUES if not i.startswith("ph_")], k=2)
    return {"is_soil": True, "texture": random.choice(["sandy", "loamy", "clay"]),
            "moisture": "dry", "color": "light brown", "organic_matter_hint": "low",
            "issues": issues,
            "test_kit": {"present": True, "ph": 5.4, "nitrogen": "low",
                         "phosphorus": "medium", "potassium": None},
            "confidence": round(random.uniform(0.5, 0.95), 2),
            "observations": ["(mock) surface crust", "(mock) fine cracks"]}


def normalize(mode, out):
    """Turn placeholder values back into None so the page shows 'unreadable'."""
    if mode == "soil":
        kit = out.get("test_kit") or {}
        if not kit.get("present") or not isinstance(kit.get("ph"), (int, float)) or kit["ph"] <= 0:
            kit["ph"] = None
        for k in ("nitrogen", "phosphorus", "potassium"):
            if kit.get(k) not in ("low", "medium", "high"):
                kit[k] = None
        out["test_kit"] = kit
    return out


async def ask_model(mode, img_b64):
    prompt, schema = (PLANT_PROMPT, PLANT_SCHEMA) if mode == "plant" else (SOIL_PROMPT, SOIL_SCHEMA)
    if BACKEND == "mock":
        return ask_mock(mode)
    ask = ask_openai if BACKEND == "openai" else ask_ollama
    last = None
    # Models occasionally return broken JSON, and thinking models (e.g. qwen3-vl:30b) can
    # use the whole budget reasoning, so each retry gets more room to finish.
    for attempt, budget in enumerate((2048, 6144, 12288)):
        try:
            return normalize(mode, await ask(prompt, schema, img_b64, budget))
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last = e
            print(f"attempt {attempt + 1} failed: {e!r}, retrying")
    raise last

# ---------------------------------------------------------------- advice

def attach_plant_advice(r):
    conf = float(r.get("confidence") or 0)
    if not r.get("is_plant"):
        return {"status": "not_target", "message": "That doesn't look like a plant leaf."}
    if conf < NOT_SURE or r.get("image_quality") == "poor":
        return {"status": "not_sure",
                "message": r.get("retake_tip") or
                "Not sure. Retake the photo closer, with one leaf on a plain background."}
    label = r.get("label")
    if label in ADVICE["plants"]:
        return {"status": "ok", "card": ADVICE["plants"][label]}
    return {"status": "unvetted",
            "message": "This isn't one of the conditions we have checked advice for. "
                       "The description above is the model's observation only. "
                       "Contact your county extension office before treating."}


def attach_soil_advice(r):
    if not r.get("is_soil"):
        return {"status": "not_target", "message": "That doesn't look like soil."}
    issues = list(r.get("issues") or [])
    kit = r.get("test_kit") or {}
    ph = kit.get("ph") if kit.get("present") else None
    if isinstance(ph, (int, float)):
        if ph < 5.8:
            issues.append("ph_low")
        elif ph > 7.5:
            issues.append("ph_high")
    cards = [ADVICE["soil"][i] | {"key": i} for i in issues if i in ADVICE["soil"]]
    status = "ok" if float(r.get("confidence") or 0) >= NOT_SURE else "not_sure"
    return {"status": status, "cards": cards,
            "kit_note": None if kit.get("present") else
            "pH and nutrients can't be judged from a photo. Add a test-kit strip "
            "to the photo for those."}

# ---------------------------------------------------------------- app

app = FastAPI(title="LeafCheck Expert Server")


def check_code(code):
    if ACCESS_CODE and code != ACCESS_CODE:
        raise HTTPException(401, "Wrong access code.")


@app.get("/api/health")
async def health():
    return {"ok": True, "backend": BACKEND, "model": MODEL,
            "access_code_required": bool(ACCESS_CODE)}


@app.post("/api/analyze")
async def analyze(photo: UploadFile = File(...), mode: str = Form("plant"),
                  lat: float | None = Form(None), lng: float | None = Form(None),
                  x_access_code: str = Header("")):
    check_code(x_access_code)
    if mode not in ("plant", "soil"):
        raise HTTPException(400, "mode must be plant or soil")
    raw = await photo.read()
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "Photo is too large (15 MB max).")
    img_b64, thumb = prepare_image(raw)

    started = time.time()
    try:
        model_out = await ask_model(mode, img_b64)
    except httpx.ConnectError:
        raise HTTPException(503, "The model server isn't running on Atlas.")
    except Exception as e:
        print("model error:", repr(e))
        raise HTTPException(502, "The model couldn't read this photo. Try again.")

    advice = attach_plant_advice(model_out) if mode == "plant" else attach_soil_advice(model_out)
    result = {"id": uuid.uuid4().hex, "mode": mode, "ts": time.time(),
              "seconds": round(time.time() - started, 1),
              "model": model_out, "advice": advice}

    with db() as con:
        con.execute("INSERT INTO scans VALUES (?,?,?,?,?,?,?)",
                    (result["id"], result["ts"], mode, json.dumps(result), thumb, lat, lng))
    return result | {"thumb": thumb}


@app.get("/api/history")
async def history(limit: int = 50, x_access_code: str = Header("")):
    check_code(x_access_code)
    with db() as con:
        rows = con.execute("SELECT result, thumb, lat, lng FROM scans "
                           "ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    return [json.loads(r) | {"thumb": t, "lat": la, "lng": ln} for r, t, la, ln in rows]


@app.delete("/api/history/{scan_id}")
async def delete_scan(scan_id: str, x_access_code: str = Header("")):
    check_code(x_access_code)
    with db() as con:
        con.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
    return {"ok": True}


from farm_tools import router as farm_router   # Soil pH Analyzer + Ask chatbot
app.include_router(farm_router)


@app.get("/")
async def index():
    return FileResponse(BASE / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/sw.js")          # served from the root so it can control the whole app
async def service_worker():
    return FileResponse(BASE / "static" / "sw.js", media_type="application/javascript",
                        headers={"Cache-Control": "no-cache"})

app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
