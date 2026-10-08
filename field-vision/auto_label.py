"""
Use Qwen3-VL (running in Ollama on this machine) as a teacher that labels photos,
so the small phone model can learn soil, tree and leaf-disease classes without
you hand-sorting thousands of images.

Two ways to use it:

  1. Check web candidates from collect_web.py (folder name = suggested class).
     Keeps a photo only if Qwen agrees with the suggested class:

        python auto_label.py --input web_raw --verify --out raw

  2. Sort your own unlabeled photos (e.g. a folder of soil pictures from the field):

        python auto_label.py --input ~/Pictures/field_soil --domain soil --out raw
        python auto_label.py --input ~/Pictures/mixed --domain auto --out raw

Results:
    raw/<domain>/<class>/      photos Qwen is confident about -> ready for prepare_data.py
    review/<domain>/<class>/   unsure or disagreeing photos -> check by eye, move the good ones into raw/
    labels.csv                 every decision with Qwen's reason (also lets you stop and resume)

Qwen is a strong labeler but not perfect: open a sample of raw/ and delete mistakes
before training, and hand-check the photos that end up in your test split.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
import re
import shutil
import threading
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUBJECT = {"soil": "soil or ground", "tree": "a tree, tree trunk, branch or tree leaves",
           "leaf": "crop or garden plant leaves", "auto": "soil, a tree, or plant leaves"}


def load_classes(taxonomy, domain):
    if domain == "auto":
        out = {}
        for d, classes in taxonomy.items():
            if d.startswith("_"):
                continue
            for k, v in classes.items():
                if not k.startswith("_"):
                    out[k] = (d, v["description"])
        return out
    if domain not in taxonomy:
        raise SystemExit(f"Domain '{domain}' is not in the taxonomy file")
    return {k: (domain, v["description"]) for k, v in taxonomy[domain].items() if not k.startswith("_")}


def build_prompt(domain, classes):
    lines = "\n".join(f"- {k}: {desc}" for k, (_, desc) in classes.items())
    return f"""You are labeling photos to train an image classifier.
Pick the ONE label that best matches what this photo actually shows.

Labels:
{lines}
- none: not a real photo of {SUBJECT[domain]}; a drawing, diagram, map, text, product or logo;
  too blurry or dark; or no label clearly fits.

confidence: 0 to 1, how sure you are. Be strict; when in doubt, lower it.
reason: one short sentence on what you see."""


def encode(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((768, 768))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def ask(args, prompt, schema, img_b64):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "stream": False, "think": False, "format": schema, "keep_alive": "30m",
        "options": {"temperature": 0.1, "num_predict": 400},
    }
    last = None
    for _ in range(3):
        try:
            req = urllib.request.Request(f"{args.ollama}/api/chat", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                msg = json.load(r).get("message", {})
            return parse_json(msg.get("content") or msg.get("thinking"))
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")
            if e.code == 400 and "think" in body.lower() and "think" in payload:
                payload.pop("think")
                continue
            last = e
        except Exception as e:
            last = e
    raise RuntimeError(f"model failed 3 times: {last!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", default="raw", type=Path)
    ap.add_argument("--review", default="review", type=Path)
    ap.add_argument("--verify", action="store_true",
                    help="input is <domain>/<class>/ folders; keep only photos Qwen agrees with")
    ap.add_argument("--domain", help="soil | tree | leaf | auto (not needed with --verify)")
    ap.add_argument("--taxonomy", default="taxonomy.json", type=Path)
    ap.add_argument("--min-conf", type=float, default=0.75)
    ap.add_argument("--model", default="qwen3-vl:32b-instruct")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--parallel", type=int, default=4,
                    help="requests at once (set OLLAMA_NUM_PARALLEL to match for a real speedup)")
    ap.add_argument("--log", default="labels.csv", type=Path)
    args = ap.parse_args()
    if not args.verify and not args.domain:
        raise SystemExit("Give --domain (soil, tree, leaf or auto), or use --verify on collect_web.py output")

    taxonomy = json.loads(args.taxonomy.read_text())
    images = sorted(p for p in args.input.expanduser().rglob("*") if p.suffix.lower() in EXTS)

    # group images by the class set they're judged against
    jobs = []   # (path, domain_for_prompt, hint_label)
    for p in images:
        if args.verify:
            rel = p.relative_to(args.input.expanduser()).parts
            if len(rel) < 3 or rel[0] not in taxonomy:
                continue
            jobs.append((p, rel[0], rel[1]))
        else:
            jobs.append((p, args.domain, ""))

    done = set()
    if args.log.exists():
        with open(args.log) as f:
            done = {row["source"] for row in csv.DictReader(f)}
    jobs = [j for j in jobs if str(j[0]) not in done]
    print(f"{len(images)} images found, {len(done)} already labeled, {len(jobs)} to do. Model: {args.model}")
    if not jobs:
        return

    prompts, schemas, class_maps = {}, {}, {}
    for d in {j[1] for j in jobs}:
        cm = load_classes(taxonomy, d)
        class_maps[d] = cm
        prompts[d] = build_prompt(d, cm)
        schemas[d] = {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": list(cm) + ["none"]},
                "confidence": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["label", "confidence", "reason"],
        }

    new_log = not args.log.exists()
    lf = open(args.log, "a", newline="")
    log = csv.writer(lf)
    if new_log:
        log.writerow(["source", "hint", "label", "confidence", "decision", "reason"])
    lock = threading.Lock()
    stats = Counter()
    kept_per_class = Counter()

    def work(job):
        path, d, hint = job
        try:
            r = ask(args, prompts[d], schemas[d], encode(path))
            label, conf = r.get("label", "none"), float(r.get("confidence") or 0)
            reason = str(r.get("reason", ""))[:200]
        except Exception as e:
            label, conf, reason = "error", 0.0, str(e)[:200]

        cm = class_maps[d]
        if label == "error":
            decision = "error"
        elif label == "none":
            decision = "rejected" if conf >= args.min_conf else "review"
        elif conf >= args.min_conf and (not hint or label == hint):
            decision = "kept"
        else:
            decision = "review"

        if decision in ("kept", "review"):
            dom = cm[label][0] if label in cm else (d if d != "auto" else "unknown")
            base = args.out if decision == "kept" else args.review
            dest = base / dom / label / (hashlib.md5(str(path).encode()).hexdigest()[:16] + path.suffix.lower())
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        with lock:
            log.writerow([str(path), hint, label, f"{conf:.2f}", decision, reason])
            stats[decision] += 1
            if decision == "kept":
                kept_per_class[label] += 1
            n = sum(stats.values())
            if n % 25 == 0 or n == len(jobs):
                lf.flush()
                print(f"  {n}/{len(jobs)}  kept {stats['kept']}  review {stats['review']}  "
                      f"rejected {stats['rejected']}  errors {stats['error']}")

    with ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(work, jobs))
    lf.close()

    print("\nKept per class:")
    for k, v in sorted(kept_per_class.items()):
        print(f"  {k:30} {v}")
    print(f"\nKept photos are in {args.out}/. Unsure ones are in {args.review}/: look through them and "
          f"move good ones into {args.out}/<domain>/<class>/.")
    print(f"Next: python prepare_data.py --source {args.out} --out data")


if __name__ == "__main__":
    main()
