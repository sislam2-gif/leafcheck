"""
Run your own photos through an exported model on the PC and compare two ways of looking:

    whole  the middle of the whole photo (what predict() in the app does)
    grid   the photo cut into small squares, each checked, answers combined
           (what predictTiled() does; built for soil, trained on close-up patches)

    python try_photos.py --app exports/soil2 my_photos/
    python try_photos.py --app exports/soil2 photo1.jpg photo2.jpg --grids 2,3 3,4 4,5

Score it: put photos in folders named after the right answer, using the class label
(texture_clay, texture_loam, texture_sand), "not_relevant", or a domain name such as
"soil" (any soil answer counts as right):

    my_photos/texture_loam/garden.jpg
    my_photos/soil/cracked_crust.jpg
    my_photos/not_relevant/monstera.jpg

and it prints how many each method got right. Use this to choose --tiles for make_app.py.
The math here mirrors web/predict.js, so the phone gives the same answers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class Model:
    def __init__(self, app: Path):
        self.meta = json.loads((app / "labels.json").read_text())
        self.sess = ort.InferenceSession(str(app / "model.onnx"), providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name
        self.labels = [c["label"] for c in self.meta["classes"]]
        self.domain_of = {c["label"]: c["domain"] for c in self.meta["classes"]}
        self.domains = set(self.domain_of.values())
        self.is_other = np.array([c["domain"] == "other" for c in self.meta["classes"]])
        self.mean = np.array(self.meta["mean"], np.float32).reshape(3, 1, 1)
        self.std = np.array(self.meta["std"], np.float32).reshape(3, 1, 1)

    def run(self, im, regions):
        size = self.meta["img"]
        batch = []
        for sx, sy, side in regions:
            crop = im.resize((size, size), Image.BILINEAR, box=(sx, sy, sx + side, sy + side))
            x = np.asarray(crop, np.float32).transpose(2, 0, 1) / 255.0
            batch.append((x - self.mean) / self.std)
        return self.sess.run(None, {self.name: np.stack(batch).astype(np.float32)})[0]

    def whole(self, im):
        w, h = im.size
        side = min(w, h) * self.meta["crop_frac"]
        return self.answer(self.run(im, [((w - side) / 2, (h - side) / 2, side)])[0])

    def grid(self, im, grids, min_frac):
        w, h = im.size
        side = min(w, h)
        x0, y0 = (w - side) / 2, (h - side) / 2
        regions = []
        for g in grids:
            t = side / g
            regions += [(x0 + c * t, y0 + r * t, t) for r in range(g) for c in range(g)]
        probs = self.run(im, regions)
        tops = probs.argmax(1)
        subject = probs[~self.is_other[tops]]
        cov = f"{len(subject)}/{len(probs)}"
        if not self.is_other.any():
            return self.answer(probs.mean(0), cov)
        if len(subject) and len(subject) / len(probs) >= min_frac:
            m = subject.mean(0)
            m[self.is_other] = 0
            return self.answer(m, cov)
        return self.answer(probs.mean(0), cov)

    def answer(self, p, cov=""):
        order = np.argsort(-p)
        return {"label": self.labels[order[0]], "conf": float(p[order[0]]), "coverage": cov,
                "top3": [(self.labels[i], float(p[i])) for i in order[:3]]}


def collect(paths):
    out = []
    for p in paths:
        p = Path(p).expanduser()
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS)
        elif p.suffix.lower() in EXTS:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--app", required=True, type=Path, help="exported app folder, e.g. exports/soil2")
    ap.add_argument("--grids", nargs="+", default=["3,4"], help="grid settings to compare, e.g. 2,3 3,4 4,5")
    ap.add_argument("--min-frac", type=float, default=0.25)
    ap.add_argument("--details", action="store_true", help="show top-3 for every method")
    args = ap.parse_args()

    model = Model(args.app)
    photos = collect(args.photos)
    if not photos:
        raise SystemExit("No photos found.")
    settings = [("whole", None)] + [(f"grid {g}", [int(x) for x in g.split(",")]) for g in args.grids]
    score = {name: [0, 0] for name, _ in settings}

    for f in photos:
        with Image.open(f) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        folder = f.parent.name
        expected = folder if (folder in model.labels or folder in model.domains) else None
        print(f"\n{f.name}" + (f"   (should be: {expected})" if expected else ""))
        for name, grids in settings:
            r = model.whole(im) if grids is None else model.grid(im, grids, args.min_frac)
            mark = ""
            if expected:
                ok = r["label"] == expected or model.domain_of[r["label"]] == expected
                score[name][0] += ok
                score[name][1] += 1
                mark = "  OK" if ok else "  WRONG"
            cov = f"  [subject in {r['coverage']} squares]" if r["coverage"] else ""
            print(f"  {name:10} {r['label']:22} {r['conf']:6.0%}{cov}{mark}")
            if args.details:
                print("             " + ", ".join(f"{l} {c:.0%}" for l, c in r["top3"]))

    if any(n for _, n in score.values()):
        print("\nScore on photos in labeled folders:")
        for name, (ok, n) in score.items():
            print(f"  {name:10} {ok}/{n} correct")


if __name__ == "__main__":
    main()
