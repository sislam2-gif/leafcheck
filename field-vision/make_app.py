"""
Turn an exported model folder into an installable offline phone app.

    python make_app.py exports/v1 --name "Field Check"

export.py runs this for you. Run it yourself to rebuild the app around an
existing export (for example after editing advice.json or the app name).

The folder becomes a complete static website:
    index.html     the app
    sw.js          saves everything on the phone at first visit -> works offline
    manifest.json  name and icons, so "Add to Home Screen" makes a real app icon
    icons/         app icons (generated)
    advice.json    optional: advice cards keyed by class label, shown under results

Upload the folder to GitHub Pages (or any https host), open it once on a phone
with internet, install it, and it works in airplane mode from then on.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
VENDOR_FILES = ["ort.wasm.min.js", "ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"]
SKIP = {"sw.js", ".nojekyll"}


def make_icon(size: int, path: Path):
    """Green rounded square with a white leaf."""
    s = size
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, s - 1, s - 1), radius=int(s * 0.22), fill=(47, 125, 79, 255))
    leaf = Image.new("L", (s, s), 0)
    ld = ImageDraw.Draw(leaf)
    ld.ellipse((int(s * .22), int(s * .30), int(s * .78), int(s * .70)), fill=255)   # leaf body
    leaf = leaf.rotate(40, resample=Image.BICUBIC, center=(s / 2, s / 2))
    im.paste((255, 255, 255, 255), (0, 0), leaf)
    d = ImageDraw.Draw(im)
    w = max(2, s // 40)
    d.line((int(s * .30), int(s * .70), int(s * .70), int(s * .30)), fill=(47, 125, 79, 255), width=w)  # vein
    d.line((int(s * .24), int(s * .76), int(s * .34), int(s * .66)), fill=(255, 255, 255, 255), width=w * 2)  # stem
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def ensure_vendor(out: Path):
    vendor = out / "vendor"
    if all((vendor / f).exists() for f in VENDOR_FILES):
        return
    script = HERE / "get_ort.sh"
    print("Downloading the in-browser model runtime (ONNX Runtime Web)...")
    subprocess.run(["bash", str(script), str(vendor)], check=True)


def build_app(out: Path, name: str = "Field Check", threshold: float | None = None,
              tiles: list[int] | None = None, tile_min_frac: float | None = None, tip: str | None = None):
    out = Path(out)
    if not (out / "model.onnx").exists() or not (out / "labels.json").exists():
        raise SystemExit(f"{out} has no model.onnx/labels.json. Run export.py first.")

    ensure_vendor(out)

    shutil.copy(HERE / "web" / "predict.js", out / "predict.js")
    shutil.copy(HERE / "web" / "test.html", out / "test.html")
    html = (HERE / "web" / "index.html").read_text().replace("__APP_NAME__", name)
    (out / "index.html").write_text(html)

    labels = json.loads((out / "labels.json").read_text())
    if threshold is not None:
        labels["threshold"] = threshold
    labels.setdefault("threshold", 0.6)
    if tiles is not None:                # [] turns grid mode off
        if tiles:
            labels["tiles"] = tiles
        else:
            labels.pop("tiles", None)
    if tile_min_frac is not None:
        labels["tile_min_frac"] = tile_min_frac
    if tip is not None:
        labels["tip"] = tip
    (out / "labels.json").write_text(json.dumps(labels, indent=2))

    make_icon(192, out / "icons" / "icon-192.png")
    make_icon(512, out / "icons" / "icon-512.png")
    manifest = {
        "name": name, "short_name": name[:12], "start_url": "./", "scope": "./",
        "display": "standalone", "background_color": "#f6f7f2", "theme_color": "#2f7d4f",
        "icons": [
            {"src": "icons/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "icons/icon-512.png", "sizes": "512x512", "type": "image/png"},
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / ".nojekyll").write_text("")   # GitHub Pages: serve files as-is

    # every file the app needs, and a version that changes when any of them change
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in SKIP and not p.name.endswith(".map"))
    digest = hashlib.sha256()
    for p in files:
        digest.update(p.relative_to(out).as_posix().encode())
        digest.update(p.read_bytes())
    version = digest.hexdigest()[:12]
    urls = ["./"] + ["./" + p.relative_to(out).as_posix() for p in files]
    prefix = "fieldvision-" + hashlib.md5(name.encode()).hexdigest()[:6]
    sw = ((HERE / "web" / "sw.js").read_text()
          .replace("__VERSION__", version)
          .replace("__CACHE_PREFIX__", prefix)
          .replace("__FILES__", json.dumps(urls, indent=2)))
    (out / "sw.js").write_text(sw)

    total = sum(p.stat().st_size for p in files) / 1e6
    print(f"App built in {out}/  ({len(files)} files, {total:.0f} MB saved on the phone, version {version})")
    if not (out / "advice.json").exists():
        print("  Tip: add advice.json (label -> {do_now, prevent, ...}) to show what to do under each result.")
    return version


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="export folder, e.g. exports/v1")
    ap.add_argument("--name", default="Field Check", help="app name on the home screen")
    ap.add_argument("--threshold", type=float, help="below this confidence the app says 'Not sure' (default 0.6)")
    ap.add_argument("--tiles", help="grid mode for texture models like soil, e.g. '3,4' checks a 3x3 and a "
                                    "4x4 grid of squares and combines them; 'off' to turn it off")
    ap.add_argument("--tile-min-frac", type=float,
                    help="grid mode: share of squares that must show the subject (default 0.25)")
    ap.add_argument("--tip", help="hint shown under the button, e.g. 'Point straight down at bare soil'")
    args = ap.parse_args()
    tiles = None
    if args.tiles:
        tiles = [] if args.tiles.lower() == "off" else [int(x) for x in args.tiles.split(",")]
    build_app(args.folder, args.name, args.threshold, tiles, args.tile_min_frac, args.tip)
    print(f"\nTry it here:  cd {args.folder} && python3 -m http.server 8000   -> http://localhost:8000/")
    print("Put it on phones: see 'Install on phones' in README.md (GitHub Pages).")


if __name__ == "__main__":
    main()
