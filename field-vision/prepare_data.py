"""
Build a clean train/val/test dataset from raw photo folders.

Default layout (one folder per domain, one subfolder per class):

    raw/
      plant/tomato_late_blight/*.jpg
      plant/tomato_healthy/*.jpg
      soil/clay_dry/*.jpg
      forest/healthy_canopy/*.jpg
      other/not_relevant/*.jpg        <- random non-target photos (recommended)

    python prepare_data.py --source raw --out data

Or pull classes out of a downloaded dataset with a map file, e.g. PlantVillage:

    python prepare_data.py --source ~/datasets/PlantVillage/color \
        --map maps/plantvillage_8.json --out data --max-per-class 1500

Several --source/--map runs can write into the same --out folder.
Images are deduplicated, auto-rotated, converted to RGB JPEG, and resized so
the long side is at most --size pixels (smaller files, faster training).
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS and p.is_file())


def collect(source: Path, map_file: Path | None):
    """Return {"domain/class": [image paths]}."""
    groups = defaultdict(list)
    if map_file:
        mapping = json.loads(map_file.read_text())
        # a dataset can have the same class folder name in several places (train/, test/...)
        # folder names match case-insensitively ("Loam" = "loam")
        dirs = defaultdict(list)
        for d in source.rglob("*"):
            if d.is_dir():
                dirs[d.name.lower()].append(d)
        for src_name, target in mapping.items():
            if src_name.startswith("_"):
                continue
            if src_name.lower() not in dirs:
                print(f"  ! folder '{src_name}' not found under {source}")
                continue
            for d in dirs[src_name.lower()]:
                groups[target] += find_images(d)
    else:
        for domain in sorted(d for d in source.iterdir() if d.is_dir()):
            for cls in sorted(c for c in domain.iterdir() if c.is_dir()):
                groups[f"{domain.name}/{cls.name}"] += find_images(cls)
    return groups


def split_hint(path: Path, source: Path):
    """Which split a dataset already put this file in, judged by its folder names."""
    for part in (p.lower() for p in path.relative_to(source).parts[:-1]):
        if "test" in part or "held" in part:
            return "test"
        if part.startswith("train"):
            return "train"
        if part.startswith(("val", "valid", "dev")):
            return "val"
    return None


def convert(job):
    src, dst, size = job
    try:
        raw = Path(src).read_bytes()
        digest = hashlib.md5(raw).hexdigest()
        if Path(dst).exists():
            return digest, "exists"
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((size, size))
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=90)
        return digest, "ok"
    except Exception as e:
        return None, f"skip {src}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--map", type=Path, help="JSON map: source folder name -> 'domain/class'")
    ap.add_argument("--out", default=Path("data"), type=Path)
    ap.add_argument("--split", default="0.8,0.1,0.1", help="train,val,test fractions")
    ap.add_argument("--max-per-class", type=int, default=0, help="cap images per class (0 = no cap)")
    ap.add_argument("--size", type=int, default=384, help="max long side in pixels")
    ap.add_argument("--respect-splits", action="store_true",
                    help="keep the dataset's own train/val/test folders (use when one photo was cut into "
                         "many crops, so crops of one photo never land in both train and test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    fr = [float(x) for x in args.split.split(",")]
    assert len(fr) == 3 and abs(sum(fr) - 1) < 1e-6, "--split must be three numbers summing to 1"

    source = args.source.expanduser()
    groups = collect(source, args.map)
    if not groups:
        raise SystemExit("No classes found. Check --source (and --map).")

    rng = random.Random(args.seed)
    jobs, plan = [], {}
    for target, files in groups.items():
        if target.count("/") != 1:
            raise SystemExit(f"Class '{target}' must look like 'domain/class'")
        # dedupe by content
        seen, unique = set(), []
        for f in files:
            h = hashlib.md5(f.read_bytes()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(f)
        rng.shuffle(unique)
        if args.respect_splits:
            parts = {"train": [], "val": [], "test": []}
            loose = []
            for f in unique:
                h = split_hint(f, source)
                (parts[h] if h else loose).append(f)
            n = len(loose)                      # files outside any split folder: split normally
            n_tr, n_va = int(n * fr[0]), int(n * fr[1])
            parts["train"] += loose[:n_tr]
            parts["val"] += loose[n_tr:n_tr + n_va]
            parts["test"] += loose[n_tr + n_va:]
            if not parts["val"] and parts["train"]:   # no val folder: borrow 10% of train
                k = max(1, len(parts["train"]) // 10)
                parts["val"], parts["train"] = parts["train"][:k], parts["train"][k:]
            if args.max_per_class:
                cap_small = max(50, int(args.max_per_class * 0.15))
                parts = {"train": parts["train"][: args.max_per_class],
                         "val": parts["val"][:cap_small], "test": parts["test"][:cap_small]}
        else:
            if args.max_per_class:
                unique = unique[: args.max_per_class]
            n = len(unique)
            n_tr, n_va = int(n * fr[0]), int(n * fr[1])
            parts = {"train": unique[:n_tr], "val": unique[n_tr:n_tr + n_va], "test": unique[n_tr + n_va:]}
        plan[target] = {k: len(v) for k, v in parts.items()}
        for split, items in parts.items():
            for f in items:
                name = hashlib.md5(str(f).encode()).hexdigest()[:16] + ".jpg"
                jobs.append((str(f), str(args.out / split / target / name), args.size))

    print(f"Converting {len(jobs)} images into {args.out}/ ...")
    skipped = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for _, status in pool.map(convert, jobs, chunksize=32):
            if status.startswith("skip"):
                skipped += 1
                if skipped <= 10:
                    print("  " + status)

    # summary across everything now in --out (supports several runs)
    summary = {}
    for split in ("train", "val", "test"):
        for cls_dir in sorted((args.out / split).glob("*/*")):
            key = f"{cls_dir.parent.name}/{cls_dir.name}"
            summary.setdefault(key, {})[split] = len(list(cls_dir.glob("*.jpg")))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'class':45} {'train':>6} {'val':>5} {'test':>5}")
    for k, v in summary.items():
        warn = "   <- under 100 train images" if v.get("train", 0) < 100 else ""
        print(f"{k:45} {v.get('train', 0):6} {v.get('val', 0):5} {v.get('test', 0):5}{warn}")
    if skipped:
        print(f"\nSkipped {skipped} unreadable files.")


if __name__ == "__main__":
    main()
