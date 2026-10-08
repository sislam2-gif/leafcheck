"""
Collect candidate photos for each class from Openverse (openly licensed images).

    python collect_web.py --domains soil tree leaf --per-class 300 --out web_raw

Writes web_raw/<domain>/<class>/*.jpg plus web_raw/credits.csv (title, creator,
license and source page of every photo).

These are only CANDIDATES. Web search results are noisy (a "dead tree" search
returns drawings, signs, people...). Always run auto_label.py --verify on this
folder next: Qwen keeps only photos that really show the class.
"""
import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://api.openverse.org/v1/images/"
UA = {"User-Agent": "field-vision-dataset-builder/1.0 (educational project)"}


def get_json(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:            # rate limited: back off
                wait = 30 * (i + 1)
                print(f"    rate limited, waiting {wait}s")
                time.sleep(wait)
                continue
            if e.code in (400, 401, 403):   # past the anonymous page limit, or blocked
                return None
            time.sleep(3)
        except Exception:
            time.sleep(3)
    return None


def download(job):
    url, dest = job
    if dest.exists():
        return True
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            data = r.read()
        if len(data) < 2000:             # tiny files are usually error pages
            return False
        dest.write_bytes(data)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taxonomy", default="taxonomy.json", type=Path)
    ap.add_argument("--domains", nargs="+", default=["soil", "tree", "leaf"])
    ap.add_argument("--per-class", type=int, default=300, help="target candidates per class")
    ap.add_argument("--out", default="web_raw", type=Path)
    ap.add_argument("--license", default="", help="e.g. 'cc0,by,by-sa' to restrict licenses (default: any open license)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--api", default=API, help=argparse.SUPPRESS)
    args = ap.parse_args()

    tax = json.loads(args.taxonomy.read_text())
    credits_path = args.out / "credits.csv"
    args.out.mkdir(parents=True, exist_ok=True)
    new_credits = not credits_path.exists()
    cf = open(credits_path, "a", newline="")
    credits = csv.writer(cf)
    if new_credits:
        credits.writerow(["file", "class", "title", "creator", "license", "license_version", "source_page"])

    pool = ThreadPoolExecutor(args.workers)
    for domain in args.domains:
        classes = {k: v for k, v in tax[domain].items() if not k.startswith("_")}
        for cls, info in classes.items():
            folder = args.out / domain / cls
            folder.mkdir(parents=True, exist_ok=True)
            have = len(list(folder.glob("*.jpg")))
            per_query = max(20, args.per_class // len(info["queries"]))
            print(f"{domain}/{cls}: {have} already")
            seen = set()
            for q in info["queries"]:
                if have >= args.per_class:
                    break
                got, page = 0, 1
                while got < per_query and have < args.per_class:
                    params = {"q": q, "page_size": 20, "page": page, "mature": "false"}
                    if args.license:
                        params["license"] = args.license
                    data = get_json(args.api + "?" + urllib.parse.urlencode(params))
                    time.sleep(1.0)      # be polite to the free API
                    if not data or not data.get("results"):
                        break
                    jobs, meta = [], []
                    for r in data["results"]:
                        if r["id"] in seen:
                            continue
                        seen.add(r["id"])
                        dest = folder / f"{r['id']}.jpg"
                        jobs.append((r.get("thumbnail") or r["url"], dest))
                        meta.append((dest, r))
                    for ok, (dest, r) in zip(pool.map(download, jobs), meta):
                        if ok:
                            got += 1
                            have += 1
                            credits.writerow([str(dest.relative_to(args.out)), f"{domain}/{cls}",
                                              r.get("title"), r.get("creator"), r.get("license"),
                                              r.get("license_version"), r.get("foreign_landing_url")])
                    cf.flush()
                    page += 1
                print(f"   '{q}': +{got}")
            print(f"   -> {have} candidates")
    cf.close()
    print(f"\nDone. Next: python auto_label.py --input {args.out} --verify --out raw")


if __name__ == "__main__":
    main()
