# Field Vision: train an offline model for plants, soil and forest photos

Train a small image model on Atlas's GPU, then run it **with no internet** on
phones, laptops and other devices.

```
photos ──▶ prepare_data.py ──▶ train.py (GPU) ──▶ export.py ──▶ model.onnx (~17 MB)
                                                                  │
                         phone browser (LeafCheck, offline) ◀─────┤
                         Android / iOS apps (ONNX Runtime)  ◀─────┤
                         Raspberry Pi / laptop (Python)     ◀─────┘
```

The model is **MobileNetV3**, a network designed for phones. It starts from a
version already trained on millions of general photos (ImageNet) and learns your
classes on top of that (transfer learning). In a browser it takes tens of
milliseconds per photo on a desktop and typically well under a second on a phone.

---

## 0. Setup on Atlas (once)

```bash
cd field-vision
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line should print `True` and your RTX PRO 6000. If it prints `False`,
your PyTorch has no CUDA support. Install a CUDA build (`pip install torch torchvision`
from PyPI includes it on Linux), and check `nvidia-smi` works.

---

## Recipe: plants, diseases, soil and trees in one model

This builds **v2**: 38 crop/disease classes from lab and field photos, 7 soil
conditions, 8 tree conditions, 8 general leaf-disease symptoms, and a
"not relevant" class. Details on each tool follow below.

```bash
# 1. public datasets: PlantVillage, PlantDoc (field photos), Imagenette ("not relevant")
./fetch_datasets.sh

# 2. plants and diseases
python prepare_data.py --source ~/datasets/PlantVillage-Dataset/raw/color \
    --map maps/plantvillage_all.json --out data_v2 --max-per-class 1000
python prepare_data.py --source ~/datasets/PlantDoc-Dataset --map maps/plantdoc.json --out data_v2
python prepare_data.py --source ~/datasets/imagenette2-320 --map maps/imagenette_other.json \
    --out data_v2 --max-per-class 1500

# 3. soil, trees, general leaf diseases: download candidates, let Qwen keep the real ones
python collect_web.py --domains soil tree leaf --per-class 400 --out web_raw
python auto_label.py --input web_raw --verify --out raw
#    -> look through review/ and raw/ (see "Qwen as teacher" below)
python prepare_data.py --source raw --out data_v2

# 4. train and export
python train.py --data data_v2 --name v2 --workers 24
python export.py --run runs/v2 --data data_v2
```

Step 3 is the slow one, because Qwen looks at every photo. Watch the first progress
line to estimate the total, and leave it running overnight if needed. For more
speed, use `--model qwen3-vl:30b` (faster, slightly less careful), or restart
Ollama with `OLLAMA_NUM_PARALLEL=4` so `--parallel 4` really runs 4 at once.

### Soil models from public datasets

| Dataset | What it teaches | License | Get it |
|---|---|---|---|
| [Soil texture, 92k patches](https://zenodo.org/records/20404766) | **Loam / Sand / Clay** from field photos (natural light, varied moisture) | CC BY 4.0 | `./fetch_soil.sh` (1.7 GB) |
| [Soil types (7)](https://github.com/Phantom-fs/Soil-Classification-Dataset) | Alluvial, black, laterite, red, yellow, arid, mountain soil | CC BY-NC 4.0 (non-commercial) | `./fetch_soil.sh --types` |
| [Soil moisture](https://data.mendeley.com/datasets/skcc44yvvg/2) | Wet / moderate / dry, smartphone photos | see page | Download by hand from the page |

**Soil texture app** (the strongest of these):

```bash
./fetch_soil.sh                  # prints the folder names inside the zip
./fetch_datasets.sh              # for the "not relevant" photos (skips what you already have)

python prepare_data.py --source ~/datasets/soil-texture-92k --map maps/soil_texture.json \
    --out data_soil --respect-splits --max-per-class 8000
python prepare_data.py --source ~/datasets/imagenette2-320 --map maps/imagenette_other.json \
    --out data_soil --max-per-class 3000

python train.py --data data_soil --name soil1 --workers 24
python export.py --run runs/soil1 --data data_soil --app-name "Soil Check"
cp advice/soil_texture.json exports/soil1/advice.json && python make_app.py exports/soil1 --name "Soil Check"
```

- `--respect-splits` matters here: the 92k images are crops cut from far fewer
  photos. Keeping the dataset's own train/test folders stops crops of the same
  photo from landing in both, which would make the test score look better than it is.
- If `prepare_data.py` says a folder isn't found, look at the folder names
  `fetch_soil.sh` printed and edit `maps/soil_texture.json` to match.
- Take soil photos close up, filling the frame with soil, like the training patches.

For soil-type classes, use `--source ~/datasets/Soil-Classification-Dataset/Orignal-Dataset`
with `maps/soil_types.json` (the `CyAUG-Dataset` folder is AI-generated images).

### Soil v3: making it work on real phone photos

`soil2` scores 99.9% on the dataset's own test photos but rejects ordinary phone
photos of soil as "not relevant". It learned from tiny close-up patches from one
research project, so a normal photo of the ground, gray soil, or soil with hands in
it looks unfamiliar. Three fixes:

1. **Grid mode** (`--tiles 3,4`): the app cuts each photo into 3x3 and 4x4 squares
   at roughly the zoom of the training patches, checks every square, and combines
   the answers. Squares showing hands or leaves are outvoted by the soil squares.
2. **Real-distance soil photos in training**: the Qwen-checked photos in `raw/soil/`
   are added to the clay and sand classes. Grayscale augmentation means gray soil
   isn't rejected.
3. **Plants and leaves as "not relevant"**: without this, grid squares of a leaf
   look like texture and get called soil.

```bash
# 0. measure soil2 on your own photos first: whole photo vs grid
python try_photos.py --app exports/soil2 test_photos --grids 3,4 2,3

# 1. build the v3 dataset
python prepare_data.py --source ~/datasets/soil-texture-92k --map maps/soil_texture.json \
    --out data_soil3 --respect-splits --max-per-class 8000
python prepare_data.py --source raw --map maps/raw_soil_to_texture.json --out data_soil3
python prepare_data.py --source ~/datasets/imagenette2-320 --map maps/imagenette_other.json \
    --out data_soil3 --max-per-class 3000
python prepare_data.py --source ~/datasets/PlantDoc-Dataset --map maps/plants_as_other.json \
    --out data_soil3 --max-per-class 2000
python prepare_data.py --source ~/datasets/PlantVillage-Dataset/raw/color --map maps/plants_as_other.json \
    --out data_soil3 --max-per-class 2000

# 2. train, export, build the app in grid mode
python train.py --data data_soil3 --name soil3 --workers 24
python export.py --run runs/soil3 --data data_soil3 --app-name "Soil Check"
cp advice/soil_texture.json exports/soil3/advice.json
python make_app.py exports/soil3 --name "Soil Check" --tiles 3,4 \
    --tip "Point straight down at bare soil, about a hand's length away."

# 3. measure again on the same photos
python try_photos.py --app exports/soil3 test_photos --grids 3,4 2,3
```

`test_photos/` starts with four real photos sorted into folders by the right
answer (`texture_loam/`, `soil/` = any soil answer is right, `not_relevant/`).
**Add your own photos the same way**; that score is the one that matters for the
demo. The app's settings match whichever `--grids` value scores best: pass that
value to `make_app.py --tiles`.

Publish the new version to the phone app:
```bash
rsync -a --delete --exclude .git exports/soil3/ ~/soil-check-site/
cd ~/soil-check-site && git add -A && git commit -m "Soil Check v3" && git push
```
Phones pick up the new model the next time they open the app with internet.

### Qwen as teacher (auto_label.py)

There are few good public photo sets for soil and tree conditions. Instead,
Qwen3-VL labels photos for you and the small model learns from its labels
(this is called distillation):

- **Web candidates** (`--verify`): `collect_web.py` searches Openverse, a
  collection of openly licensed photos, for each class (search terms are in
  `taxonomy.json`). Searches are noisy, so `auto_label.py --verify` keeps a photo
  only if Qwen independently picks the same class with confidence of at least
  0.75. Credits for every photo are saved in `web_raw/credits.csv`.
- **Your own photos** (`--domain soil|tree|leaf|auto`): point it at a folder of
  unsorted field photos and it sorts them into classes. Real photos from your
  area beat web photos, so this is the best long-term source.
- **Check its work.** Open 20–30 photos from a few `raw/` classes and delete
  wrong ones. Skim `review/` for good photos Qwen wasn't sure about. If the model
  only ever learns from Qwen, its test score measures agreement with Qwen, so
  hand-check a few test photos for each soil and tree class.
- **Add or change classes** by editing `taxonomy.json`. Each class needs a
  clear visual description (Qwen reads it) and a few search phrases. Only use
  conditions you can see in a photo; pH or nutrient levels can't be learned
  from images.

Before showing any new class in LeafCheck, give it an advice card (`diseases.json`
or the expert server's `advice.json`).

---

## 1. Collect photos

Put them in **one folder per domain, one subfolder per class**:

```
raw/
├── plant/
│   ├── tomato_healthy/         *.jpg
│   ├── tomato_late_blight/
│   └── ...
├── soil/
│   ├── clay_dry/
│   ├── sandy/
│   ├── waterlogged/
│   └── eroded/
├── forest/
│   ├── healthy_canopy/
│   ├── bark_beetle_damage/
│   └── defoliation/
└── other/
    └── not_relevant/           random photos: people, cars, rooms, sky...
```

Rules that matter more than the code:

- **Class names must be unique across domains** (no `soil/healthy` and `forest/healthy`; use `healthy_canopy`). The app shows the class name, and LeafCheck looks up advice by it.
- **Use at least 200 photos per class**; 500+ is better. Classes with fewer than 100 training photos are flagged.
- **Always include `other/not_relevant`.** A classifier always picks one of its classes, so without this it will call a photo of your shoe "tomato_late_blight".
- **Mix lab and field photos.** PlantVillage photos are leaves on plain backgrounds. A model trained only on those does worse in a real field. Add real phone photos, like the ones collected by the LeafCheck server in `scans.db`.
- **Label soil and forest by what's visible in a photo.** "Clay, dry, cracked" can be seen. pH and nutrients can't, so don't make classes for them.

### Where to get photos

| Domain | Source |
|---|---|
| Plant disease | **PlantVillage** on Kaggle (lab photos, 38 classes). Search Kaggle for **PlantDoc** too: real in-field photos of many of the same diseases. |
| Soil, forest | Mostly your own phone photos. Search Kaggle for soil-type and forest-health datasets, and check each one's license and photo style before mixing it in. |
| Not relevant | Any mix of everyday photos. |

### Build the dataset

From your own `raw/` folder:

```bash
python prepare_data.py --source raw --out data
```

From PlantVillage, with a map file that renames its folders to your classes:

```bash
# LeafCheck's 8 classes:
python prepare_data.py --source ~/datasets/PlantVillage/color --map maps/plantvillage_8.json \
    --out data --max-per-class 1500
# or all 38 PlantVillage classes:
python prepare_data.py --source ~/datasets/PlantVillage/color --map maps/plantvillage_all.json \
    --out data --max-per-class 1500
```

You can run it several times into the same `data/` folder (PlantVillage, then
your `raw/`). It removes duplicate photos, fixes rotation, shrinks images, and
splits each class 80% train / 10% val / 10% test. The **test** photos are never
used in training, so the final score is honest.

---

## 2. Train

```bash
python train.py --data data --name v1 --workers 24
```

`--workers 24` uses your Threadripper's cores to load images; the GPU waits on
the CPU otherwise. Training stops by itself when the validation score stops
improving. Results go to `runs/v1/`.

At the end it prints:

```
Test accuracy: 94.2% (balanced 93.8%) on 2,450 unseen images
Weakest classes: soil/silty 71%, plant/tomato_early_blight 86%, ...
Confidence threshold -> share answered / accuracy:
   >= 0.6: 92% answered, 97.1% correct
   >= 0.8: 81% answered, 99.0% correct
```

(Example numbers.) Use them like this:

- **Weakest classes** need more or better photos.
- **`confusion.csv`** (open it in a spreadsheet) shows which classes get mixed up. For example, early vs. late blight confused often means you need closer photos of each.
- **The threshold table** sets the app's "Not sure, retake" cutoff. Pick the lowest threshold whose accuracy you're comfortable with. Confidence is calibrated, so "80% sure" really is right about 80% of the time.

Useful options:

| Option | When |
|---|---|
| `--model efficientnet_b0` | A bit more accurate, a bit slower on phones |
| `--model mobilenetv4_conv_medium` | Newer phone-optimized model |
| `--img 256` | Small details (tiny spots) matter; slightly slower |
| `--epochs 50 --patience 10` | Big dataset, train longer |

---

## 3. Export for phones

```bash
python export.py --run runs/v1 --app-name "Field Check"
```

`exports/v1/` is now a **complete offline phone app**: a static website you can
host anywhere. The phone runs the model itself; no server does any work.

| File | What |
|---|---|
| `index.html` | The app: check a photo, see the result and alternatives, recent history |
| `model.onnx` | The model, with confidence calibration built in |
| `labels.json` | Class names, image settings, and the "Not sure" `threshold` |
| `sw.js` | Service worker: saves the whole app on the phone at first visit |
| `manifest.json`, `icons/` | Name and icon for "Add to Home Screen" |
| `predict.js` | Model code (also a drop-in for LeafCheck) |
| `vendor/` | ONNX Runtime Web, the in-browser engine (downloaded automatically) |
| `test.html` | Bare test page |
| `advice.json` | Optional: what to do for each result (see below) |

Export also re-checks accuracy using the exported file, so you know the phone
version gives the same answers. To rebuild just the app part (new name, new
threshold, edited advice), run `python make_app.py exports/v1 --name "..." --threshold 0.7`.

### Try it on Atlas

```bash
cd exports/v1
python3 -m http.server 8000
```

Open `http://localhost:8000/` on Atlas. Phones on the same network can use
`http://<atlas-ip>:8000/`, but **offline install only works over https**
(browsers only allow it on `localhost` or https). For phones, use GitHub Pages.

### Install on phones (GitHub Pages: free https hosting)

1. On github.com, create a new **public** repository, for example `field-check`.
2. Push the app folder's contents into it:
   ```bash
   cp -r exports/v1 ~/field-check-site && cd ~/field-check-site
   git init -b main && git add . && git commit -m "Field Check app"
   git remote add origin https://github.com/<your-username>/field-check.git
   git push -u origin main
   ```
   (Git asks for your GitHub login. Use a personal access token, `gh auth login`, or GitHub Desktop.)
3. On GitHub: **Settings → Pages → Deploy from a branch → main → / (root) → Save**.
   After a minute or two the app is at `https://<your-username>.github.io/field-check/`.
4. On each phone, **with internet**, open that address once. Wait until the header
   says **"works offline ✓"** (it's saving about 30 MB).
5. Install it: on Android, tap **Install on this phone** or use Chrome's menu,
   then **Add to Home screen**. On iPhone, open it in Safari, tap **Share**, then **Add to Home Screen**.
6. Test: turn on **airplane mode**, open the app from the home screen, check a photo.

**Publishing a better model later:** export it, copy the new files over the old
ones, then commit and push:
```bash
rsync -a --delete --exclude .git exports/v2/ ~/field-check-site/
cd ~/field-check-site && git add -A && git commit -m "model v2" && git push
```
`sw.js` gets a new version number automatically whenever any file changes, so
each phone downloads the new model the next time it opens the app with internet.

### Advice under each result (optional)

Put an `advice.json` in the app folder, keyed by class label, then rerun `make_app.py`:
```json
{
  "tomato_late_blight": {
    "signs": "Dark, greasy-looking patches that spread fast.",
    "do_now": ["Remove and bag affected plants.", "Check nearby tomato and potato plants today."],
    "prevent": ["Water at the base, not the leaves."],
    "call_expert_if": "It is on more than a few plants."
  }
}
```
The LeafCheck expert server's `advice.json` already has cards for the 8 LeafCheck
labels in this format. Copy them over with:
```bash
python3 -c "import json; a=json.load(open('PATH/leafcheck-server/advice.json')); json.dump(a['plants'], open('exports/v1/advice.json','w'), indent=2)"
python make_app.py exports/v1
```
Every card should be checked against an extension-service source before farmers rely on it.

---

## 4. Put it in LeafCheck (offline)

`predict.js` keeps LeafCheck's **Contract 1** exactly: same `loadModel()` and
`predict(img)`, and returns `{ label, confidence, top3 }`, plus `domain` and `id`.
Nothing else in the app has to change. It replaces the Teachable Machine model,
so **check with the lead first**, per the team rules.

1. Copy `model.onnx`, `labels.json`, `predict.js` into LeafCheck's `model/` folder (replacing the Teachable Machine files).
2. Copy the three files from `vendor/` into LeafCheck's `vendor/`.
3. In `index.html`, replace the two TensorFlow script lines with:
   ```html
   <script src="vendor/ort.wasm.min.js"></script>
   ```
4. In `sw.js`, swap the model and vendor entries in `FILES` for:
   ```js
   "./model/predict.js", "./model/model.onnx", "./model/labels.json",
   "./vendor/ort.wasm.min.js", "./vendor/ort-wasm-simd-threaded.wasm",
   "./vendor/ort-wasm-simd-threaded.mjs",
   ```
   and bump `CACHE` to the next version.
5. Make sure `data/diseases.json` has a card for every plant label, and add cards for soil and forest classes if you show them.
6. Run the airplane-mode test.

---

## 5. Other devices

`model.onnx` runs anywhere ONNX Runtime does:

- **Android / iOS apps:** ONNX Runtime Mobile (`onnxruntime-android`, `onnxruntime-objc`).
- **Raspberry Pi, laptops, field computers:** Python

  ```python
  import json, numpy as np, onnxruntime as ort
  from PIL import Image
  from common import eval_transform

  meta = json.load(open("exports/v1/labels.json"))
  sess = ort.InferenceSession("exports/v1/model.onnx")
  x = eval_transform(meta["img"])(Image.open("leaf.jpg").convert("RGB")).unsqueeze(0).numpy()
  probs = sess.run(None, {"image": x})[0][0]
  best = probs.argmax()
  print(meta["classes"][best]["id"], f"{probs[best]:.0%}")
  ```

Preprocessing must match training: center-square crop of 87.5% of the short side,
resize, ImageNet normalization. `common.py` and `predict.js` both do this.

---

## Files

| File | What it does |
|---|---|
| `prepare_data.py` | Raw photos → deduplicated train/val/test splits |
| `train.py` | Fine-tunes the model; writes `best.pt`, `report.json`, `confusion.csv` |
| `export.py` | `best.pt` → `model.onnx` + `labels.json`; verifies accuracy |
| `common.py` | Dataset, augmentation, preprocessing shared by all scripts |
| `web/predict.js` | Offline browser inference (LeafCheck Contract 1) |
| `web/test.html` | Test page |
| `web/index.html`, `web/sw.js` | The offline app page and service worker template |
| `make_app.py` | Turns an export into the installable offline app (run by export.py) |
| `get_ort.sh` | Downloads ONNX Runtime Web for offline use (run automatically) |
| `maps/*.json` | Dataset folder → class name maps (PlantVillage, PlantDoc, Imagenette) |
| `fetch_datasets.sh` | Downloads PlantVillage, PlantDoc and Imagenette |
| `fetch_soil.sh` | Downloads soil texture (and optionally soil type) datasets |
| `advice/*.json` | Starter advice cards to copy into an app folder |
| `try_photos.py` | Runs your own photos through an exported app on the PC: whole photo vs grid, with a score |
| `test_photos/` | Real test photos sorted by correct answer; add your own |
| `taxonomy.json` | Soil, tree and leaf-symptom classes: descriptions and search phrases |
| `collect_web.py` | Downloads candidate photos per class from Openverse |
| `auto_label.py` | Qwen3-VL labels or verifies photos into `raw/` |

## Improving the model

1. Look at `confusion.csv` and add photos for the classes that get mixed up.
2. Add more **real field photos**. This matters more than anything else.
3. Retrain as `--name v2` and compare `report.json` between runs. Keep the better one.
4. Use the threshold table to set the app's "not sure" cutoff.
