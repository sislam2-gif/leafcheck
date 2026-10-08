"""
Turn a trained model into a phone-ready offline package.

    python export.py --run runs/v1

Creates exports/v1/, a complete offline phone app (see make_app.py):
    model.onnx     the model (softmax + calibration built in), one file
    labels.json    class names and preprocessing settings
    index.html     the app; sw.js + manifest.json + icons/ make it installable and offline
    predict.js     also a drop-in for LeafCheck's model/predict.js (same loadModel/predict contract)
    vendor/        ONNX Runtime Web (downloaded automatically if missing)

It also re-checks accuracy on data/test with ONNX Runtime, so you know the
exported file gives the same answers as the PyTorch model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from common import FolderDataset, eval_transform, make_model
from make_app import build_app


class Deployable(torch.nn.Module):
    """Model + temperature + softmax, so the phone just reads probabilities."""
    def __init__(self, model, temperature):
        super().__init__()
        self.model = model
        self.register_buffer("t", torch.tensor(float(temperature)))

    def forward(self, x):
        return (self.model(x) / self.t).softmax(dim=1)


def export_onnx(net, img, path: Path):
    dummy = torch.randn(1, 3, img, img)
    common = dict(input_names=["image"], output_names=["probs"],
                  dynamic_axes={"image": {0: "batch"}, "probs": {0: "batch"}})
    try:   # classic exporter: most compatible with ONNX Runtime Web
        torch.onnx.export(net, dummy, str(path), opset_version=17, dynamo=False, **common)
    except Exception as e:
        print(f"Classic exporter failed ({e.__class__.__name__}), trying the new one...")
        torch.onnx.export(net, dummy, str(path), opset_version=18, dynamo=True, **common)
    # keep weights inside the one file (browsers can't easily load a separate .data file)
    m = onnx.load(str(path), load_external_data=True)
    for extra in path.parent.glob(path.name + ".data"):
        extra.unlink()
    onnx.save(m, str(path), save_as_external_data=False)
    onnx.checker.check_model(str(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="training run folder, e.g. runs/v1")
    ap.add_argument("--data", default="data", type=Path, help="dataset used for the accuracy re-check")
    ap.add_argument("--out", type=Path, help="default: exports/<run name>")
    ap.add_argument("--check", type=int, default=2000, help="max test images to re-check (0 = skip)")
    ap.add_argument("--app-name", default="Field Check", help="name of the phone app")
    ap.add_argument("--threshold", type=float, help="'Not sure' below this confidence (default 0.6; see runs/<name>/report.json)")
    args = ap.parse_args()

    ck = torch.load(args.run / "best.pt", map_location="cpu")
    classes, img = ck["classes"], ck["img"]
    temp = ck.get("temperature", 1.0)
    out = args.out or Path("exports") / args.run.name
    out.mkdir(parents=True, exist_ok=True)

    model = make_model(ck["model"], len(classes), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    net = Deployable(model, temp).eval()

    onnx_path = out / "model.onnx"
    export_onnx(net, img, onnx_path)
    size_mb = onnx_path.stat().st_size / 1e6

    labels = {
        "model": ck["model"], "img": img, "crop_frac": ck["crop_frac"],
        "mean": list(ck["mean"]), "std": list(ck["std"]), "temperature": temp,
        "classes": [{"id": c, "domain": c.split("/")[0], "label": c.split("/")[1]} for c in classes],
    }
    (out / "labels.json").write_text(json.dumps(labels, indent=2))

    print(f"Exported {onnx_path} ({size_mb:.1f} MB, {len(classes)} classes)")

    # ---- does the ONNX file give the same answers as PyTorch?
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, img, img)
    with torch.no_grad():
        ref = net(x).numpy()
    got = sess.run(None, {"image": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    print(f"PyTorch vs ONNX max difference: {diff:.2e} {'(ok)' if diff < 1e-3 else '(CHECK THIS)'}")

    if args.check and (args.data / "test").exists():
        ds = FolderDataset(args.data / "test", classes, eval_transform(img))
        idx = np.random.default_rng(0).permutation(len(ds))[: args.check]
        correct = 0
        for i in idx:
            t, y = ds[int(i)]
            p = sess.run(None, {"image": t.unsqueeze(0).numpy()})[0][0]
            correct += int(p.argmax() == y)
        print(f"ONNX accuracy on {len(idx)} test images: {correct / len(idx):.1%}")

    print()
    build_app(out, args.app_name, args.threshold)
    print(f"\nTry it:  cd {out} && python3 -m http.server 8000   -> http://localhost:8000/")
    print("Install on phones: see 'Install on phones' in README.md")


if __name__ == "__main__":
    main()
