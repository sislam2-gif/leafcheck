"""
Train a small image classifier that can run offline on phones.

    python train.py --data data --name v1

Defaults to MobileNetV3-Large (about 5M parameters, ~20 MB), pretrained on
ImageNet and fine-tuned on your classes. Output goes to runs/<name>/:

    best.pt          the trained model (use export.py to make the phone version)
    metrics.csv      loss and accuracy per epoch
    report.json      test accuracy per class + accuracy at each confidence threshold
    confusion.csv    which classes get mixed up with which
"""
import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from common import (CROP_FRAC, MEAN, STD, FolderDataset, eval_transform,
                    list_classes, make_model, train_transform)


def get_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", type=Path)
    ap.add_argument("--name", default="v1", help="run name; output in runs/<name>")
    ap.add_argument("--model", default="mobilenetv3_large_100",
                    help="any timm model; phone-friendly: mobilenetv3_large_100, "
                         "efficientnet_b0, mobilenetv4_conv_medium")
    ap.add_argument("--img", type=int, default=224, help="input size in pixels")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=7, help="stop after N epochs without improvement")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--no-pretrained", action="store_true", help="train from scratch (testing only)")
    ap.add_argument("--no-balance", action="store_true", help="don't oversample small classes")
    return ap.parse_args()


@torch.no_grad()
def collect_logits(model, loader, device, amp):
    model.eval()
    out, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out.append(model(x).float().cpu())
        ys.append(y)
    return torch.cat(out), torch.cat(ys)


def balanced_acc(pred, y, n_cls):
    accs = [(pred[y == c] == c).float().mean().item() for c in range(n_cls) if (y == c).any()]
    return sum(accs) / len(accs)


def fit_temperature(logits, y):
    """Temperature scaling: makes '80% sure' actually mean right ~80% of the time."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def step():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(step)
    return float(log_t.detach().exp())


def main():
    args = get_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("No GPU found, training on CPU (slow).")

    classes = list_classes(args.data)
    if len(classes) < 2:
        raise SystemExit(f"Need at least 2 classes in {args.data}/train/<domain>/<class>/")
    n_cls = len(classes)
    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)

    train_ds = FolderDataset(args.data / "train", classes, train_transform(args.img))
    val_ds = FolderDataset(args.data / "val", classes, eval_transform(args.img))
    test_ds = FolderDataset(args.data / "test", classes, eval_transform(args.img))
    print(f"{n_cls} classes | train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

    counts = Counter(train_ds.targets)
    if args.no_balance:
        sampler, shuffle = None, True
    else:   # every class seen about equally often
        w = [1.0 / counts[t] for t in train_ds.targets]
        sampler, shuffle = WeightedRandomSampler(w, len(w), replacement=True), False

    kw = dict(num_workers=args.workers, pin_memory=amp, persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, args.batch, sampler=sampler, shuffle=shuffle, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, args.batch * 2, **kw)
    test_dl = DataLoader(test_ds, args.batch * 2, **kw)

    model = make_model(args.model, n_cls, not args.no_pretrained).to(device)
    model = model.to(memory_format=torch.channels_last)
    print(f"Model {args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(train_dl)
    warm = len(train_dl)   # 1 epoch warmup, then cosine decay
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    meta = {"model": args.model, "classes": classes, "img": args.img,
            "crop_frac": CROP_FRAC, "mean": MEAN, "std": STD}
    best, bad_epochs = -1.0, 0
    with open(out / "metrics.csv", "w", newline="") as fcsv:
        log = csv.writer(fcsv)
        log.writerow(["epoch", "train_loss", "train_acc", "val_acc", "val_balanced_acc", "lr", "seconds"])
        for ep in range(1, args.epochs + 1):
            model.train()
            t0, tot, correct, loss_sum = time.time(), 0, 0, 0.0
            for x, y in train_dl:
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                sched.step()
                loss_sum += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                tot += len(y)

            vl, vy = collect_logits(model, val_dl, device, amp)
            vpred = vl.argmax(1)
            v_acc = (vpred == vy).float().mean().item()
            v_bal = balanced_acc(vpred, vy, n_cls)
            secs = time.time() - t0
            log.writerow([ep, f"{loss_sum / tot:.4f}", f"{correct / tot:.4f}", f"{v_acc:.4f}",
                          f"{v_bal:.4f}", f"{sched.get_last_lr()[0]:.2e}", f"{secs:.0f}"])
            fcsv.flush()
            flag = ""
            if v_bal > best:
                best, bad_epochs, flag = v_bal, 0, "  * best"
                torch.save({**meta, "state_dict": model.state_dict()}, out / "best.pt")
            else:
                bad_epochs += 1
            print(f"epoch {ep:3d} | loss {loss_sum / tot:.3f} | train {correct / tot:.1%} | "
                  f"val {v_acc:.1%} (balanced {v_bal:.1%}) | {secs:.0f}s{flag}")
            if bad_epochs >= args.patience:
                print(f"No improvement for {args.patience} epochs, stopping.")
                break

    # ------------------------------------------------ final test on unseen photos
    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    vl, vy = collect_logits(model, val_dl, device, amp)
    temp = fit_temperature(vl, vy)
    ckpt["temperature"] = temp
    torch.save(ckpt, out / "best.pt")

    tl, ty = collect_logits(model, test_dl, device, amp)
    probs = (tl / temp).softmax(1)
    conf, pred = probs.max(1)
    report = {
        "model": args.model, "classes": n_cls, "test_images": len(ty),
        "temperature": round(temp, 3),
        "test_accuracy": round((pred == ty).float().mean().item(), 4),
        "test_balanced_accuracy": round(balanced_acc(pred, ty, n_cls), 4),
        "per_class": {}, "by_threshold": [],
    }
    for c, name in enumerate(classes):
        m = ty == c
        if m.any():
            report["per_class"][name] = {"n": int(m.sum()), "accuracy": round((pred[m] == c).float().mean().item(), 4)}
    # use this to choose the app's "not sure" threshold
    for th in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        keep = conf >= th
        report["by_threshold"].append({
            "threshold": th,
            "answered": round(keep.float().mean().item(), 4),
            "accuracy_when_answered": round((pred[keep] == ty[keep]).float().mean().item(), 4) if keep.any() else None,
        })
    (out / "report.json").write_text(json.dumps(report, indent=2))

    cm = torch.zeros(n_cls, n_cls, dtype=torch.long)
    for t, p in zip(ty, pred):
        cm[t, p] += 1
    with open(out / "confusion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted"] + classes)
        for i, name in enumerate(classes):
            w.writerow([name] + cm[i].tolist())

    print(f"\nTest accuracy: {report['test_accuracy']:.1%} "
          f"(balanced {report['test_balanced_accuracy']:.1%}) on {len(ty)} unseen images")
    worst = sorted(report["per_class"].items(), key=lambda kv: kv[1]["accuracy"])[:5]
    print("Weakest classes:", ", ".join(f"{k} {v['accuracy']:.0%}" for k, v in worst))
    print("Confidence threshold -> share answered / accuracy:")
    for r in report["by_threshold"]:
        if r["accuracy_when_answered"] is not None:
            print(f"   >= {r['threshold']:.1f}: {r['answered']:.0%} answered, {r['accuracy_when_answered']:.1%} correct")
    print(f"\nSaved to {out}/. Next: python export.py --run {out}")


if __name__ == "__main__":
    main()
