import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent   # ocr_model/
_ROOT = _HERE.parent                       # project root

for _p in [str(_HERE), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ───────────────────────────────────────────────────────────────────

import argparse
import json
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import NewaDataset, get_eval_transform, load_class_map
from model import build_model
from torch.utils.data import DataLoader


def evaluate(
    checkpoint_path: str,
    data_dir: str,
    split: str = "test",
    batch_size: int = 64,
    out_dir: str = "eval_results",
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch        = ckpt.get("arch", "convnet")
    num_classes = int(ckpt.get("num_classes", 67))
    img_size    = int(ckpt.get("img_size", 64))

    # FIX: load_class_map signature is load_class_map(path=str).
    # Pyright was unhappy with Path object — pass str() explicitly.
    class_map_from_ckpt: Optional[dict] = ckpt.get("class_map")
    if class_map_from_ckpt:
        class_map: dict = class_map_from_ckpt
    else:
        class_map_path = str(Path(data_dir) / "class_map.json")
        class_map = load_class_map(class_map_path)

    idx_to_name: Dict[int, str] = {int(v): k for k, v in class_map.items()}

    model = build_model(arch=arch, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = NewaDataset(
        Path(data_dir) / split,
        class_map,
        transform=get_eval_transform(img_size),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds:  List[int]   = []
    all_labels: List[int]   = []
    all_conf:   List[float] = []
    top5_correct = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

            k = min(5, probs.shape[1])
            _, top5 = probs.topk(k, dim=1)
            top5_correct += sum(
                int(labels[i].item()) in top5[i].tolist()
                for i in range(len(labels))
            )

            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_conf.extend(conf.cpu().tolist())

    preds_arr  = np.array(all_preds,  dtype=np.int64)
    labels_arr = np.array(all_labels, dtype=np.int64)
    conf_arr   = np.array(all_conf,   dtype=np.float32)
    n = len(labels_arr)

    top1_acc = float((preds_arr == labels_arr).mean())
    top5_acc = top5_correct / n

    # ── Per-class precision / recall / F1 ──────────────────────────
    per_class: Dict[str, dict] = {}
    for c in range(num_classes):
        tp = int(((preds_arr == c) & (labels_arr == c)).sum())
        fp = int(((preds_arr == c) & (labels_arr != c)).sum())
        fn = int(((preds_arr != c) & (labels_arr == c)).sum())
        support = int((labels_arr == c).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        per_class[idx_to_name.get(c, str(c))] = dict(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            support=support,
        )

    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    micro_f1 = top1_acc

    # ── Confidence calibration ──────────────────────────────────────
    bins = np.linspace(0, 1, 11)
    calibration: List[dict] = []
    for i in range(10):
        lo, hi = float(bins[i]), float(bins[i + 1])
        mask = (conf_arr >= lo) & (conf_arr < hi if i < 9 else conf_arr <= hi)
        if mask.sum() > 0:
            bucket_acc = float((preds_arr[mask] == labels_arr[mask]).mean())
            calibration.append(dict(
                range=f"{lo:.1f}-{hi:.1f}",
                n=int(mask.sum()),
                avg_confidence=round(float(conf_arr[mask].mean()), 3),
                actual_accuracy=round(bucket_acc, 3),
            ))

    zero_support = [name for name, v in per_class.items() if v["support"] == 0]
    weak_classes = sorted(
        [(name, v) for name, v in per_class.items() if v["support"] > 0],
        key=lambda kv: kv[1]["f1"],
    )[:15]

    # ── Confusion matrix heatmap ────────────────────────────────────
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, lb in zip(preds_arr, labels_arr):
        cm[lb, p] += 1
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    names = [idx_to_name.get(i, str(i)) for i in range(num_classes)]
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({split} split, row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(str(out / f"confusion_matrix_{split}.png"), dpi=150)
    plt.close(fig)

    # ── Print summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EVALUATION -- {split} split ({n} samples, {num_classes} classes)")
    print(f"{'='*60}")
    print(f"  Top-1 accuracy : {top1_acc*100:.2f}%")
    print(f"  Top-5 accuracy : {top5_acc*100:.2f}%")
    print(f"  Macro F1       : {macro_f1:.4f}  (treats every class equally)")
    print(f"  Micro F1       : {micro_f1:.4f}  (== top-1 accuracy)")
    print(f"\n  Classes with ZERO test examples ({len(zero_support)}):")
    print(f"    {zero_support if zero_support else 'none'}")
    print(f"\n  15 weakest classes (lowest F1, support > 0):")
    for name, v in weak_classes:
        print(f"    {name:14s} F1={v['f1']:.3f}  precision={v['precision']:.3f}  "
              f"recall={v['recall']:.3f}  support={v['support']}")
    print(f"\n  Confidence calibration:")
    for b in calibration:
        gap  = b["avg_confidence"] - b["actual_accuracy"]
        flag = "  <- overconfident" if gap > 0.15 else ""
        print(f"    conf {b['range']}: n={b['n']:5d}  "
              f"avg_conf={b['avg_confidence']:.2f}  actual_acc={b['actual_accuracy']:.2f}{flag}")

    # ── Save JSON ───────────────────────────────────────────────────
    report = dict(
        split=split,
        n_samples=n,
        num_classes=num_classes,
        top1_accuracy=round(top1_acc, 4),
        top5_accuracy=round(top5_acc, 4),
        macro_f1=round(macro_f1, 4),
        micro_f1=round(micro_f1, 4),
        zero_support_classes=zero_support,
        weakest_classes=[{"class": nm, **v} for nm, v in weak_classes],
        per_class=per_class,
        confidence_calibration=calibration,
    )
    metrics_path = out / f"metrics_{split}.json"
    with open(str(metrics_path), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved -> {metrics_path}")
    print(f"  Saved -> {out}/confusion_matrix_{split}.png")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the Newa OCR model with full metrics")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    p.add_argument("--data",       default="dataset_final")
    p.add_argument("--split",      default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out",        default="eval_results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.checkpoint, args.data, args.split, args.batch_size, args.out)