"""
generate_evaluation_figures.py

Drop this into your project (same folder as evaluate.py) and run it AFTER
evaluate.py has produced real predictions. It does NOT invent any numbers --
it only plots what you feed it. This is the script that turns your real
evaluate.py output into the four figures your thesis needs:

  1. confusion_matrix.png          -- Development section evidence
  2. per_class_metrics.png         -- Findings RQ1 evidence
  3. accuracy_comparison.png       -- synthetic vs manuscript-only split
  4. confidence_threshold_sweep.png -- justifies your 0.40 threshold choice

HOW TO WIRE THIS UP
--------------------
In evaluate.py, wherever you currently compute accuracy, also collect:
    all_true_labels  -> list[int]   (ground-truth class index for every sample)
    all_pred_labels  -> list[int]   (model's predicted class index)
    all_confidences  -> list[float] (softmax confidence of the predicted class)
    class_names      -> list[str]   (67 entries, index-aligned with your label map)

Then dump them once, e.g.:
    import json
    with open("eval_output.json", "w", encoding="utf-8") as f:
        json.dump({
            "y_true": all_true_labels,
            "y_pred": all_pred_labels,
            "confidences": all_confidences,
            "class_names": class_names,
        }, f, ensure_ascii=False)

Then just run:
    python generate_evaluation_figures.py eval_output.json

If you want the synthetic-vs-manuscript comparison chart, run evaluate.py
twice (once per split) and pass both JSON files:
    python generate_evaluation_figures.py eval_synthetic.json eval_manuscript.json
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_confusion_matrix(y_true, y_pred, class_names, out="confusion_matrix.png", top_n=None):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    if top_n:
        # Show only the top_n most frequent classes so it's readable in a thesis page
        freq = cm.sum(axis=1)
        idx = np.argsort(freq)[::-1][:top_n]
        cm_norm = cm_norm[np.ix_(idx, idx)]
        names = [class_names[i] for i in idx]
    else:
        names = class_names

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.28), max(8, len(names) * 0.28)), dpi=220)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion Matrix (row-normalized)" + (f" -- top {top_n} classes" if top_n else ""))
    plt.colorbar(im, fraction=0.046, pad=0.04, label="proportion of true class")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


def plot_per_class_metrics(y_true, y_pred, class_names, out="per_class_metrics.png"):
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    order = np.argsort(support)[::-1]
    names = [class_names[i] for i in order]
    p, r, f1, support = p[order], r[order], f1[order], support[order]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.3), 5), dpi=220)
    x = np.arange(len(names))
    w = 0.27
    ax.bar(x - w, p, width=w, label="Precision", color="#1f4e79")
    ax.bar(x, r, width=w, label="Recall", color="#4a7c59")
    ax.bar(x + w, f1, width=w, label="F1", color="#8a3d3d")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1 (sorted by support, most-supported first)")
    ax.legend()

    # annotate classes with near-zero support -- these are your ~47 under-covered classes
    for i, s in enumerate(support):
        if s == 0:
            ax.text(i, 0.02, "no data", rotation=90, fontsize=6, color="red", ha="center")

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


def plot_accuracy_comparison(results, out="accuracy_comparison.png"):
    """results: dict like {"Synthetic / augmented": 0.9613, "Manuscript-only": 0.41}"""
    labels = list(results.keys())
    values = list(results.values())
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=220)
    bars = ax.bar(labels, values, color=["#1f4e79", "#8a3d3d"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Accuracy: Synthetic/Augmented Split vs. Real Manuscript Split")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v*100:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


def plot_confidence_sweep(confidences, correct_flags, out="confidence_threshold_sweep.png"):
    """
    confidences: list[float] confidence of the predicted (top-1) class for every sample
    correct_flags: list[bool] whether that prediction was correct
    Sweeps a threshold and shows: accuracy among kept predictions vs. coverage (% kept)
    """
    confidences = np.array(confidences)
    correct_flags = np.array(correct_flags)
    thresholds = np.linspace(0.0, 0.99, 40)
    accs, covs = [], []
    for t in thresholds:
        keep = confidences >= t
        cov = keep.mean()
        acc = correct_flags[keep].mean() if keep.sum() > 0 else np.nan
        accs.append(acc)
        covs.append(cov)

    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=220)
    ax1.plot(thresholds, accs, color="#1f4e79", label="Accuracy of kept predictions")
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Accuracy", color="#1f4e79")
    ax1.axvline(0.40, color="red", linestyle="--", linewidth=1.2, label="chosen threshold (0.40)")

    ax2 = ax1.twinx()
    ax2.plot(thresholds, covs, color="#4a7c59", label="Coverage (% predictions kept)")
    ax2.set_ylabel("Coverage", color="#4a7c59")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)
    ax1.set_title("Confidence Threshold Sweep: Accuracy vs. Coverage Trade-off")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    data = load(sys.argv[1])
    y_true, y_pred = data["y_true"], data["y_pred"]
    class_names = data["class_names"]

    plot_confusion_matrix(y_true, y_pred, class_names, top_n=30)
    plot_per_class_metrics(y_true, y_pred, class_names)

    if "confidences" in data:
        correct = [int(t == p) for t, p in zip(y_true, y_pred)]
        plot_confidence_sweep(data["confidences"], correct)

    if len(sys.argv) == 3:
        data2 = load(sys.argv[2])
        acc1 = np.mean([int(t == p) for t, p in zip(data["y_true"], data["y_pred"])])
        acc2 = np.mean([int(t == p) for t, p in zip(data2["y_true"], data2["y_pred"])])
        plot_accuracy_comparison({"Synthetic / augmented": acc1, "Manuscript-only": acc2})