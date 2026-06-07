"""Evaluation: metrics, confusion matrix, training curves, ROC-AUC, results.csv."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from . import config as C


@torch.no_grad()
def predict(model, loader):
    """Return (y_true, y_pred, y_prob) over a loader."""
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        x = x.to(C.DEVICE, non_blocking=True)
        with torch.autocast(device_type=C.DEVICE.type,
                            enabled=C.DEVICE.type == "cuda"):
            logits = model(x)
        prob = F.softmax(logits.float(), dim=1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(prob.argmax(1))
        probs.append(prob)
    return (np.concatenate(ys), np.concatenate(ps), np.concatenate(probs))


def plot_confusion(y_true, y_pred, classes, name, scenario):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"Confusion Matrix — {name} ({scenario})")
    fig.tight_layout()
    out = C.FIGURES_DIR / f"cm_{name}_{scenario}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_curves(history, name, scenario):
    ep = history["epoch"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, history["train_loss"], label="train")
    a1.plot(ep, history["val_loss"], label="val")
    a1.set(title=f"Loss — {name}", xlabel="epoch", ylabel="loss"); a1.legend()
    a2.plot(ep, history["train_acc"], label="train")
    a2.plot(ep, history["val_acc"], label="val")
    a2.set(title=f"Accuracy — {name}", xlabel="epoch", ylabel="acc"); a2.legend()
    fig.tight_layout()
    out = C.FIGURES_DIR / f"curves_{name}_{scenario}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _roc_auc_ovr(y_true, y_prob, n_classes):
    try:
        y_oh = np.eye(n_classes)[y_true]
        return float(roc_auc_score(y_oh, y_prob, average="macro",
                                   multi_class="ovr"))
    except ValueError:
        return float("nan")


def evaluate_model(model, loader, classes, name, scenario="baseline",
                   history=None):
    """Compute all metrics + figures and append a row to results.csv."""
    y_true, y_pred, y_prob = predict(model, loader)
    n = len(classes)
    metrics = {
        "model": name,
        "scenario": scenario,
        "accuracy": float((y_true == y_pred).mean()),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro",
                                                 zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "roc_auc_ovr": _roc_auc_ovr(y_true, y_prob, n),
    }

    report = classification_report(y_true, y_pred, target_names=classes,
                                   zero_division=0)
    print(f"\n=== {name} ({scenario}) ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print(report)
    (C.RESULTS_DIR / f"report_{name}_{scenario}.txt").write_text(report)

    cm_png = plot_confusion(y_true, y_pred, classes, name, scenario)
    curve_png = None
    if history is not None and history.get("epoch"):
        curve_png = plot_curves(history, name, scenario)
    metrics["confusion_png"] = cm_png.name
    metrics["curves_png"] = curve_png.name if curve_png else ""
    append_results(metrics)
    return metrics


def append_results(row: dict, path: Path = C.RESULTS_CSV):
    """Append a metrics row to results.csv (idempotent on header)."""
    fields = ["model", "scenario", "accuracy", "precision_macro",
              "recall_macro", "f1_macro", "roc_auc_ovr",
              "confusion_png", "curves_png"]
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})
