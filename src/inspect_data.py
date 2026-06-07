"""STEP 1 — Data inspection. Prints REAL outputs and saves figures.

Run:  python -m src.inspect_data
"""
from __future__ import annotations

import shutil
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from . import config as C
from . import data as D


def quarantine(records, reason: str):
    """Move bad/duplicate files out of the dataset into /quarantine/<reason>."""
    dest_root = C.QUARANTINE_DIR / reason
    moved = 0
    for r in records:
        try:
            dest = dest_root / r.label
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(r.path), str(dest / r.path.name))
            moved += 1
        except OSError:
            pass
    return moved


def class_distribution_chart(good, classes):
    counts = Counter(r.label for r in good)
    vals = [counts[c] for c in classes]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(classes, vals, color="steelblue")
    ax.bar_label(bars)
    ax.set(title="Class distribution (clean images)", xlabel="emotion",
           ylabel="image count")
    fig.tight_layout()
    out = C.FIGURES_DIR / "class_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def sample_montage(good, classes):
    fig, axes = plt.subplots(1, len(classes), figsize=(3 * len(classes), 3))
    if len(classes) == 1:
        axes = [axes]
    for ax, cls in zip(axes, classes):
        rec = next(r for r in good if r.label == cls)
        ax.imshow(Image.open(rec.path).convert("RGB"))
        ax.set_title(f"{cls}\n{rec.width}x{rec.height}")
        ax.axis("off")
    fig.suptitle("One sample per class")
    fig.tight_layout()
    out = C.FIGURES_DIR / "sample_montage.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def size_distribution_chart(good):
    ws = np.array([r.width for r in good])
    hs = np.array([r.height for r in good])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(ws, hs, s=4, alpha=0.3)
    ax.set(title="Image size distribution", xlabel="width (px)",
           ylabel="height (px)")
    fig.tight_layout()
    out = C.FIGURES_DIR / "size_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run(do_quarantine: bool = True):
    print("Scanning dataset (verify + hash + metadata)...")
    records = D.scan_dataset()
    classes = D.list_class_dirs()
    good, corrupt, dup = D.clean_records(records)

    print("\n================ STEP 1: DATA INSPECTION ================")
    print(f"Classes detected ({len(classes)}): {classes}")
    print(f"Total files scanned        : {len(records)}")
    print(f"Readable (non-corrupt)     : {sum(r.ok for r in records)}")
    print(f"Corrupt / unreadable       : {len(corrupt)}")
    print(f"Exact duplicates (md5)     : {len(dup)}")
    print(f"Clean images kept          : {len(good)}")

    print("\nPer-class counts (clean):")
    counts = Counter(r.label for r in good)
    for c in classes:
        print(f"  {c:10s}: {counts[c]}")

    modes = Counter(r.mode for r in good)
    print(f"\nChannel/mode distribution  : {dict(modes)}")
    n_gray = sum(v for k, v in modes.items() if k in ("L", "LA", "1"))
    print(f"  grayscale-ish images     : {n_gray}")
    print(f"  RGB/other                : {len(good) - n_gray}")

    ws = np.array([r.width for r in good])
    hs = np.array([r.height for r in good])
    print(f"\nImage width  : min={ws.min()} max={ws.max()} "
          f"mean={ws.mean():.1f} median={int(np.median(ws))}")
    print(f"Image height : min={hs.min()} max={hs.max()} "
          f"mean={hs.mean():.1f} median={int(np.median(hs))}")
    uniq = Counter(zip(ws.tolist(), hs.tolist()))
    print("Top-5 exact (w,h) sizes    :")
    for (w, h), n in uniq.most_common(5):
        print(f"  {w}x{h}: {n}")

    figs = [
        class_distribution_chart(good, classes),
        sample_montage(good, classes),
        size_distribution_chart(good),
    ]
    print("\nFigures saved:")
    for f in figs:
        print(f"  {f.relative_to(C.ROOT)}")

    if do_quarantine and (corrupt or dup):
        mc = quarantine(corrupt, "corrupt")
        md = quarantine(dup, "duplicate")
        print(f"\nQuarantined {mc} corrupt and {md} duplicate files "
              f"-> {C.QUARANTINE_DIR.relative_to(C.ROOT)}")
    elif not (corrupt or dup):
        print("\nNo corrupt or duplicate files to quarantine.")

    return good, classes, corrupt, dup


if __name__ == "__main__":
    run()
