"""End-to-end driver: inspect -> split -> train 3 models -> evaluate -> CSV.

Usage:
    python -m src.run_all                # full run (all 3 models, all epochs)
    python -m src.run_all --smoke        # fast pipeline check (subset, 1 epoch)
    python -m src.run_all --models cnn   # train a subset of models
    python -m src.run_all --aug          # enable train-time augmentation

Windows note: the __main__ guard below is required for DataLoader workers.
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from . import config as C
from . import data as D
from . import evaluate as E
from . import inspect_data as I
from . import models as M
from . import train as T

ALL_MODELS = ["cnn", "mobilenet_v2", "resnet50"]


def seed_everything(seed: int = C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subset_per_class(records, classes, k):
    """Keep at most k records per class (for the smoke test)."""
    out = []
    for c in classes:
        out += [r for r in records if r.label == c][:k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="fast pipeline validation on a small subset")
    ap.add_argument("--aug", action="store_true",
                    help="enable train-time augmentation")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS,
                    choices=ALL_MODELS)
    ap.add_argument("--no-quarantine", action="store_true")
    args = ap.parse_args()

    seed_everything()
    print(f"Device: {C.DEVICE}")

    # STEP 1 — inspection (also returns the cleaned record list)
    good, classes, _, _ = I.run(do_quarantine=not args.no_quarantine)

    # Smoke mode: shrink everything so the pipeline runs in ~minutes.
    epochs = 1 if args.smoke else C.EPOCHS
    ft_epochs = 1 if args.smoke else C.FINETUNE_EPOCHS
    if args.smoke:
        good = subset_per_class(good, classes, 200)
        print(f"\n[SMOKE] using {len(good)} images "
              f"({len(good)//len(classes)}/class)")

    # STEP 2 — stratified split + class weights
    train_recs, val_recs, test_recs = D.stratified_split(good, classes)
    print(f"\nSplit: train={len(train_recs)} val={len(val_recs)} "
          f"test={len(test_recs)} (70/15/15, seed={C.SEED})")
    cw = D.class_weights(train_recs, classes)
    print(f"Class weights (imbalance handling): "
          f"{dict(zip(classes, [round(float(x),3) for x in cw]))}")

    scenario = "aug" if args.aug else "baseline"
    for name in args.models:
        norm = M.NORMALIZE[name]
        train_ld, val_ld, test_ld = D.make_loaders(
            train_recs, val_recs, test_recs, classes,
            augment=args.aug, normalize=norm)
        print(f"\n########## TRAIN {name} (norm={norm}, aug={args.aug}) ##########")
        model, history, _ = T.train_model(
            name, train_ld, val_ld, len(classes), class_weights=cw,
            epochs=epochs, finetune_epochs=ft_epochs)
        E.evaluate_model(model, test_ld, classes, name,
                         scenario=scenario, history=history)

    print(f"\nDone. Results -> {C.RESULTS_CSV.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
