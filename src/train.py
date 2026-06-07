"""Shared training loop with EarlyStopping, best-checkpointing and mixed
precision (torch.amp). Equivalent to a Keras `fit` with EarlyStopping +
ModelCheckpoint callbacks.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config as C
from . import models as M


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.bad = 0
        self.stop = False

    def step(self, metric: float) -> bool:
        improved = (self.best is None or
                    (metric > self.best if self.mode == "max" else metric < self.best))
        if improved:
            self.best, self.bad = metric, 0
        else:
            self.bad += 1
            self.stop = self.bad >= self.patience
        return improved


def _run_epoch(model, loader, criterion, optimizer, scaler, train: bool, desc):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    use_amp = C.DEVICE.type == "cuda"
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(C.DEVICE, non_blocking=True), y.to(C.DEVICE, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=C.DEVICE.type, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def _fit(model, name, train_ld, val_ld, criterion, optimizer,
         epochs, patience, ckpt_path, history, phase):
    scaler = torch.amp.GradScaler(enabled=C.DEVICE.type == "cuda")
    stopper = EarlyStopping(patience, mode="max")
    best_acc = -1.0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(model, train_ld, criterion, optimizer,
                                     scaler, True, f"{name}[{phase}] ep{ep} train")
        va_loss, va_acc = _run_epoch(model, val_ld, criterion, optimizer,
                                     scaler, False, f"{name}[{phase}] ep{ep} val")
        improved = stopper.step(va_acc)
        history["epoch"].append(len(history["epoch"]) + 1)
        history["phase"].append(phase)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(f"  [{name}/{phase}] epoch {ep:2d}/{epochs} | "
              f"train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"val {va_loss:.4f}/{va_acc:.4f} | {time.time()-t0:.0f}s")
        if improved and va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), ckpt_path)  # ModelCheckpoint (best)
        if stopper.stop:
            print(f"  early stopping at epoch {ep} (best val_acc={best_acc:.4f})")
            break
    return best_acc


def train_model(name: str, train_ld: DataLoader, val_ld: DataLoader,
                num_classes: int, class_weights: torch.Tensor | None = None,
                epochs: int = C.EPOCHS, finetune_epochs: int = C.FINETUNE_EPOCHS,
                patience: int = C.EARLY_STOP_PATIENCE):
    """Train one model and return (model, history, ckpt_path).

    Transfer-learning models use the two-phase recipe: train the head with the
    backbone frozen, then unfreeze and fine-tune the whole network at a low LR.
    """
    model = M.build_model(name, num_classes)
    weight = class_weights.to(C.DEVICE) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    ckpt_path = C.CHECKPOINT_DIR / f"{name}_best.pt"
    history = {k: [] for k in ("epoch", "phase", "train_loss", "train_acc",
                               "val_loss", "val_acc")}

    if name == "cnn":
        opt = torch.optim.Adam(model.parameters(), lr=C.LR,
                               weight_decay=C.WEIGHT_DECAY)
        _fit(model, name, train_ld, val_ld, criterion, opt,
             epochs, patience, ckpt_path, history, "scratch")
    else:
        # Phase 1: frozen backbone, train head only.
        M.set_backbone_trainable(model, name, False)
        opt = torch.optim.Adam(M.trainable_parameters(model), lr=C.LR,
                               weight_decay=C.WEIGHT_DECAY)
        _fit(model, name, train_ld, val_ld, criterion, opt,
             epochs, patience, ckpt_path, history, "frozen")
        # Phase 2: unfreeze and fine-tune the whole net at a small LR.
        model.load_state_dict(torch.load(ckpt_path, map_location=C.DEVICE))
        M.set_backbone_trainable(model, name, True)
        opt = torch.optim.Adam(model.parameters(), lr=C.FINETUNE_LR,
                               weight_decay=C.WEIGHT_DECAY)
        _fit(model, name, train_ld, val_ld, criterion, opt,
             finetune_epochs, patience, ckpt_path, history, "finetune")

    # Restore best checkpoint before returning.
    model.load_state_dict(torch.load(ckpt_path, map_location=C.DEVICE))
    hist_path = C.RESULTS_DIR / f"history_{name}.json"
    Path(hist_path).write_text(json.dumps(history, indent=2))
    return model, history, ckpt_path
