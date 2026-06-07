"""Dataset scanning, cleaning, stratified splitting, transforms and loaders.

The "preprocessing" here is the IMAGE analogue of tabular cleaning:
  * corrupt/unreadable image handling  <-> missing-value handling
  * exact-duplicate removal            <-> duplicate-row removal
  * class-weight / imbalance handling  <-> class imbalance handling
  * resize + pixel rescale/standardize <-> feature scaling / normalization
  * train-only augmentation            <-> (image-specific) regularization
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import config as C


# ---------------------------------------------------------------------------
# Scanning & cleaning
# ---------------------------------------------------------------------------
@dataclass
class FileRecord:
    path: Path
    label: str
    ok: bool          # readable & not corrupt
    width: int
    height: int
    mode: str         # PIL mode, e.g. "RGB", "L"
    md5: str


def list_class_dirs(data_dir: Path = C.DATA_DIR) -> list[str]:
    """Detect class folders (one folder per emotion)."""
    return sorted(p.name for p in Path(data_dir).iterdir() if p.is_dir())


def _inspect_file(path: Path, label: str) -> FileRecord:
    """Read one file: compute hash, verify it decodes, capture size/mode."""
    try:
        raw = path.read_bytes()
        md5 = hashlib.md5(raw).hexdigest()
    except OSError:
        return FileRecord(path, label, False, 0, 0, "?", "")
    try:
        # verify() detects truncated/corrupt files without full decode...
        with Image.open(path) as im:
            im.verify()
        # ...but verify() leaves the file unusable, so reopen to read metadata.
        with Image.open(path) as im:
            w, h = im.size
            mode = im.mode
        return FileRecord(path, label, True, w, h, mode, md5)
    except (OSError, SyntaxError, Image.DecompressionBombError):
        return FileRecord(path, label, False, 0, 0, "?", md5)


def scan_dataset(data_dir: Path = C.DATA_DIR,
                 max_workers: int = 8) -> list[FileRecord]:
    """Scan every image, validating it and capturing metadata (threaded)."""
    data_dir = Path(data_dir)
    jobs: list[tuple[Path, str]] = []
    for cls in list_class_dirs(data_dir):
        for p in (data_dir / cls).iterdir():
            if p.suffix.lower() in C.VALID_EXTS:
                jobs.append((p, cls))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        records = list(ex.map(lambda a: _inspect_file(*a), jobs))
    return records


def clean_records(records: list[FileRecord]) -> tuple[list[FileRecord],
                                                      list[FileRecord],
                                                      list[FileRecord]]:
    """Split records into (good, corrupt, duplicate).

    A duplicate is any file whose md5 matches one already kept (we keep the
    first occurrence). Corrupt files are dropped outright.
    """
    good: list[FileRecord] = []
    corrupt: list[FileRecord] = []
    duplicate: list[FileRecord] = []
    seen: set[str] = set()
    for r in records:
        if not r.ok:
            corrupt.append(r)
            continue
        if r.md5 in seen:
            duplicate.append(r)
            continue
        seen.add(r.md5)
        good.append(r)
    return good, corrupt, duplicate


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------
def stratified_split(records: list[FileRecord],
                     classes: list[str],
                     split: tuple[float, float, float] = C.SPLIT,
                     seed: int = C.SEED):
    """Return (train, val, test) record lists, stratified per class."""
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for cls in classes:
        items = [r for r in records if r.label == cls]
        idx = rng.permutation(len(items))
        n = len(items)
        n_tr = int(round(split[0] * n))
        n_va = int(round(split[1] * n))
        tr_i, va_i, te_i = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
        train += [items[i] for i in tr_i]
        val += [items[i] for i in va_i]
        test += [items[i] for i in te_i]
    return train, val, test


def class_weights(records: list[FileRecord],
                  classes: list[str]) -> torch.Tensor:
    """Inverse-frequency class weights (normalized to mean 1) for the loss."""
    counts = np.array([sum(r.label == c for r in records) for c in classes],
                      dtype=np.float64)
    counts = np.clip(counts, 1, None)
    w = counts.sum() / (len(classes) * counts)
    return torch.tensor(w, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Transforms & Dataset
# ---------------------------------------------------------------------------
def build_transforms(train: bool, augment: bool,
                     normalize: str = "imagenet") -> Callable:
    """Compose the preprocessing pipeline.

    normalize:
      "imagenet" -> standardize with ImageNet mean/std (transfer-learning;
                    the PyTorch analogue of Keras preprocess_input).
      "unit"     -> keep pixels in [0,1] (ToTensor only) for the scratch CNN.
    """
    ops: list = [transforms.Resize((C.IMG_SIZE, C.IMG_SIZE))]
    if train and augment:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(C.IMG_SIZE, scale=(0.85, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    ops.append(transforms.ToTensor())  # -> float32 in [0,1]
    if normalize == "imagenet":
        ops.append(transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD))
    return transforms.Compose(ops)


class EmotionDataset(Dataset):
    """Loads RGB images from a list of FileRecords."""

    def __init__(self, records: list[FileRecord], classes: list[str],
                 transform: Callable):
        self.records = records
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        r = self.records[i]
        img = Image.open(r.path).convert("RGB")  # force 3 channels
        return self.transform(img), self.class_to_idx[r.label]


def make_loaders(train_recs, val_recs, test_recs, classes,
                 augment: bool, normalize: str = "imagenet",
                 batch_size: int = C.BATCH_SIZE,
                 num_workers: int = C.NUM_WORKERS):
    """Build train/val/test DataLoaders."""
    tf_train = build_transforms(True, augment, normalize)
    tf_eval = build_transforms(False, False, normalize)
    pin = C.DEVICE.type == "cuda"
    train_ld = DataLoader(EmotionDataset(train_recs, classes, tf_train),
                          batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin,
                          drop_last=True, persistent_workers=num_workers > 0)
    val_ld = DataLoader(EmotionDataset(val_recs, classes, tf_eval),
                        batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin,
                        persistent_workers=num_workers > 0)
    test_ld = DataLoader(EmotionDataset(test_recs, classes, tf_eval),
                         batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin,
                         persistent_workers=num_workers > 0)
    return train_ld, val_ld, test_ld
