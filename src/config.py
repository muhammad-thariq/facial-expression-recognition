"""Central configuration for the Facial Emotion Recognition project.

All paths are resolved relative to the repository root so the code runs the
same way from the notebook, a script, or any working directory.
"""
from __future__ import annotations

from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "human-face-emotions" / "Data"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
QUARANTINE_DIR = ROOT / "quarantine"

for _d in (FIGURES_DIR, RESULTS_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = RESULTS_DIR / "results.csv"

# ---------------------------------------------------------------------------
# Reproducibility / data
# ---------------------------------------------------------------------------
SEED = 42
IMG_SIZE = 224
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# Stratified split fractions (train / val / test)
SPLIT = (0.70, 0.15, 0.15)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
NUM_WORKERS = 2                # Windows-safe (requires __main__ guard)
LR = 1e-3                      # head / from-scratch learning rate
FINETUNE_LR = 1e-5            # learning rate when unfreezing the backbone
WEIGHT_DECAY = 1e-4
EPOCHS = 30                    # max epochs; EarlyStopping usually stops sooner
FINETUNE_EPOCHS = 10
EARLY_STOP_PATIENCE = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet statistics used by torchvision pretrained backbones. These ARE the
# PyTorch equivalent of Keras' per-model `preprocess_input`.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
