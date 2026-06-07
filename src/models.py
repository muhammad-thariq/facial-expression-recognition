"""Model factory: a from-scratch CNN and two transfer-learning backbones.

build_model(name) returns an nn.Module with `num_classes` outputs (raw logits;
CrossEntropyLoss applies softmax internally, the PyTorch equivalent of
"categorical cross-entropy").
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from . import config as C


# ---------------------------------------------------------------------------
# A) Custom CNN from scratch  (>=3 conv blocks + BatchNorm + Dropout)
# ---------------------------------------------------------------------------
class CustomCNN(nn.Module):
    def __init__(self, num_classes: int, in_ch: int = 3):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_ch, 32),   # 224 -> 112
            block(32, 64),      # 112 -> 56
            block(64, 128),     # 56  -> 28
            block(128, 256),    # 28  -> 14
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


# ---------------------------------------------------------------------------
# B/C) Transfer-learning backbones
# ---------------------------------------------------------------------------
def _mobilenet_v2(num_classes: int) -> nn.Module:
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, num_classes)
    return m


def _resnet50(num_classes: int) -> nn.Module:
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


_BUILDERS = {
    "cnn": lambda n: CustomCNN(n),
    "mobilenet_v2": _mobilenet_v2,
    "resnet50": _resnet50,
}

# Which normalization each model expects (see data.build_transforms).
NORMALIZE = {"cnn": "unit", "mobilenet_v2": "imagenet", "resnet50": "imagenet"}


def build_model(name: str, num_classes: int) -> nn.Module:
    if name not in _BUILDERS:
        raise ValueError(f"unknown model '{name}', choose from {list(_BUILDERS)}")
    return _BUILDERS[name](num_classes).to(C.DEVICE)


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers for the two-phase transfer-learning recipe
# ---------------------------------------------------------------------------
def set_backbone_trainable(model: nn.Module, name: str, trainable: bool) -> None:
    """Freeze (or unfreeze) every parameter except the classification head."""
    if name == "cnn":
        return  # trained fully from scratch; nothing to freeze
    head_params = set()
    if name == "mobilenet_v2":
        head_params = set(model.classifier.parameters())
    elif name == "resnet50":
        head_params = set(model.fc.parameters())
    for p in model.parameters():
        if p not in head_params:
            p.requires_grad = trainable


def trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]
