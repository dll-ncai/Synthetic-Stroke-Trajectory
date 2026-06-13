"""
Phase 5b - scratch CNN classifier (replacement for the frozen Maqsood
encoder + linear head).

Why this exists: the Maqsood encoder is OCR-pretrained on full lines and
treats individual stroke crops as "ink somewhere in a patch" — its features
do not strongly discriminate between similar stroke shapes. Epoch 0 with
the frozen-encoder approach showed `val_top1_noDIAC = 0.8%` (= chance),
confirming the encoder bottleneck.

This is a small VGG-style CNN trained end-to-end on the (1, 64, 256) crops
with cross-entropy over K+1 = 201 classes. ~500K params, no pretraining.

Architecture:
    Conv(1->32, 3x3) + BN + ReLU + MaxPool(2,2)   -> (32, 32, 128)
    Conv(32->64, 3x3) + BN + ReLU + MaxPool(2,2)  -> (64, 16, 64)
    Conv(64->128, 3x3) + BN + ReLU + MaxPool(2,2) -> (128, 8, 32)
    Conv(128->256, 3x3) + BN + ReLU + MaxPool(2,4)-> (256, 4, 8)
    GlobalAvgPool                                  -> (256,)
    Linear(256->256) + ReLU + Dropout
    Linear(256->n_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int, pool=(2, 2)) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool),
    )


class StrokeCNNClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int = 201,
        hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32, pool=(2, 2)),     # (32, 32, 128)
            conv_block(32, 64, pool=(2, 2)),    # (64, 16, 64)
            conv_block(64, 128, pool=(2, 2)),   # (128, 8, 32)
            conv_block(128, 256, pool=(2, 4)),  # (256, 4, 8)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)     # (256, 1, 1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, H, W)  e.g. (B, 1, 64, 256)
        returns: (B, n_classes) logits
        """
        feats = self.features(x)
        pooled = self.pool(feats)
        return self.head(pooled)
