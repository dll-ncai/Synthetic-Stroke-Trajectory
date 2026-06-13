"""
Phase 10 - small U-Net for stroke instance segmentation.

Input:  (B, 1, H, W) grayscale image, H=64, W=1600
Output: (B, 3, H, W) logits for {background, stroke-interior, stroke-boundary}

The strided-down + transposed-up topology gives a 4-stage encoder/decoder
with skip connections. Channels: 16 -> 32 -> 64 -> 128 -> 256 (bottleneck).
Approximately 1.9M parameters - fits on CPU and trains in a few hours on
this dataset size.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Conv-BN-ReLU x 2 - the standard U-Net block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool 2x2 -> DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Up(nn.Module):
    """Upsample 2x2 -> concat with skip -> DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if odd-size mismatch
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        if diff_h != 0 or diff_w != 0:
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class StrokeUNet(nn.Module):
    def __init__(self, n_classes: int = 3, base_ch: int = 16):
        super().__init__()
        c1, c2, c3, c4, c5 = (base_ch, base_ch * 2, base_ch * 4,
                              base_ch * 8, base_ch * 16)
        self.inc = DoubleConv(1, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)
        self.up1 = Up(c5, c4, c4)
        self.up2 = Up(c4, c3, c3)
        self.up3 = Up(c3, c2, c2)
        self.up4 = Up(c2, c1, c1)
        self.outc = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y = self.up1(x5, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)
        return self.outc(y)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = StrokeUNet(n_classes=3, base_ch=16)
    print(f"Parameters: {count_parameters(m):,}")
    x = torch.zeros(1, 1, 64, 1600)
    with torch.no_grad():
        y = m(x)
    print(f"Input  : {tuple(x.shape)}")
    print(f"Output : {tuple(y.shape)}")
