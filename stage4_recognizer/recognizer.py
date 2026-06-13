"""
Phase 11 - BiLSTM-CTC recognizer (matches the proven ICDAR-paper architecture).

Input:  (B, T, 3) float - [x, y, pen_state] with x, y z-score normalized
Output: (B, T, V) logits over the character vocabulary (CTC blank at index 0)

Architecture:
    Linear(3 -> 64) + LayerNorm + ReLU
    pack_padded_sequence
    BiLSTM(64 -> 128, num_layers=2, dropout=0.2)
    pad_packed_sequence
    Linear(256 -> V)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TrajRecognizer(nn.Module):
    def __init__(self, vocab_size_with_blank: int, input_dim: int = 3):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
        )
        self.encoder = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=True,
        )
        self.proj = nn.Linear(256, vocab_size_with_blank)

    def forward(self, x_pad: torch.Tensor,
                input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_pad: (B, T, 3)
            input_lengths: (B,) - actual sequence lengths
        Returns:
            logits: (B, T, V)
        """
        h = self.embed(x_pad)                                # (B, T, 64)
        packed = pack_padded_sequence(
            h, input_lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        packed_out, _ = self.encoder(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        logits = self.proj(out)                              # (B, T, V)
        return logits


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = TrajRecognizer(vocab_size_with_blank=74, input_dim=3)
    print(f"params: {count_params(m):,}")
    x = torch.randn(4, 500, 3)
    lens = torch.tensor([500, 300, 450, 200])
    y = m(x, lens)
    print(f"input: {tuple(x.shape)}  output: {tuple(y.shape)}")
