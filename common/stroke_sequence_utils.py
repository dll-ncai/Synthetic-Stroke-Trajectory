"""
Utilities for converting full trajectories into ordered pen-down stroke sequences.

Representation
--------------
Each extracted sequence row is:
    [x, y, stroke_state]

where stroke_state is one of:
  - STROKE_CONTINUE: ordinary point inside a stroke
  - STROKE_BREAK:    final point of a stroke when another stroke follows
  - STROKE_EOS:      final point of the final stroke

This preserves:
  - temporal stroke order
  - stroke-local geometry
  - explicit stroke boundaries

while discarding explicit pen-up travel points.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


STROKE_CONTINUE = 0
STROKE_BREAK = 1
STROKE_EOS = 2
STROKE_TOKEN_DIM = 5  # [dx, dy, continue, break, eos]


def stroke_state_to_pen_state(stroke_state: np.ndarray) -> np.ndarray:
    """
    Map flattened stroke labels onto the legacy pen-state convention used by
    trajectory_loss.

    Legacy convention
    -----------------
    0 = pen-down / ordinary in-stroke point
    1 = pen-up / boundary marker

    We mark both STROKE_BREAK and STROKE_EOS as boundary markers. The loss
    later injects its own trailing EOS span, so we only need a binary boundary
    signal here.
    """
    state = np.asarray(stroke_state)
    out = np.zeros(state.shape, dtype=np.int64)
    out[state != STROKE_CONTINUE] = 1
    return out


def extract_pen_down_strokes(traj: np.ndarray):
    """Split a full [x, y, pen_state] trajectory into ordered pen-down strokes."""
    if traj.shape[0] == 0:
        return []

    strokes = []
    current = []
    for row in traj:
        is_down = float(row[2]) < 0.5
        if is_down:
            current.append(row.copy())
        elif current:
            strokes.append(np.asarray(current, dtype=np.float32))
            current = []

    if current:
        strokes.append(np.asarray(current, dtype=np.float32))
    return strokes


def flatten_strokes_to_sequence(strokes):
    """
    Flatten ordered pen-down strokes into a trainable point sequence.

    Returns
    -------
    seq : (T, 3) float32
        Columns are [x, y, stroke_state].
    stroke_lengths : list[int]
        Number of points per extracted stroke.
    """
    if not strokes:
        return np.zeros((0, 3), dtype=np.float32), []

    rows = []
    stroke_lengths = []
    num_strokes = len(strokes)
    for stroke_idx, stroke in enumerate(strokes):
        if stroke.shape[0] == 0:
            continue
        stroke_lengths.append(int(stroke.shape[0]))
        last_row_idx = stroke.shape[0] - 1
        for point_idx, point in enumerate(stroke):
            if point_idx == last_row_idx:
                state = STROKE_EOS if stroke_idx == num_strokes - 1 else STROKE_BREAK
            else:
                state = STROKE_CONTINUE
            rows.append([float(point[0]), float(point[1]), float(state)])

    return np.asarray(rows, dtype=np.float32), stroke_lengths


def extract_ordered_stroke_sequence(traj: np.ndarray):
    """Convenience wrapper: full trajectory -> ordered strokes -> flattened stroke sequence."""
    strokes = extract_pen_down_strokes(traj)
    seq, stroke_lengths = flatten_strokes_to_sequence(strokes)
    return seq, strokes, stroke_lengths


def sequence_to_strokes(seq: np.ndarray):
    """Invert the flattened stroke sequence back into ordered stroke point lists."""
    if seq.shape[0] == 0:
        return []

    strokes = []
    current = []
    for row in seq:
        current.append(np.asarray([row[0], row[1], 0.0], dtype=np.float32))
        state = int(round(float(row[2])))
        if state in (STROKE_BREAK, STROKE_EOS):
            strokes.append(np.asarray(current, dtype=np.float32))
            current = []
    if current:
        strokes.append(np.asarray(current, dtype=np.float32))
    return strokes


def build_stroke_delta_targets(traj, lengths, max_seq_len, device):
    """
    Build delta-space targets for flattened stroke sequences.

    Input traj format
    -----------------
    traj[..., 0:2] = absolute x,y
    traj[..., 2]   = stroke_state {CONTINUE, BREAK, EOS}
    """
    batch_size, raw_steps, _ = traj.shape
    seq_len = min(raw_steps, max_seq_len)

    traj = traj[:, :seq_len, :]
    lengths = lengths.clamp(max=seq_len)

    xy = traj[:, :, :2]
    state = traj[:, :, 2].long()

    delta = torch.zeros_like(xy)
    delta[:, 1:, :] = xy[:, 1:, :] - xy[:, :-1, :]

    state_target = state.clone()
    teacher_onehot = torch.zeros(batch_size, seq_len, 3, dtype=traj.dtype, device=traj.device)
    teacher_onehot.scatter_(2, state_target.unsqueeze(-1), 1.0)

    teacher_tokens = torch.zeros(batch_size, seq_len, STROKE_TOKEN_DIM, dtype=traj.dtype, device=traj.device)
    if seq_len > 1:
        teacher_tokens[:, 1:, :2] = delta[:, :-1, :]
        teacher_tokens[:, 1:, 2:] = teacher_onehot[:, :-1, :]

    return (
        delta.to(device),
        state_target.to(device),
        teacher_tokens.to(device),
        lengths.to(device),
    )


def collate_stroke_sequence_batch(batch):
    images, trajectories, lengths, metas = zip(*batch)
    images = torch.stack(images, dim=0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    max_len = int(lengths.max().item()) if len(lengths) > 0 else 0
    padded = torch.zeros(len(trajectories), max_len, 3, dtype=torch.float32)
    for idx, traj in enumerate(trajectories):
        padded[idx, : traj.size(0)] = traj
    return images, padded, lengths, list(metas)


class OrderedStrokeDataset(Dataset):
    """Dataset wrapper that converts full processed trajectories into ordered stroke sequences."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, traj, _, meta = self.base_dataset[idx]
        stroke_seq, strokes, stroke_lengths = extract_ordered_stroke_sequence(traj.numpy())
        stroke_tensor = torch.from_numpy(stroke_seq).float()

        out_meta = dict(meta)
        out_meta["stroke_meta"] = {
            "num_strokes": len(strokes),
            "stroke_lengths": stroke_lengths,
            "flattened_length": int(stroke_tensor.size(0)),
        }
        return image, stroke_tensor, int(stroke_tensor.size(0)), out_meta
