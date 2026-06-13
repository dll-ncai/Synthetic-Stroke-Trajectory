"""
Phase 11 - convert per-line trajectories into the recognizer input format.

Input formats supported:
  A. ONLINE data: (T_total, 3) array of [x, y, pen_state] from
     TrajectoryDataset (already has explicit pen states).
  B. PATH B output: list of per-stroke sub-paths (each (T_i, 2)) from
     the U-Net + skeleton pipeline.

Output (matches the proven ICDAR-paper format):
    (T_out, 3) float32 with columns [x_zscore, y_zscore, pen_state]
    - x, y: z-score normalized per LINE (mean 0, std 1)
    - pen_state: 0 inside a stroke, 1 at the LAST point of each stroke
"""

from __future__ import annotations

import math
import numpy as np
from typing import List


def _zscore_xy(xy: np.ndarray) -> np.ndarray:
    """Per-line mean-std normalization. Matches the working ICDAR script."""
    xs = xy[:, 0]; ys = xy[:, 1]
    mean_x = float(xs.mean()); mean_y = float(ys.mean())
    var_x = float(((xs - mean_x) ** 2).mean())
    var_y = float(((ys - mean_y) ** 2).mean())
    std_x = math.sqrt(var_x) if var_x > 1e-12 else 1.0
    std_y = math.sqrt(var_y) if var_y > 1e-12 else 1.0
    out = np.empty_like(xy, dtype=np.float32)
    out[:, 0] = (xs - mean_x) / std_x
    out[:, 1] = (ys - mean_y) / std_y
    return out


# ----------------------------------------------------------------------
# A.  ONLINE-data path
# ----------------------------------------------------------------------

def online_traj_to_tokens(traj: np.ndarray) -> np.ndarray:
    """traj: (T, 3) [x, y, pen_state], 0 = pen-down, 1 = pen-up between strokes.

    We KEEP only pen-down rows, then mark the LAST point of each stroke
    with pen_state=1.
    """
    if traj.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    pen_down_rows = []
    stroke_lens = []
    current = []
    for row in traj:
        x, y, ps = float(row[0]), float(row[1]), float(row[2])
        if ps < 0.5:
            current.append([x, y])
        else:
            if current:
                pen_down_rows.extend(current)
                stroke_lens.append(len(current))
                current = []
    if current:
        pen_down_rows.extend(current)
        stroke_lens.append(len(current))

    if not pen_down_rows:
        return np.zeros((0, 3), dtype=np.float32)

    xy = np.asarray(pen_down_rows, dtype=np.float32)
    xy_norm = _zscore_xy(xy)
    pen_state = np.zeros(xy_norm.shape[0], dtype=np.float32)
    cursor = 0
    for L in stroke_lens:
        cursor += L
        pen_state[cursor - 1] = 1.0
    return np.column_stack([xy_norm, pen_state]).astype(np.float32)


# ----------------------------------------------------------------------
# B.  PATH-B path
# ----------------------------------------------------------------------

def path_b_polylines_to_tokens(
    polylines_per_stroke: List[List[np.ndarray]],
) -> np.ndarray:
    """Convert Path B's nested output into the same recognizer input format."""
    all_xy = []
    stroke_lens = []
    for sub_paths in polylines_per_stroke:
        stroke_points = []
        for sp in sub_paths:
            if sp.shape[0] == 0:
                continue
            stroke_points.append(sp.astype(np.float32))
        if not stroke_points:
            continue
        merged = np.vstack(stroke_points)
        all_xy.append(merged)
        stroke_lens.append(int(merged.shape[0]))

    if not all_xy:
        return np.zeros((0, 3), dtype=np.float32)

    xy = np.vstack(all_xy)
    xy_norm = _zscore_xy(xy)
    pen_state = np.zeros(xy_norm.shape[0], dtype=np.float32)
    cursor = 0
    for L in stroke_lens:
        cursor += L
        pen_state[cursor - 1] = 1.0
    return np.column_stack([xy_norm, pen_state]).astype(np.float32)


# ----------------------------------------------------------------------
# Light cleanup
# ----------------------------------------------------------------------

def drop_near_duplicates(tokens: np.ndarray,
                          min_step: float = 0.01) -> np.ndarray:
    """Drop consecutive within-stroke points that are essentially identical."""
    if tokens.shape[0] < 2:
        return tokens
    keep = np.ones(tokens.shape[0], dtype=bool)
    for i in range(1, tokens.shape[0]):
        if tokens[i, 2] > 0.5 or tokens[i - 1, 2] > 0.5:
            continue
        d2 = (tokens[i, 0] - tokens[i - 1, 0]) ** 2 + (tokens[i, 1] - tokens[i - 1, 1]) ** 2
        if d2 < min_step ** 2:
            keep[i] = False
    return tokens[keep]


def cap_sequence_length(tokens: np.ndarray, max_T: int = 2000) -> np.ndarray:
    if tokens.shape[0] <= max_T:
        return tokens
    stride = int(math.ceil(tokens.shape[0] / max_T))
    return tokens[::stride]


def simplify_trajectory(tokens: np.ndarray,
                         target_step: float = 0.117,
                         smooth_sigma: float = 0.6) -> np.ndarray:
    """Simplify a trajectory: arc-length resample + Gaussian smooth.

    Used to make ONLINE trajectories look cleaner / more uniform so they
    match the structural simplicity of Path B trajectories. PRESERVES:
        - stroke order
        - within-stroke direction (writer's actual)
        - pen-up boundaries

    REMOVES:
        - tablet sampling jitter (via smoothing)
        - velocity variation (via uniform arc-length resampling)
        - redundant dense points

    `tokens`: (T, 3) [x, y, pen_state]
    `target_step`: target arc-length spacing (post-zscore). 0.117 matches
        the calibrated online median.
    `smooth_sigma`: Gaussian sigma applied to x,y within each stroke.
    """
    if tokens.shape[0] < 3:
        return tokens

    # Split by pen-up markers
    pen_up = np.where(tokens[:, 2] > 0.5)[0].tolist()
    strokes = []
    cursor = 0
    for idx in pen_up:
        strokes.append(tokens[cursor:idx + 1])
        cursor = idx + 1
    if cursor < tokens.shape[0]:
        strokes.append(tokens[cursor:])

    out_parts = []
    for stroke in strokes:
        if stroke.shape[0] < 3:
            out_parts.append(stroke); continue
        xy = stroke[:, :2].astype(np.float32)
        # Step 1: arc-length resample to uniform spacing
        seg = np.diff(xy, axis=0)
        seg_lens = np.linalg.norm(seg, axis=1)
        cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
        total = float(cum[-1])
        if total < target_step * 1.5:
            out_parts.append(stroke); continue
        n_samples = max(3, int(np.floor(total / target_step)) + 1)
        target_d = np.linspace(0.0, total, n_samples)
        new_x = np.interp(target_d, cum, xy[:, 0])
        new_y = np.interp(target_d, cum, xy[:, 1])
        # Step 2: small Gaussian smoothing to remove jitter
        if smooth_sigma > 0 and n_samples >= 3:
            try:
                from scipy.ndimage import gaussian_filter1d
                new_x = gaussian_filter1d(new_x, sigma=smooth_sigma,
                                            mode="nearest")
                new_y = gaussian_filter1d(new_y, sigma=smooth_sigma,
                                            mode="nearest")
            except ImportError:
                pass
        new_pen = np.zeros(n_samples, dtype=np.float32)
        new_pen[-1] = 1.0
        out_parts.append(
            np.column_stack([new_x, new_y, new_pen]).astype(np.float32),
        )
    return np.vstack(out_parts) if out_parts else tokens


def arc_length_resample_per_stroke(tokens: np.ndarray,
                                    target_step: float = 0.05) -> np.ndarray:
    """Resample each stroke to have uniform arc-length spacing.

    `tokens`: (T, 3) [x_zscore, y_zscore, pen_state]
    `target_step`: target arc-length distance between adjacent points,
        in the SAME UNITS as x/y. After z-score normalization the line's
        spread is ~3-4 std units; a target_step of 0.05 gives ~60-80
        points per stroke, matching online density.

    Each pen-up marker is preserved. Output has uniform spacing within
    each stroke.
    """
    if tokens.shape[0] < 3:
        return tokens

    # Split tokens into strokes by pen_state markers (1 = end of stroke)
    pen_up_idx = np.where(tokens[:, 2] > 0.5)[0].tolist()
    strokes = []
    cursor = 0
    for end_idx in pen_up_idx:
        strokes.append(tokens[cursor:end_idx + 1])
        cursor = end_idx + 1
    if cursor < tokens.shape[0]:
        strokes.append(tokens[cursor:])

    out_parts = []
    for stroke in strokes:
        if stroke.shape[0] < 2:
            out_parts.append(stroke); continue
        xy = stroke[:, :2]
        seg = np.diff(xy, axis=0)
        seg_lens = np.linalg.norm(seg, axis=1)
        cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
        total = float(cum[-1])
        if total < target_step * 1.5:
            # Stroke too short to resample meaningfully - keep as is
            out_parts.append(stroke); continue
        # Uniformly sample at target_step intervals
        n_samples = max(2, int(np.floor(total / target_step)) + 1)
        target_d = np.linspace(0.0, total, n_samples)
        new_x = np.interp(target_d, cum, xy[:, 0])
        new_y = np.interp(target_d, cum, xy[:, 1])
        new_pen = np.zeros(n_samples, dtype=np.float32)
        new_pen[-1] = 1.0   # mark stroke boundary at last point
        new_stroke = np.column_stack([new_x, new_y, new_pen]).astype(np.float32)
        out_parts.append(new_stroke)

    return np.vstack(out_parts) if out_parts else tokens


if __name__ == "__main__":
    sample = np.array([
        [10, 20, 0], [11, 21, 0], [12, 22, 0],
        [50, 50, 1],
        [30, 40, 0], [31, 41, 0],
    ], dtype=np.float32)
    out = online_traj_to_tokens(sample)
    print("online_traj_to_tokens:"); print(out)
