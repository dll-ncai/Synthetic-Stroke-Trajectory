"""
Phase 11 - calibrate the target arc-length step size for Path B resampling.

Measures the median within-stroke point spacing in the ONLINE training
trajectories AFTER z-score normalization. That value is what we should
target when resampling Path B trajectories for the recognizer.
"""

from __future__ import annotations
import config

import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(THIS_DIR))

from trajectory_dataset import TrajectoryDataset
from traj_tokenizer import online_traj_to_tokens


def main(csv_path: str, n_samples: int = 200):
    ds = TrajectoryDataset(
        meta_csv_path=csv_path,
        normalize_coords=True,
        normalization_mode="canvas",
    )
    rng = np.random.RandomState(0)
    idx = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    all_steps = []
    for di in idx:
        _, traj_t, traj_len, _ = ds[int(di)]
        tokens = online_traj_to_tokens(traj_t.numpy()[:traj_len])
        if tokens.shape[0] < 3: continue
        # Compute step sizes within strokes (not across pen-ups)
        for i in range(1, tokens.shape[0]):
            if tokens[i, 2] > 0.5 or tokens[i - 1, 2] > 0.5:
                continue
            d = float(np.linalg.norm(tokens[i, :2] - tokens[i - 1, :2]))
            all_steps.append(d)
    arr = np.asarray(all_steps)
    print(f"[calibrate] computed over {len(arr)} within-stroke steps from "
          f"{len(idx)} lines")
    print(f"  mean step size:    {arr.mean():.4f}")
    print(f"  median step size:  {float(np.median(arr)):.4f}")
    print(f"  P25 / P75:         {float(np.percentile(arr, 25)):.4f} / "
          f"{float(np.percentile(arr, 75)):.4f}")
    print()
    print(f"  USE target_step ~ {float(np.median(arr)):.3f}  "
          f"for arc_length_resample_per_stroke()")


if __name__ == "__main__":
    main(config.MAIN_ALL_CSV, n_samples=200)
