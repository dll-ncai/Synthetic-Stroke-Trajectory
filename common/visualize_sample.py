"""
Phase 10 - sanity check the segmentation dataset.

Renders a few samples showing:
    row 1: input image (what the U-Net will see)
    row 2: instance mask (each stroke a different colour - what we want the
                          U-Net to recover, modulo the integer labels)
    row 3: semantic mask (bg / interior / boundary - what the U-Net directly
                          predicts as 3-channel softmax)
"""

from __future__ import annotations
import config

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from stroke_seg_dataset import StrokeSegDataset, build_canvas_traj_dataset


def colorize_instance(instance_mask: np.ndarray) -> np.ndarray:
    """Map integer instance labels to an RGB image with a random palette."""
    h, w = instance_mask.shape
    n_inst = int(instance_mask.max())
    rng = np.random.RandomState(0)
    palette = rng.uniform(0.2, 1.0, size=(n_inst + 1, 3))
    palette[0] = [1.0, 1.0, 1.0]  # background = white
    flat = instance_mask.reshape(-1)
    rgb = palette[flat].reshape(h, w, 3)
    return rgb


def colorize_semantic(semantic: np.ndarray) -> np.ndarray:
    """3-class semantic mask -> RGB. 0=white, 1=black (interior), 2=red (boundary)."""
    h, w = semantic.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)
    rgb[semantic == 1] = [0.0, 0.0, 0.0]
    rgb[semantic == 2] = [1.0, 0.2, 0.2]
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default=config.MASTER_RAW_CSV)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--augment", action="store_true",
                        help="apply offline-style augmentation (default off for clarity)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str,
                        default=str(THIS_DIR / "sample_sanity.png"))
    args = parser.parse_args()

    base = build_canvas_traj_dataset(args.csv)
    ds = StrokeSegDataset(base, augment=args.augment, seed=args.seed)

    rng = np.random.RandomState(args.seed)
    picks = rng.choice(len(ds), size=min(args.n_samples, len(ds)), replace=False)

    n = len(picks)
    fig, axes = plt.subplots(3 * n, 1, figsize=(15, 1.4 * 3 * n))
    if n == 1:
        axes = np.array(axes).reshape(-1)

    for row_i, idx in enumerate(picks):
        s = ds[int(idx)]
        img = s["image"].squeeze(0).numpy()
        inst = s["instance_mask"].numpy()
        sem = s["semantic"].numpy()
        n_str = s["n_strokes"]

        ax_img = axes[row_i * 3 + 0]
        ax_img.imshow(img, cmap="gray_r", aspect="auto")
        ax_img.set_title(f"sample {idx} - input image ({n_str} strokes)",
                         fontsize=9)
        ax_img.set_xticks([]); ax_img.set_yticks([])

        ax_inst = axes[row_i * 3 + 1]
        ax_inst.imshow(colorize_instance(inst), aspect="auto")
        ax_inst.set_title("instance mask (per-stroke colour - the ground truth)",
                          fontsize=9)
        ax_inst.set_xticks([]); ax_inst.set_yticks([])

        ax_sem = axes[row_i * 3 + 2]
        ax_sem.imshow(colorize_semantic(sem), aspect="auto")
        ax_sem.set_title("semantic mask: black=interior, red=boundary (U-Net target)",
                         fontsize=9)
        ax_sem.set_xticks([]); ax_sem.set_yticks([])

    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {args.out}")


if __name__ == "__main__":
    main()
