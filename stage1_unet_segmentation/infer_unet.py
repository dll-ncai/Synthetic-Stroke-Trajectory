"""
Phase 10 - U-Net inference + practical stroke-count evaluation.

The U-Net predicts 3-class semantic. To recover stroke instances:
    1. ink_mask = (interior_prob > thresh_interior)
    2. cut_mask = ink_mask AND NOT (boundary_prob > thresh_boundary)
    3. connected components on cut_mask -> N instances

This module evaluates *practical* stroke separation quality:
    predicted_n vs GT_n on each val line,
    and produces side-by-side viz so we can see if instance counts match.

It can also run on completely external images (e.g., UHWR) with no GT.
"""

from __future__ import annotations
import config

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PARENT_DIR))

from stroke_seg_dataset import (StrokeSegDataset, build_canvas_traj_dataset,
                                  line_level_split_indices)
from unet import StrokeUNet


def predict_semantic(model: StrokeUNet, image: np.ndarray,
                      device: torch.device) -> np.ndarray:
    """Run U-Net on a (H, W) image, return (3, H, W) softmax probs."""
    x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(x)[0]
        probs = F.softmax(logits, dim=0).cpu().numpy()
    return probs


def extract_instances(
    probs: np.ndarray,
    thresh_interior: float = 0.4,
    thresh_boundary: float = 0.3,
    min_area: int = 5,
) -> np.ndarray:
    """Convert (3, H, W) softmax probs into (H, W) instance mask.

    Logic:
        ink = pixels where (interior + boundary) prob > thresh_interior
             i.e., pixels the model thinks belong to a stroke (any kind)
        keep = ink AND NOT (boundary prob > thresh_boundary)
             i.e., drop the predicted boundary pixels to *break* connections
        CC -> instance labels
    """
    interior = probs[1]
    boundary = probs[2]
    ink = (interior + boundary) > thresh_interior
    not_boundary = boundary <= thresh_boundary
    keep = ink & not_boundary
    keep_u8 = keep.astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        keep_u8, connectivity=8,
    )
    out = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        out[labels == i] = next_id
        next_id += 1
    return out


def colorize_instance(instance_mask: np.ndarray) -> np.ndarray:
    h, w = instance_mask.shape
    n_inst = int(instance_mask.max())
    rng = np.random.RandomState(0)
    palette = rng.uniform(0.2, 1.0, size=(n_inst + 1, 3))
    palette[0] = [1.0, 1.0, 1.0]
    flat = instance_mask.reshape(-1)
    return palette[flat].reshape(h, w, 3)


def evaluate_val(
    model: StrokeUNet, val_ds: StrokeSegDataset, val_indices,
    device: torch.device,
    thresh_interior: float = 0.4,
    thresh_boundary: float = 0.3,
    min_area: int = 5,
    n_viz: int = 6,
    out_path: Optional[Path] = None,
) -> dict:
    model.eval()
    gt_counts, pred_counts = [], []
    viz_rows = []

    for vi, ds_idx in enumerate(val_indices):
        s = val_ds[int(ds_idx)]
        img = s["image"].squeeze(0).numpy()
        gt_n = int(s["n_strokes"])

        probs = predict_semantic(model, img, device)
        inst = extract_instances(probs, thresh_interior, thresh_boundary, min_area)
        pred_n = int(inst.max())

        gt_counts.append(gt_n)
        pred_counts.append(pred_n)

        if vi < n_viz:
            viz_rows.append((int(ds_idx), img, s["instance_mask"].numpy(),
                              inst, gt_n, pred_n))

    gt_arr = np.asarray(gt_counts)
    pred_arr = np.asarray(pred_counts)
    abs_err = np.abs(gt_arr - pred_arr)
    rel_err = abs_err / np.maximum(gt_arr, 1)

    print(f"[eval] n_lines={len(gt_arr)}")
    print(f"[eval] mean GT strokes={gt_arr.mean():.1f}  "
          f"mean predicted={pred_arr.mean():.1f}")
    print(f"[eval] mean abs error={abs_err.mean():.1f}  "
          f"median={np.median(abs_err):.1f}  "
          f"P90={np.percentile(abs_err, 90):.1f}")
    print(f"[eval] mean rel error={rel_err.mean()*100:.1f}%  "
          f"within +/-20%: {(rel_err <= 0.2).mean()*100:.1f}%")

    if out_path is not None and len(viz_rows) > 0:
        n = len(viz_rows)
        fig, axes = plt.subplots(3 * n, 1, figsize=(15, 1.3 * 3 * n))
        if 3 * n == 1:
            axes = np.array(axes).reshape(-1)
        for i, (ds_idx, img, gt_inst, pred_inst, gt_n, pred_n) in enumerate(viz_rows):
            ax = axes[i * 3 + 0]
            ax.imshow(img, cmap="gray_r", aspect="auto")
            ax.set_title(f"val[{ds_idx}] input  GT={gt_n}  pred={pred_n}",
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

            ax = axes[i * 3 + 1]
            ax.imshow(colorize_instance(gt_inst), aspect="auto")
            ax.set_title("GT instance mask", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

            ax = axes[i * 3 + 2]
            ax.imshow(colorize_instance(pred_inst), aspect="auto")
            ax.set_title("U-Net predicted instances", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[eval] viz saved to {out_path}")

    return {
        "gt_counts": gt_arr,
        "pred_counts": pred_arr,
        "mean_abs_err": float(abs_err.mean()),
        "mean_rel_err": float(rel_err.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=str(config.UNET_DEPLOY))
    parser.add_argument("--csv", type=str,
                        default=config.MASTER_RAW_CSV)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thresh-interior", type=float, default=0.4)
    parser.add_argument("--thresh-boundary", type=float, default=0.3)
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--n-viz", type=int, default=6)
    parser.add_argument("--out-dir", type=str,
                        default=str(THIS_DIR / "unet_eval"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[eval] device={device}")

    base = build_canvas_traj_dataset(args.csv)
    _, val_idx = line_level_split_indices(
        len(base), val_frac=args.val_frac, seed=args.seed,
    )
    val_ds = StrokeSegDataset(base, augment=False, seed=args.seed)
    print(f"[eval] val lines: {len(val_idx)}")

    model = StrokeUNet(n_classes=3, base_ch=16).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[eval] loaded checkpoint: {args.ckpt}")
    if "best_val" in ckpt:
        print(f"        best_val={ckpt['best_val']:.4f}  epoch={ckpt.get('epoch', '?')}")

    evaluate_val(
        model, val_ds, val_idx, device,
        thresh_interior=args.thresh_interior,
        thresh_boundary=args.thresh_boundary,
        min_area=args.min_area,
        n_viz=args.n_viz,
        out_path=out_dir / "val_eval.png",
    )


if __name__ == "__main__":
    main()
