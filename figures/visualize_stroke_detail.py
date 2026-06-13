"""
Single-stroke detail figure for the Trajectory Recovery section (thesis 3.4).

The end-to-end overview figure shows input -> 3-class -> instances -> trajectory,
but it hides the geometric sub-steps. This script zooms into ONE stroke / ligature
and shows them side by side:

    (a) blob              -- the binary instance mask (connected component)
    (b) skeleton          -- 1-pixel-wide centerline (Zhang-Suen thinning)
    (c) walked polyline   -- multi-pass walk, arrows show traversal direction
    (d) Gaussian-smoothed -- staircase removed
    (e) arc-length resampled -- points equally spaced along the path

Steps (a)-(d) live in pixel space. Step (e) reproduces the pipeline's resampling
faithfully: the smoothed polyline is z-scored with the *whole-line* mean/std
(as path_b_polylines_to_tokens does), resampled at target_step in that space,
then mapped back to pixels for display.

The recognizer is intentionally NOT loaded -- this figure is about geometry only.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARENT_DIR / "common"))
sys.path.insert(0, str(PARENT_DIR / "stage1_unet_segmentation"))
sys.path.insert(0, str(PARENT_DIR / "stage2_trajectory_recovery"))
sys.path.insert(0, str(PARENT_DIR / "stage3_order_predictor"))
sys.path.insert(0, str(PARENT_DIR / "stage4_recognizer"))
sys.path.insert(0, str(THIS_DIR))

import config

from infer_unet import extract_instances, predict_semantic
from template_stitcher import extract_skeleton
from test_uhwr import preprocess_uhwr_image
from unet import StrokeUNet

# Reuse the helpers already defined (and now fixed) in the overview script.
from visualize_end_to_end_sample import (
    UNET_CKPT,
    load_checkpoint,
    order_skeleton_multi_pass,
    smooth_polyline,
)

DEFAULT_IMAGE = Path(config.TEST_IMAGE_DIR) / "00100.jpg"


def build_components(inst: np.ndarray, min_area: int) -> List[Dict]:
    comps = []
    for k in range(1, int(inst.max()) + 1):
        mask_k = (inst == k).astype(np.uint8)
        area = int(mask_k.sum())
        if area < min_area:
            continue
        ys, xs = np.where(mask_k > 0)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        comps.append({
            "mask": mask_k,
            "bbox": (x0, y0, x1, y1),
            "area": area,
            "extent": (x1 - x0, y1 - y0),
        })
    return comps


def walk_and_smooth(comp: Dict, sigma: float) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Return (skeleton_local, walked_subs_global, smoothed_subs_global)."""
    bx0, by0, bx1, by1 = comp["bbox"]
    blob = comp["mask"][by0:by1, bx0:bx1]
    skel = extract_skeleton(blob)
    walked, smoothed = [], []
    for sp in order_skeleton_multi_pass(skel):
        sp = np.asarray(sp, dtype=np.float32)
        g = sp.copy()
        g[:, 0] += bx0
        g[:, 1] += by0
        walked.append(g)
        s = smooth_polyline(g.copy(), sigma=sigma)
        smoothed.append(s)
    return skel, walked, smoothed


def line_zscore_params(all_smoothed: List[np.ndarray]) -> Tuple[float, float, float, float]:
    """Whole-line mean/std over walked-and-smoothed points (matches _zscore_xy)."""
    xy = np.vstack(all_smoothed)
    mx, my = float(xy[:, 0].mean()), float(xy[:, 1].mean())
    vx = float(((xy[:, 0] - mx) ** 2).mean())
    vy = float(((xy[:, 1] - my) ** 2).mean())
    sx = math.sqrt(vx) if vx > 1e-12 else 1.0
    sy = math.sqrt(vy) if vy > 1e-12 else 1.0
    return mx, my, sx, sy


def _resample_one(xy: np.ndarray, zparams, target_step: float) -> np.ndarray:
    """Z-score (whole-line params) -> uniform arc-length resample -> back to px."""
    mx, my, sx, sy = zparams
    norm = np.empty_like(xy, dtype=np.float32)
    norm[:, 0] = (xy[:, 0] - mx) / sx
    norm[:, 1] = (xy[:, 1] - my) / sy
    cum = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(norm, axis=0), axis=1))))
    total = float(cum[-1])
    if total < target_step * 1.5:
        return xy.astype(np.float32)
    n = max(2, int(np.floor(total / target_step)) + 1)
    td = np.linspace(0.0, total, n)
    out = np.empty((n, 2), dtype=np.float32)
    out[:, 0] = np.interp(td, cum, norm[:, 0]) * sx + mx
    out[:, 1] = np.interp(td, cum, norm[:, 1]) * sy + my
    return out


def arc_length_resample_pixels(smoothed_subs: List[np.ndarray],
                               zparams: Tuple[float, float, float, float],
                               target_step: float,
                               final_sigma: float) -> List[np.ndarray]:
    """Resample EACH sub-path independently so points are evenly spaced along the
    actual path and no spurious points bridge the gaps between sub-paths, then
    apply the light post-resampling smoothing pass (thesis 3.4.6).

    (The recognizer pipeline merges a stroke's sub-paths before resampling, which
    drops a few points across those gaps; for an illustrative figure we keep the
    sub-paths separate.)
    """
    out = []
    for sp in smoothed_subs:
        rp = _resample_one(np.asarray(sp, dtype=np.float32), zparams, target_step)
        rp = smooth_polyline(rp, sigma=final_sigma)  # final light pass (3.4.6)
        out.append(rp)
    return out


def _setup_axis(ax, bbox, pad, title):
    x0, y0, x1, y1 = bbox
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y1 + pad, y0 - pad)  # inverted -> image y-down
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=13, pad=8)


def _draw_arrows(ax, pts, color, n_arrows=5):
    if len(pts) < 4:
        return
    idxs = np.linspace(1, len(pts) - 1, n_arrows).astype(int)
    for i in idxs:
        p0, p1 = pts[i - 1], pts[i]
        ax.annotate(
            "", xy=(p1[0], p1[1]), xytext=(p0[0], p0[1]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, shrinkA=0, shrinkB=0),
        )


def make_detail_figure(out_path: Path, image_name: str, comp: Dict,
                       skel_local: np.ndarray, walked: List[np.ndarray],
                       smoothed: List[np.ndarray], resampled: List[np.ndarray],
                       walk_sigma: float, final_sigma: float,
                       target_step: float) -> None:
    bx0, by0, bx1, by1 = comp["bbox"]
    bbox = comp["bbox"]
    pad = max(3, int(0.08 * max(bx1 - bx0, by1 - by0)))

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    # (a) blob
    ax = axes[0]
    ax.imshow(comp["mask"][by0:by1, bx0:bx1], cmap="gray_r",
              extent=[bx0, bx1, by1, by0], interpolation="nearest")
    _setup_axis(ax, bbox, pad, "(a) blob\n(connected component)")

    # (b) skeleton
    ax = axes[1]
    ax.imshow(skel_local, cmap="gray_r",
              extent=[bx0, bx1, by1, by0], interpolation="nearest")
    _setup_axis(ax, bbox, pad, "(b) skeleton\n(1-px centerline)")

    # (c) walked polyline + direction arrows
    ax = axes[2]
    cmap = plt.get_cmap("turbo", max(len(walked), 1))
    for j, sp in enumerate(walked):
        col = cmap(j)
        ax.plot(sp[:, 0], sp[:, 1], "-", color=col, lw=1.6, alpha=0.9)
        ax.plot(sp[0, 0], sp[0, 1], "o", color=col, ms=5)  # start
        _draw_arrows(ax, sp, col)
    _setup_axis(ax, bbox, pad,
                f"(c) walked polyline\n({len(walked)} sub-path(s), arrows = order)")

    # (d) Gaussian-smoothed
    ax = axes[3]
    for j, sp in enumerate(smoothed):
        ax.plot(sp[:, 0], sp[:, 1], "-", color=cmap(j), lw=2.0)
    _setup_axis(ax, bbox, pad,
                f"(d) Gaussian-smoothed\n(post-walk, sigma = {walk_sigma})")

    # (e) arc-length resampled (per sub-path, no bridging across gaps)
    ax = axes[4]
    n_pts = 0
    for j, sp in enumerate(smoothed):
        ax.plot(sp[:, 0], sp[:, 1], "-", color="0.75", lw=1.2, zorder=1)
    for rp in resampled:
        ax.scatter(rp[:, 0], rp[:, 1], s=22, c="#c0392b",
                   zorder=3, edgecolors="white", linewidths=0.5)
        n_pts += len(rp)
    _setup_axis(ax, bbox, pad,
                f"(e) arc-length resampled\n({n_pts} pts, step={target_step}, +sigma={final_sigma})")

    fig.suptitle(
        f"{image_name}  -  trajectory-recovery sub-steps for one stroke "
        f"(area={comp['area']} px, extent={comp['extent']})",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, default=str(DEFAULT_IMAGE))
    p.add_argument("--out", type=str, default=str(THIS_DIR / "stroke_detail.png"))
    p.add_argument("--unet-ckpt", type=str, default=str(UNET_CKPT))
    p.add_argument("--stroke-rank", type=int, default=0,
                   help="0 = largest-area stroke, 1 = next largest, ...")
    p.add_argument("--thresh-interior", type=float, default=0.4)
    p.add_argument("--thresh-boundary", type=float, default=0.7)
    p.add_argument("--min-area", type=int, default=2)
    p.add_argument("--walk-sigma", type=float, default=1.2,
                   help="post-walk staircase-removal smoothing (thesis 3.4.5)")
    p.add_argument("--final-sigma", type=float, default=0.6,
                   help="light post-resampling smoothing (thesis 3.4.6)")
    p.add_argument("--target-step", type=float, default=0.117)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    unet = StrokeUNet(n_classes=3, base_ch=16).to(device)
    unet.load_state_dict(load_checkpoint(args.unet_ckpt, device)["state_dict"])
    unet.eval()

    import cv2
    raw = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(args.image)
    image, _ = preprocess_uhwr_image(raw)
    probs = predict_semantic(unet, image, device)
    inst = extract_instances(probs, args.thresh_interior, args.thresh_boundary, args.min_area)
    comps = build_components(inst, args.min_area)
    if not comps:
        raise RuntimeError("no stroke components found")

    # Walk + smooth EVERY component (needed for the whole-line z-score params).
    all_smoothed: List[np.ndarray] = []
    per_comp = []
    for c in comps:
        skel, walked, smoothed = walk_and_smooth(c, args.walk_sigma)
        per_comp.append((skel, walked, smoothed))
        if smoothed:
            all_smoothed.extend(smoothed)
    zparams = line_zscore_params(all_smoothed)

    # Choose target stroke: rank by area, skip degenerate ones.
    order = sorted(range(len(comps)), key=lambda i: -comps[i]["area"])
    viable = [i for i in order if per_comp[i][2]]  # has a smoothed walk
    if not viable:
        raise RuntimeError("no walkable stroke found")
    pick = viable[min(args.stroke_rank, len(viable) - 1)]
    comp = comps[pick]
    skel, walked, smoothed = per_comp[pick]

    resampled = arc_length_resample_pixels(smoothed, zparams, args.target_step,
                                           args.final_sigma)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_detail_figure(out_path, Path(args.image).name, comp, skel, walked,
                       smoothed, resampled, args.walk_sigma, args.final_sigma,
                       args.target_step)

    walk_pts = sum(len(s) for s in walked)
    resamp_pts = sum(len(r) for r in resampled)
    print(f"[detail] image: {args.image}")
    print(f"[detail] {len(comps)} components; picked rank {args.stroke_rank} "
          f"(area={comp['area']} px, extent={comp['extent']})")
    print(f"[detail] walk pts: {walk_pts} -> resampled: {resamp_pts} "
          f"across {len(resampled)} sub-path(s)")
    print(f"[detail] wrote: {out_path}")


if __name__ == "__main__":
    main()
