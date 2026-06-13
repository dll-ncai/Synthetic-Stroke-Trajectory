"""
Phase 10 - convert U-Net stroke masks into training-style patches.

The CNN classifier was trained on `render_clean_stroke(stroke_xy)` patches
(2-pixel polylines from trajectory points). At inference, the existing
`render_component_to_patch(component)` scales the actual ink mask, which
produces visually different filled-region patches.

This module bridges the gap:
    mask  ->  skeletonize  ->  order skeleton pixels into a polyline
          ->  render via render_clean_stroke  ->  training-style patch

Now the CNN sees the same style at inference as it did at training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARENT_DIR / "phase5b"))
sys.path.insert(0, str(PARENT_DIR / "phase7"))
sys.path.insert(0, str(PARENT_DIR / "phase9"))
sys.path.insert(0, str(THIS_DIR))

from clean_stroke_dataset import render_clean_stroke
from template_stitcher import extract_skeleton
from full_pipeline_unet_uhwr_skeleton import order_skeleton_points


def order_skeleton_multi_pass(skeleton_mask: np.ndarray):
    """Multi-pass version of order_skeleton_points.

    The single-pass greedy walk picks one path through a branched skeleton
    and abandons the rest. For complex Urdu letters like seen (س) with
    multiple humps connected to a baseline, this drops 30-50% of skeleton
    pixels.

    This function walks the skeleton multiple times:
      1. Do a greedy walk from the leftmost endpoint (or leftmost pixel
         if no endpoints).
      2. Find any unvisited skeleton pixels.
      3. From the leftmost unvisited pixel, do another greedy walk.
      4. Repeat until all skeleton pixels are covered.

    Returns:
        sub_paths: list of (T_i, 2) arrays of points. Each sub-path is a
                    continuous segment in pixel-coord local crop space.
                    Together they cover the full skeleton.
    """
    if skeleton_mask.sum() == 0:
        return []
    if skeleton_mask.sum() == 1:
        ys, xs = np.where(skeleton_mask > 0)
        return [np.array([[xs[0], ys[0]]], dtype=np.float32)]

    remaining = skeleton_mask.copy()
    sub_paths = []
    safety = 100   # max number of sub-paths (way more than ever needed)

    while remaining.sum() > 0 and safety > 0:
        safety -= 1
        # Walk the remaining skeleton
        path = order_skeleton_points(remaining)
        if path.shape[0] < 1:
            break
        sub_paths.append(path)
        # Mark visited pixels as no-longer-skeleton
        for x, y in path.astype(np.int32):
            if 0 <= y < remaining.shape[0] and 0 <= x < remaining.shape[1]:
                remaining[y, x] = 0
        # If we visited fewer than 2 pixels, we're stuck - bail
        if path.shape[0] < 2:
            break

    return sub_paths


def render_multi_polyline_to_patch(
    sub_paths,
    crop_size=(64, 256),
    pen_width: int = 2,
    padding_frac: float = 0.05,
) -> np.ndarray:
    """Render multiple sub-polylines onto a single training-style patch.

    All sub-paths share one global bbox so their relative positions are
    preserved; each sub-path is drawn as its own polyline so there are
    NO fake connecting lines between unrelated pieces.
    """
    H, W = crop_size
    canvas = np.zeros((H, W), dtype=np.float32)
    if not sub_paths:
        return canvas

    all_pts = np.vstack([sp for sp in sub_paths if sp.shape[0] > 0])
    if all_pts.shape[0] == 0:
        return canvas
    x_min, x_max = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
    y_min, y_max = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())
    src_w = max(x_max - x_min, 1.0)
    src_h = max(y_max - y_min, 1.0)
    avail_w = W * (1.0 - 2 * padding_frac)
    avail_h = H * (1.0 - 2 * padding_frac)
    scale = min(avail_w / src_w, avail_h / src_h)
    off_x = (W - src_w * scale) * 0.5
    off_y = (H - src_h * scale) * 0.5

    for sp in sub_paths:
        if sp.shape[0] == 0:
            continue
        pts = np.empty_like(sp, dtype=np.float32)
        pts[:, 0] = (sp[:, 0] - x_min) * scale + off_x
        pts[:, 1] = (sp[:, 1] - y_min) * scale + off_y
        if pts.shape[0] == 1:
            cv2.circle(canvas, tuple(pts[0].astype(int)),
                       max(1, pen_width // 2 + 1), color=1.0, thickness=-1)
        else:
            cv2.polylines(canvas, [pts.astype(np.int32).reshape(-1, 1, 2)],
                          isClosed=False, color=1.0,
                          thickness=max(1, pen_width), lineType=cv2.LINE_AA)
    return canvas


def _smooth_polyline(points: np.ndarray, sigma: float) -> np.ndarray:
    """Apply 1-D Gaussian smoothing to x and y independently.

    Skeleton-ordered points are integer pixel positions with sharp
    1-pixel zig-zags. A small sigma (~1.0-1.5) removes the staircase
    while preserving overall stroke geometry.
    """
    if sigma <= 0 or len(points) < 3:
        return points
    try:
        from scipy.ndimage import gaussian_filter1d
        smoothed = points.astype(np.float32).copy()
        smoothed[:, 0] = gaussian_filter1d(smoothed[:, 0], sigma=sigma, mode="nearest")
        smoothed[:, 1] = gaussian_filter1d(smoothed[:, 1], sigma=sigma, mode="nearest")
        return smoothed
    except ImportError:
        # Manual box-filter fallback if scipy missing
        k = max(1, int(round(sigma * 2)))
        n = len(points)
        out = points.astype(np.float32).copy()
        for i in range(n):
            lo, hi = max(0, i - k), min(n, i + k + 1)
            out[i] = points[lo:hi].mean(axis=0)
        return out


def mask_to_training_style_patch(
    component: dict,
    crop_size: tuple = (64, 256),
    pen_width: int = 2,
    diacritic_max_extent_px: float = 5.0,
    smooth_sigma: float = 1.2,
) -> np.ndarray:
    """Convert a U-Net stroke component into a CNN-training-style patch.

    Pipeline:
        1. If the component is small (diacritic), use render_clean_stroke's
           diacritic-dot fallback (matches training behaviour).
        2. Otherwise, extract the skeleton of the mask, order the skeleton
           pixels into a polyline, smooth the polyline with a small Gaussian
           filter (sigma=smooth_sigma) to remove pixel-step staircasing,
           then render via the same function used during classifier training.

    Args:
        smooth_sigma: sigma for Gaussian smoothing of the polyline before
            rasterization. ~1.0-1.5 cleans skeleton-step artifacts without
            distorting stroke shape. Set to 0.0 to disable smoothing.

    Returns:
        patch: (H, W) float32 in [0, 1] - same format as training data
    """
    H, W = crop_size
    bx0, by0, bx1, by1 = component["bbox"]
    src_w = bx1 - bx0
    src_h = by1 - by0
    max_extent = max(src_w, src_h)

    # Diacritic short-circuit (matches render_clean_stroke's behaviour)
    if max_extent < diacritic_max_extent_px:
        return render_clean_stroke(
            np.array([[component["x_center"], component["y_center"]]],
                      dtype=np.float32),
            crop_size=crop_size, pen_width=pen_width,
            diacritic_max_extent_px=diacritic_max_extent_px,
        )

    # Crop the mask region and skeletonize
    blob = component["mask"][by0:by1, bx0:bx1]
    if blob.sum() < 3:
        return np.zeros(crop_size, dtype=np.float32)
    skel = extract_skeleton(blob)
    if skel.sum() < 2:
        # Skeleton degenerate - fall back to centroid dot
        return render_clean_stroke(
            np.array([[component["x_center"], component["y_center"]]],
                      dtype=np.float32),
            crop_size=crop_size, pen_width=pen_width,
            diacritic_max_extent_px=diacritic_max_extent_px,
        )

    # Multi-pass walk: covers branched skeletons completely (each branch
    # becomes its own sub-path). Sub-paths are in LOCAL crop coords.
    sub_paths = order_skeleton_multi_pass(skel)
    if not sub_paths:
        return np.zeros(crop_size, dtype=np.float32)
    # Smooth each sub-path independently to remove staircase
    sub_paths = [_smooth_polyline(sp, sigma=smooth_sigma) for sp in sub_paths]
    # Render every sub-path on the same canvas with a shared global bbox;
    # each sub-path is its own cv2.polylines call, so no fake lines connect
    # disjoint pieces of the skeleton.
    return render_multi_polyline_to_patch(
        sub_paths,
        crop_size=crop_size,
        pen_width=pen_width,
    )


if __name__ == "__main__":
    # Quick smoke test
    print("[mask_to_training_patch] module loads OK")
