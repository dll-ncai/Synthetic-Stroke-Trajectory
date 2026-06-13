"""
Phase 9 - template stitching + skeleton refinement.

The classifier predicts ONE class per stroke. Each class maps to a vocab
template — 50 normalized [0,1]^2 points. To produce a recovered trajectory
on the line canvas, we need to:

    1. Stitch the template into the stroke's bbox (basic placement)
    2. Refine by snapping template points to the actual ink skeleton

Step 1 alone gives the "Path A / Path B baseline" reconstruction.
Step 2 adds the writer-specific geometry — predicted SHAPE category,
drawn along the writer's ACTUAL ink centerline.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def stitch_template_into_bbox(template: np.ndarray,
                              bbox: tuple) -> np.ndarray:
    """
    Place a [0,1]^2 template into a canvas-coordinate bbox.

    Args:
        template: (T, 2) array in [0, 1]^2 (vocab medoid)
        bbox: (x0, y0, x1, y1) in canvas pixels

    Returns:
        (T, 2) array of points in canvas pixel coords
    """
    x0, y0, x1, y1 = bbox
    w = max(float(x1 - x0), 1.0)
    h = max(float(y1 - y0), 1.0)
    out = np.empty_like(template, dtype=np.float32)
    out[:, 0] = x0 + template[:, 0] * w
    out[:, 1] = y0 + template[:, 1] * h
    return out


def extract_skeleton(mask: np.ndarray) -> np.ndarray:
    """
    Extract a one-pixel-thick centerline (skeleton) from a binary mask.

    Args:
        mask: (h, w) uint8 with 1=ink

    Returns:
        skeleton: (h, w) uint8 with 1=centerline pixel
    """
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    # Prefer cv2.ximgproc.thinning (fast); fall back to skimage if missing
    try:
        thinned = cv2.ximgproc.thinning(
            (mask * 255).astype(np.uint8),
            thinningType=cv2.ximgproc.THINNING_ZHANGSUEN,
        )
        return (thinned > 0).astype(np.uint8)
    except (AttributeError, cv2.error):
        try:
            from skimage.morphology import skeletonize
            return skeletonize(mask > 0).astype(np.uint8)
        except ImportError:
            # Last-resort fallback: morphological erosion (rough skeleton)
            sk = mask.copy().astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            for _ in range(5):
                eroded = cv2.erode(sk, kernel, iterations=1)
                if eroded.sum() == 0:
                    break
                sk = eroded
            return sk


def refine_with_skeleton(
    template_canvas: np.ndarray,
    component_mask: np.ndarray,
    bbox: tuple,
    snap_radius_px: float = 12.0,
) -> np.ndarray:
    """
    Snap each stitched template point to the nearest skeleton pixel.

    If a template point is more than snap_radius_px away from any
    skeleton pixel, it is left unchanged (i.e., we don't drag points
    across long empty regions). This preserves smooth interpolation
    where the template extends beyond the actual ink.

    Args:
        template_canvas: (T, 2) stitched template in canvas coords
        component_mask: (H_canvas, W_canvas) uint8 — full-canvas mask
                         of just this stroke's ink
        bbox: stroke's (x0, y0, x1, y1) for cropping the mask region
        snap_radius_px: max distance to snap; points further than this
                        keep their stitched position

    Returns:
        (T, 2) refined trajectory in canvas coords
    """
    bx0, by0, bx1, by1 = bbox
    blob = component_mask[by0:by1, bx0:bx1]
    if blob.sum() < 3:
        return template_canvas

    skeleton = extract_skeleton(blob)
    sk_y, sk_x = np.where(skeleton > 0)
    if len(sk_x) == 0:
        return template_canvas

    # Skeleton points in CANVAS coordinates (cv2 (x, y) order)
    skel_canvas = np.column_stack([
        sk_x.astype(np.float32) + bx0,
        sk_y.astype(np.float32) + by0,
    ])

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(skel_canvas)
        distances, indices = tree.query(template_canvas)
    except ImportError:
        # Brute force fallback
        distances = np.zeros(len(template_canvas))
        indices = np.zeros(len(template_canvas), dtype=np.int64)
        for i, pt in enumerate(template_canvas):
            d = np.linalg.norm(skel_canvas - pt, axis=1)
            j = int(np.argmin(d))
            distances[i] = float(d[j])
            indices[i] = j

    refined = template_canvas.copy()
    snap_mask = distances < snap_radius_px
    refined[snap_mask] = skel_canvas[indices[snap_mask]]
    return refined


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """Standard 2D segment-segment intersection test."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)
            and ccw(p1, p2, p3) != ccw(p1, p2, p4))


def count_self_intersections(traj: np.ndarray) -> int:
    """
    Count self-intersections in a polyline trajectory.

    Non-adjacent segments are tested for intersection. With 50-point
    trajectories this is ~1200 segment pairs — slow but fine for
    diagnostics on a small batch.
    """
    n = len(traj) - 1   # number of segments
    if n < 3:
        return 0
    count = 0
    for i in range(n - 2):
        for j in range(i + 2, n):
            if _segments_intersect(
                traj[i], traj[i + 1], traj[j], traj[j + 1],
            ):
                count += 1
    return count


def cumulative_turning_angle(traj: np.ndarray) -> float:
    """
    Sum of absolute turning angles along the trajectory (radians).

    Straight line = 0; spiral / curly stroke = large.
    Useful as a "curliness" feature distinct from arc length.
    """
    if len(traj) < 3:
        return 0.0
    vecs = np.diff(traj, axis=0)
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    turns = np.diff(angles)
    # wrap to [-pi, pi]
    turns = (turns + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(np.abs(turns)))


def count_direction_inflections(traj: np.ndarray,
                                 min_turn_threshold: float = 0.4) -> int:
    """
    Count "inflection points" — places where the trajectory direction
    reverses meaningfully. Wavy strokes have many; smooth curves have few.

    Implementation: count sign-changes of the signed turning angle that
    exceed a threshold. Matches what a human would call "a bump".
    """
    if len(traj) < 4:
        return 0
    vecs = np.diff(traj, axis=0)
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    turns = np.diff(angles)
    # wrap to [-pi, pi]
    turns = (turns + np.pi) % (2 * np.pi) - np.pi
    # Count sign changes in turning (only for sufficiently large turns)
    significant = np.abs(turns) > min_turn_threshold
    signs = np.sign(turns) * significant
    # Count adjacent non-zero pairs with opposite sign
    n = 0
    last = 0.0
    for s in signs:
        if s == 0:
            continue
        if last != 0 and s != last:
            n += 1
        last = s
    return int(n)


def shape_features(traj: np.ndarray) -> dict:
    """
    Topology + shape features that complement Chamfer distance.

    Returns dict with:
      start_end_dist:   distance between first and last point (closed loops → 0)
      n_self_intersect: how many times the stroke crosses itself
      turning:          cumulative |turn angle| along the path
      arc_length:       total path length
      n_inflections:    number of direction reversals ("bumps") in the stroke
    """
    if len(traj) < 2:
        return {"start_end_dist": 0.0, "n_self_intersect": 0,
                "turning": 0.0, "arc_length": 0.0, "n_inflections": 0}
    se = float(np.linalg.norm(traj[-1] - traj[0]))
    arc = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    return {
        "start_end_dist": se,
        "n_self_intersect": int(count_self_intersections(traj)),
        "turning": cumulative_turning_angle(traj),
        "arc_length": arc,
        "n_inflections": int(count_direction_inflections(traj)),
    }


def topology_aware_distance(query_traj: np.ndarray,
                             candidate_traj: np.ndarray,
                             w_chamfer: float = 1.0,
                             w_se: float = 0.3,
                             w_intersect: float = 0.2,
                             w_turning: float = 0.05,
                             w_inflections: float = 0.15,
                             w_arc_length: float = 0.10) -> float:
    """
    Topology-aware shape distance: Chamfer + per-feature differences.

    Penalises candidates that:
      - have a closed loop where the query is open (or vice versa)   → w_se
      - differ in self-intersection count                            → w_intersect
      - differ in cumulative turning angle (overall "curliness")     → w_turning
      - differ in number of inflection points / direction reversals  → w_inflections
      - differ significantly in arc length (compressed vs stretched) → w_arc_length

    Returns scalar (lower = more similar).
    """
    ch = chamfer_distance(query_traj, candidate_traj)
    qf = shape_features(query_traj)
    cf = shape_features(candidate_traj)
    se_diff = abs(qf["start_end_dist"] - cf["start_end_dist"])
    intersect_diff = abs(qf["n_self_intersect"] - cf["n_self_intersect"])
    turning_diff = abs(qf["turning"] - cf["turning"])
    inflect_diff = abs(qf["n_inflections"] - cf["n_inflections"])
    arc_diff = abs(qf["arc_length"] - cf["arc_length"])
    return (w_chamfer * ch
            + w_se * se_diff
            + w_intersect * intersect_diff
            + w_turning * turning_diff
            + w_inflections * inflect_diff
            + w_arc_length * arc_diff)


def chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """
    Symmetric Chamfer distance between two 2D point sets.

    For each point in A, find the nearest point in B (and vice-versa);
    return the average of those nearest-neighbour distances.

    This measures how well two shapes (as point clouds) overlap,
    independent of point ordering or sampling density.

    Args:
        pts_a: (M, 2)
        pts_b: (N, 2)
    Returns:
        float (lower = more similar shapes)
    """
    if pts_a.size == 0 or pts_b.size == 0:
        return float("inf")
    try:
        from scipy.spatial import cKDTree
        tree_b = cKDTree(pts_b)
        d_ab, _ = tree_b.query(pts_a)
        tree_a = cKDTree(pts_a)
        d_ba, _ = tree_a.query(pts_b)
        return float(d_ab.mean() + d_ba.mean())
    except ImportError:
        # Brute-force fallback
        d_ab = np.zeros(len(pts_a))
        for i, p in enumerate(pts_a):
            d_ab[i] = float(np.linalg.norm(pts_b - p, axis=1).min())
        d_ba = np.zeros(len(pts_b))
        for i, p in enumerate(pts_b):
            d_ba[i] = float(np.linalg.norm(pts_a - p, axis=1).min())
        return float(d_ab.mean() + d_ba.mean())


def diacritic_template_at_center(component_mask: np.ndarray,
                                  bbox: tuple,
                                  template_len: int = 50) -> np.ndarray:
    """
    For diacritic components (DIAC class), return a degenerate trajectory
    at the bbox's centroid — `template_len` points all at the same
    canvas location. Renders as a single dot at that position.
    """
    bx0, by0, bx1, by1 = bbox
    cx = 0.5 * (bx0 + bx1)
    cy = 0.5 * (by0 + by1)
    out = np.empty((template_len, 2), dtype=np.float32)
    out[:, 0] = cx
    out[:, 1] = cy
    return out
