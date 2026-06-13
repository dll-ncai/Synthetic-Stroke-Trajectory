"""
Phase 9 - classical stroke detector (no learning).

Pipeline:
    raw line image (grayscale)
            ↓
    Otsu binarization → binary ink/no-ink
            ↓
    cv2.connectedComponentsWithStats → list of ink blobs
            ↓
    filter (min area, min extent) → drop noise
            ↓
    order for Urdu reading direction → right-to-left primary,
                                       top-to-bottom secondary
            ↓
    list of Component dicts ready for classifier

Each Component dict has:
    mask:     (H, W) uint8       — binary mask of this blob in CANVAS coords
    bbox:     (x0, y0, x1, y1)   — inclusive bbox in canvas pixels
    area:     int                — number of ink pixels
    x_center: float              — centroid x in canvas
    y_center: float              — centroid y in canvas
    extent:   (w, h)             — bbox width and height

This module deliberately has zero deep-learning dependencies — just
NumPy + OpenCV. Pure image processing.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


def binarize_image(img: np.ndarray,
                    morphology: str = "none",
                    ksize: int = 3) -> np.ndarray:
    """
    Otsu binarization with optional morphological pre-processing.

    Args:
        img: float [0, 1] or uint8 [0, 255]
        morphology: one of "none", "dilate", "erode", "opening", "closing"
            "opening" = erode then dilate (breaks touching strokes while
            preserving thickness — empirically best for Urdu Nastaleeq CC)
        ksize: morphology kernel size (3x3 default)

    Returns:
        binary: (H, W) uint8 with 1 = ink, 0 = background.
    """
    if img.dtype != np.uint8:
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img_u8 = img
    _, binary = cv2.threshold(
        img_u8, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    binary = (binary > 0).astype(np.uint8)
    if morphology != "none":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        if morphology == "dilate":
            binary = cv2.dilate(binary, kernel, iterations=1)
        elif morphology == "erode":
            binary = cv2.erode(binary, kernel, iterations=1)
        elif morphology == "opening":
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif morphology == "closing":
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        else:
            raise ValueError(f"Unknown morphology mode: {morphology}")
    return binary


def find_connected_components(
    binary: np.ndarray,
    min_area: int = 3,
    min_extent: int = 1,
) -> List[Dict]:
    """
    Run 8-connectivity connected components on a binary image.

    Args:
        binary: (H, W) uint8 with 1 = ink
        min_area: drop components with fewer than this many ink pixels
        min_extent: drop components whose bbox is shorter than this in
                    BOTH dimensions

    Returns:
        list of Component dicts (see module docstring)
    """
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )
    components: List[Dict] = []
    H, W = binary.shape
    for i in range(1, n_labels):  # 0 is the background label
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if max(w, h) < min_extent:
            continue
        mask = (labels == i).astype(np.uint8)
        components.append({
            "mask": mask,
            "bbox": (x, y, x + w, y + h),
            "area": area,
            "x_center": float(centroids[i, 0]),
            "y_center": float(centroids[i, 1]),
            "extent": (w, h),
        })
    return components


def order_components_urdu(
    components: List[Dict],
    canvas_size: Tuple[int, int],
    diacritic_max_extent_px: float = 5.0,
) -> List[Dict]:
    """
    Order components for Urdu Nastaleeq reading direction.

    Heuristic:
      1. Split into "main" (extent >= diacritic_max_extent_px) and
         "diacritic" (extent < diacritic_max_extent_px) components.
      2. Sort main components by x_center DESCENDING (right-to-left).
      3. For each main component, find diacritics whose x_center falls
         inside (or near) its bbox and attach them AFTER the main
         component (matches typical writer behaviour: letter base then
         add the dots).
      4. Diacritics that don't anchor to any main component (rare) are
         appended at the end ordered right-to-left.

    Args:
        components: from find_connected_components
        canvas_size: (H, W) for any future use; not currently consumed
        diacritic_max_extent_px: tiny blobs are treated as dots

    Returns:
        ordered list of components
    """
    main_list = [c for c in components
                 if max(c["extent"]) >= diacritic_max_extent_px]
    diac_list = [c for c in components
                 if max(c["extent"]) < diacritic_max_extent_px]

    # Sort main components right-to-left
    main_list.sort(key=lambda c: -c["x_center"])

    # Assign each diacritic to a main component by x-overlap
    assigned: Dict[int, List[Dict]] = {}
    leftover: List[Dict] = []
    for d in diac_list:
        best = -1
        best_dx = float("inf")
        for i, m in enumerate(main_list):
            mx0, my0, mx1, my1 = m["bbox"]
            # Allow a small horizontal margin around the main bbox
            margin = max(2, int(0.1 * (mx1 - mx0)))
            if (mx0 - margin) <= d["x_center"] <= (mx1 + margin):
                # Prefer the closest main center
                dx = abs(d["x_center"] - m["x_center"])
                if dx < best_dx:
                    best_dx = dx
                    best = i
        if best >= 0:
            assigned.setdefault(best, []).append(d)
        else:
            leftover.append(d)

    ordered: List[Dict] = []
    for i, m in enumerate(main_list):
        ordered.append(m)
        # Diacritics attached to this main: top-to-bottom (above before below)
        for d in sorted(assigned.get(i, []), key=lambda c: c["y_center"]):
            ordered.append(d)
    # Leftover diacritics (no main parent): append by x descending
    leftover.sort(key=lambda c: -c["x_center"])
    ordered.extend(leftover)
    return ordered


def detect_strokes(
    image: np.ndarray,
    min_area: int = 3,
    min_extent: int = 1,
    diacritic_max_extent_px: float = 5.0,
    morphology: str = "opening",
) -> List[Dict]:
    """
    Convenience: image → ordered list of components.

    Args:
        image: (H, W) grayscale, float [0,1] or uint8 [0,255]
        morphology: pre-CC morphological preprocessing (default "opening" — see
            phase9/cc_morphology_test.py for the empirical comparison that
            chose this default for Urdu Nastaleeq lines)

    Returns:
        ordered list of Component dicts
    """
    binary = binarize_image(image, morphology=morphology)
    components = find_connected_components(
        binary, min_area=min_area, min_extent=min_extent,
    )
    return order_components_urdu(
        components, image.shape,
        diacritic_max_extent_px=diacritic_max_extent_px,
    )
