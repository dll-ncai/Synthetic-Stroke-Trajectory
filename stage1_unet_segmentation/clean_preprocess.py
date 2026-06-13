"""
Phase 10 - cleaner UHWR preprocessing.

Adds an explicit binarization + small-component removal step BEFORE
feeding to the U-Net. This eliminates background noise (e.g., scan dashes,
paper texture, faint dust) that would otherwise get picked up as
"ink instances" by the U-Net.

Pipeline:
    raw UHWR (grayscale)
        -> normalize + polarity detect + resize + letterbox (existing)
        -> Otsu threshold -> binary
        -> remove connected components below min_noise_area pixels
        -> binary image ready for U-Net

This matches the training distribution more closely - the U-Net was
trained on binary-ish renderings (ink=1, bg=0).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from test_uhwr import preprocess_uhwr_image


def skeletonize_binary(binary: np.ndarray) -> np.ndarray:
    """Skeletonize a (H, W) binary image (0/1) to a 1-px-thick centerline.

    Tries cv2.ximgproc.thinning first (Zhang-Suen), falls back to skimage,
    then to morphological erosion if nothing else is available.
    """
    if binary.dtype != np.uint8:
        b = (binary > 0).astype(np.uint8)
    else:
        b = (binary > 0).astype(np.uint8)
    if b.sum() == 0:
        return b.astype(np.float32)
    try:
        thinned = cv2.ximgproc.thinning(
            (b * 255).astype(np.uint8),
            thinningType=cv2.ximgproc.THINNING_ZHANGSUEN,
        )
        return (thinned > 0).astype(np.float32)
    except (AttributeError, cv2.error):
        try:
            from skimage.morphology import skeletonize
            return skeletonize(b > 0).astype(np.float32)
        except ImportError:
            sk = b.copy()
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            for _ in range(5):
                eroded = cv2.erode(sk, kernel, iterations=1)
                if eroded.sum() == 0:
                    break
                sk = eroded
            return sk.astype(np.float32)


def binarize_and_clean(image: np.ndarray,
                       min_noise_area: int = 8,
                       diacritic_keep_distance: float = 15.0,
                       use_otsu: bool = True,
                       fixed_thresh: float = 0.3) -> np.ndarray:
    """Binarize a preprocessed float image with smart noise removal.

    Distinguishes noise from real diacritics by SPATIAL CONTEXT:
        - Small component (area < min_noise_area) that is FAR from any
          larger component  -> scan noise, DROP.
        - Small component that is CLOSE to a larger component (within
          diacritic_keep_distance px)  -> likely diacritic dot/mark, KEEP.
        - Components with area >= min_noise_area  -> always KEEP.

    Args:
        image: (H, W) float32 in [0, 1] with ink approximately = 1
        min_noise_area: components below this size are subject to the
            spatial-context check. Above this size, always kept.
        diacritic_keep_distance: a small component is kept if its bbox is
            within this many pixels of any larger component's bbox. Real
            diacritics sit close to their parent letter; isolated noise
            pixels are typically much further away.
        use_otsu: Otsu vs fixed threshold for binarization
        fixed_thresh: threshold if not Otsu

    Returns:
        clean: (H, W) float32 in {0.0, 1.0}
    """
    if image.dtype != np.uint8:
        u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        u8 = image.copy()

    if use_otsu:
        _, binary = cv2.threshold(u8, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = (u8 > int(fixed_thresh * 255)).astype(np.uint8) * 255
    binary = (binary > 0).astype(np.uint8)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )
    if n_labels <= 1:
        return binary.astype(np.float32)

    # Separate components into "large" (always kept) and "small" (need check)
    areas = stats[1:, cv2.CC_STAT_AREA]
    is_small = areas < min_noise_area
    large_idx = np.where(~is_small)[0] + 1   # +1 because we skipped bg (0)
    small_idx = np.where(is_small)[0] + 1

    keep_set = set(int(i) for i in large_idx)

    # For each small component, check distance to nearest large component
    # using bbox-edge distance (faster than per-pixel distance, and
    # accurate enough for "is this near a letter?" check).
    if len(small_idx) > 0 and len(large_idx) > 0:
        # Build bbox arrays for large components
        large_boxes = stats[large_idx, :4]   # (N_large, 4): [x, y, w, h]
        large_x0 = large_boxes[:, 0]
        large_y0 = large_boxes[:, 1]
        large_x1 = large_x0 + large_boxes[:, 2]
        large_y1 = large_y0 + large_boxes[:, 3]

        for si in small_idx:
            sx0 = stats[si, cv2.CC_STAT_LEFT]
            sy0 = stats[si, cv2.CC_STAT_TOP]
            sx1 = sx0 + stats[si, cv2.CC_STAT_WIDTH]
            sy1 = sy0 + stats[si, cv2.CC_STAT_HEIGHT]
            # Per-large bbox-to-bbox gap distance
            dx = np.maximum(0, np.maximum(large_x0 - sx1, sx0 - large_x1))
            dy = np.maximum(0, np.maximum(large_y0 - sy1, sy0 - large_y1))
            min_gap = np.min(np.sqrt(dx * dx + dy * dy))
            if min_gap <= diacritic_keep_distance:
                keep_set.add(int(si))

    clean = np.zeros_like(binary)
    for i in keep_set:
        clean[labels == i] = 1
    return clean.astype(np.float32)


def preprocess_uhwr_image_clean(raw: np.ndarray,
                                 min_noise_area: int = 8,
                                 diacritic_keep_distance: float = 15.0,
                                 compute_skeleton: bool = True):
    """Full preprocessing: existing resize/polarity logic + binarize+clean
    + optional skeletonization.

    Returns dict with keys:
        clean_binary: (H, W) float32 in {0,1} — the binarized image (default
                       U-Net input)
        preprocessed_float: (H, W) float32 in [0,1] — pre-binarize version
        skeleton: (H, W) float32 in {0,1} — 1-px-thick centerline of the
                  clean_binary (only if compute_skeleton=True)
        meta: dict from preprocess_uhwr_image
    """
    pre, meta = preprocess_uhwr_image(raw)
    clean = binarize_and_clean(
        pre,
        min_noise_area=min_noise_area,
        diacritic_keep_distance=diacritic_keep_distance,
    )
    skel = skeletonize_binary(clean) if compute_skeleton else None
    return {
        "clean_binary": clean,
        "preprocessed_float": pre,
        "skeleton": skel,
        "meta": meta,
    }


if __name__ == "__main__":
    print("[clean_preprocess] module loads OK")
