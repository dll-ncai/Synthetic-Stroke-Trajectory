"""
Phase 9 - hybrid offline-to-online trajectory recovery.

End-to-end pipeline:

    raw line image  (no GT needed at inference)
            ↓
    [stroke_detector] OpenCV connected components → ordered Components
            ↓
    [stroke_renderer]  render each component to a clean 64×256 patch
            ↓
    [CNN classifier]   per-patch class id (uses Phase 7 trained model)
            ↓
    [template_stitcher] place template into bbox  +  skeleton-refine
            ↓
    ordered list of (class_id, bbox, canvas-coord trajectory)

This module exposes one entry point `hybrid_recover_from_image` that does
the whole pipeline and returns a dict of intermediate results, suitable
for visualization and metric computation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARENT_DIR / "phase5b"))
sys.path.insert(0, str(THIS_DIR))

from cnn_classifier import StrokeCNNClassifier
from stroke_detector import detect_strokes
from template_stitcher import (
    stitch_template_into_bbox,
    refine_with_skeleton,
    diacritic_template_at_center,
)


def render_component_to_patch(
    component: Dict,
    crop_size=(64, 256),
    pen_width: int = 2,
    padding_frac: float = 0.05,
    diacritic_max_extent_px: float = 5.0,
) -> np.ndarray:
    """
    Render a component's mask onto a clean (H, W) patch, aspect-preserving.

    Matches the training-time render_clean_stroke distribution for the
    Phase 7 classifier. DIAC-sized components are rendered as a small
    centered dot (NOT scaled up).

    Returns:
        (H, W) float32 in [0, 1]
    """
    H, W = crop_size
    canvas = np.zeros((H, W), dtype=np.float32)
    bx0, by0, bx1, by1 = component["bbox"]
    src_w = bx1 - bx0
    src_h = by1 - by0
    max_extent = max(src_w, src_h)

    if max_extent < diacritic_max_extent_px:
        cv2.circle(canvas, (W // 2, H // 2),
                   max(2, pen_width // 2 + 1), color=1.0, thickness=-1)
        return canvas

    blob = component["mask"][by0:by1, bx0:bx1].astype(np.float32)
    if blob.size == 0 or blob.sum() == 0:
        return canvas

    avail_w = W * (1.0 - 2 * padding_frac)
    avail_h = H * (1.0 - 2 * padding_frac)
    scale = min(avail_w / max(src_w, 1), avail_h / max(src_h, 1))
    new_w = max(int(round(src_w * scale)), 1)
    new_h = max(int(round(src_h * scale)), 1)

    blob_resized = cv2.resize(blob, (new_w, new_h),
                              interpolation=cv2.INTER_AREA)
    off_x = (W - new_w) // 2
    off_y = (H - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = np.maximum(
        canvas[off_y:off_y + new_h, off_x:off_x + new_w], blob_resized,
    )
    return canvas


def build_full_canvas_mask(components: List[Dict], canvas_shape) -> np.ndarray:
    """Stack all component masks into a single full-canvas mask (1=any ink)."""
    H, W = canvas_shape
    out = np.zeros((H, W), dtype=np.uint8)
    for c in components:
        out |= c["mask"]
    return out


def hybrid_recover_from_image(
    image: np.ndarray,
    classifier: torch.nn.Module,
    vocab_centers: np.ndarray,
    diacritic_class_id: int,
    crop_size=(64, 256),
    pen_width: int = 2,
    padding_frac: float = 0.05,
    snap_radius_px: float = 12.0,
    min_area: int = 3,
    min_extent: int = 1,
    device: str = "cpu",
) -> Dict:
    """
    Run the hybrid pipeline end-to-end on one line image.

    Args:
        image: (H, W) grayscale float [0, 1] or uint8 [0, 255]
        classifier: trained StrokeCNNClassifier
        vocab_centers: (K_total, T, 2) — vocab templates in [0,1]^2
        diacritic_class_id: which class id is DIAC
        crop_size: classifier input size
        snap_radius_px: max pixels to snap during skeleton refinement

    Returns:
        dict with:
            components:       ordered list from detector
            class_ids:        per-component predicted class id (np.ndarray)
            top_probs:        per-component top-1 probability
            top5_ids:         (N, 5) top-5 class ids
            stitched_strokes: list of (T, 2) canvas-coord trajectories (no refine)
            refined_strokes:  list of (T, 2) canvas-coord trajectories (with refine)
    """
    components = detect_strokes(
        image, min_area=min_area, min_extent=min_extent,
        diacritic_max_extent_px=5.0,
    )
    if len(components) == 0:
        return {
            "components": [],
            "class_ids": np.zeros(0, dtype=np.int64),
            "top_probs": np.zeros(0, dtype=np.float32),
            "top5_ids": np.zeros((0, 5), dtype=np.int64),
            "stitched_strokes": [],
            "refined_strokes": [],
        }

    # Render each component to a clean patch
    patches = np.stack([
        render_component_to_patch(c, crop_size=crop_size,
                                   pen_width=pen_width,
                                   padding_frac=padding_frac)
        for c in components
    ])
    batch = torch.from_numpy(patches).unsqueeze(1).float().to(device)

    classifier.eval()
    with torch.no_grad():
        logits = classifier(batch)
        probs = torch.softmax(logits, dim=-1)
        class_ids = logits.argmax(dim=-1).cpu().numpy()
        top_probs = probs.gather(
            1, torch.from_numpy(class_ids).long().unsqueeze(-1)
        ).squeeze(-1).cpu().numpy()
        top5 = probs.topk(min(5, probs.size(-1)), dim=-1).indices.cpu().numpy()

    # Full-canvas mask (every stroke's ink) for the refinement step
    canvas_shape = image.shape
    full_mask = build_full_canvas_mask(components, canvas_shape)

    stitched_strokes: List[np.ndarray] = []
    refined_strokes: List[np.ndarray] = []
    for c, k in zip(components, class_ids):
        template = vocab_centers[int(k)]
        if int(k) == diacritic_class_id:
            stitched = diacritic_template_at_center(
                c["mask"], c["bbox"], template_len=template.shape[0],
            )
            refined = stitched
        else:
            stitched = stitch_template_into_bbox(template, c["bbox"])
            # Only consider THIS component's mask when refining (don't
            # snap onto a neighbouring stroke's ink)
            refined = refine_with_skeleton(
                stitched, c["mask"], c["bbox"],
                snap_radius_px=snap_radius_px,
            )
        stitched_strokes.append(stitched)
        refined_strokes.append(refined)

    return {
        "components": components,
        "class_ids": class_ids,
        "top_probs": top_probs,
        "top5_ids": top5,
        "stitched_strokes": stitched_strokes,
        "refined_strokes": refined_strokes,
    }


def load_classifier(ckpt_path: Path,
                    n_classes: int,
                    hidden: int = 256,
                    device: str = "cpu") -> StrokeCNNClassifier:
    """Convenience: load Phase 7 scratch CNN classifier from checkpoint."""
    model = StrokeCNNClassifier(n_classes=n_classes, hidden=hidden, dropout=0.0)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model
