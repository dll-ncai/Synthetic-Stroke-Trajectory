"""
Phase 7 - clean per-stroke dataset.

Replaces the contaminated bbox-crop pipeline with cleanly-rendered
per-stroke images:

    GT stroke trajectory (canvas px)
        ↓
    rasterize polyline on blank canvas at native aspect ratio
        ↓
    place centered in fixed (crop_h, crop_w) patch with zero padding
        ↓
    CNN input

This guarantees the model sees ONLY the target stroke's ink — no overlap
from neighbouring strokes. Aspect ratio is preserved; the stroke isn't
stretched.

Also exposes a LINE-LEVEL train/val split so a writer's strokes don't
appear in both sets (the existing stroke-level split has 100% leakage).
"""

from __future__ import annotations
import config

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARENT_DIR / "phase5b"))

from trajectory_dataset import TrajectoryDataset
from stroke_sequence_utils import extract_pen_down_strokes


def render_clean_stroke(
    stroke_xy: np.ndarray,
    crop_size: Tuple[int, int] = (64, 256),
    pen_width: int = 2,
    padding_frac: float = 0.05,
    diacritic_max_extent_px: float = 5.0,
    diacritic_dot_radius_px: int = 3,
) -> np.ndarray:
    """
    Rasterize a single stroke trajectory onto a clean (H, W) canvas.

    For NORMAL strokes (canvas extent >= diacritic_max_extent_px):
        aspect-preserving scale to fit (1 - 2*padding_frac) of the crop.

    For DIACRITIC strokes (canvas extent < diacritic_max_extent_px):
        render as a small centered dot — do NOT scale up. Scaling a 2-pixel
        diacritic to fill a 64x256 patch turns it into a fake "line" and
        breaks the diacritic-class assumption.

    Args:
        stroke_xy: (T, 2) array of (x, y) in canvas pixel coordinates
        crop_size: (H, W) output patch size
        pen_width: brush stroke width in pixels (for normal strokes)
        padding_frac: fraction of each axis reserved as margin for normal strokes
        diacritic_max_extent_px: strokes smaller than this in raw canvas are dots
        diacritic_dot_radius_px: radius of the rendered diacritic dot in pixels

    Returns:
        patch: (H, W) float32 in [0, 1] with 0=white, 1=black-ink
    """
    H, W = crop_size
    canvas = np.zeros((H, W), dtype=np.float32)

    if stroke_xy.shape[0] == 0:
        return canvas

    x_min, x_max = float(stroke_xy[:, 0].min()), float(stroke_xy[:, 0].max())
    y_min, y_max = float(stroke_xy[:, 1].min()), float(stroke_xy[:, 1].max())
    src_w = x_max - x_min
    src_h = y_max - y_min
    max_extent = max(src_w, src_h)

    # ---- Diacritic case: render as a small dot at the center, no scaling ----
    if max_extent < diacritic_max_extent_px:
        cx, cy = W // 2, H // 2
        cv2.circle(canvas, (cx, cy), diacritic_dot_radius_px,
                   color=1.0, thickness=-1)
        return canvas

    # ---- Normal case: aspect-preserving fit ----
    src_w = max(src_w, 1.0)
    src_h = max(src_h, 1.0)
    avail_w = W * (1.0 - 2 * padding_frac)
    avail_h = H * (1.0 - 2 * padding_frac)
    scale = min(avail_w / src_w, avail_h / src_h)
    scaled_w = src_w * scale
    scaled_h = src_h * scale
    off_x = (W - scaled_w) * 0.5
    off_y = (H - scaled_h) * 0.5

    pts = np.empty_like(stroke_xy, dtype=np.float32)
    pts[:, 0] = (stroke_xy[:, 0] - x_min) * scale + off_x
    pts[:, 1] = (stroke_xy[:, 1] - y_min) * scale + off_y

    pts_int = pts.astype(np.int32).reshape(-1, 1, 2)
    if pts.shape[0] == 1:
        cv2.circle(canvas, (int(pts_int[0, 0, 0]), int(pts_int[0, 0, 1])),
                   max(1, pen_width // 2 + 1), color=1.0, thickness=-1)
    else:
        cv2.polylines(canvas, [pts_int], isClosed=False, color=1.0,
                      thickness=max(1, pen_width), lineType=cv2.LINE_AA)
    return canvas


class CleanStrokeDataset(Dataset):
    """
    Per-stroke dataset that yields CLEAN rendered patches (one stroke only,
    no neighbouring contamination) + class labels from the Phase 5b vocab.

    Yields: (patch, label, meta)
        patch: torch.float32 (1, H, W)
        label: torch.long scalar in [0, K_total - 1]
        meta:  dict with line_idx, stroke_idx, raw_lengths, gt_local_traj,
               gt_local_len, is_diacritic
    """

    def __init__(
        self,
        meta_csv_path: str,
        vocab_dir: Path,
        crop_size: Tuple[int, int] = (64, 256),
        pen_width: int = 2,
        padding_frac: float = 0.05,
        min_stroke_points: int = 1,
        max_stroke_points: int = 200,
        base_path: str = config.DATA_ROOT_STR,
    ):
        self.crop_size = crop_size
        self.pen_width = pen_width
        self.padding_frac = padding_frac
        self.min_stroke_points = min_stroke_points
        self.max_stroke_points = max_stroke_points

        self.base = TrajectoryDataset(
            meta_csv_path=meta_csv_path,
            normalize_coords=True,
            normalization_mode="canvas",
            base_path=base_path,
        )

        labels_npz = vocab_dir / "label_assignments.npz"
        if not labels_npz.exists():
            raise FileNotFoundError(f"label_assignments.npz not found at {labels_npz}")
        data = np.load(labels_npz)
        line_idx_arr = data["line_idx"]
        stroke_idx_arr = data["stroke_idx"]
        labels_arr = data["labels"]
        is_diac_arr = (
            data["is_diacritic"].astype(np.uint8)
            if "is_diacritic" in data.files
            else np.zeros(len(labels_arr), dtype=np.uint8)
        )
        self.diacritic_class_id = (
            int(data["diacritic_class_id"])
            if "diacritic_class_id" in data.files else -1
        )

        self.lookup: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for li, si, lab, di in zip(line_idx_arr, stroke_idx_arr,
                                    labels_arr, is_diac_arr):
            self.lookup[(int(li), int(si))] = (int(lab), int(di))

        # Build (line_idx, stroke_idx) index over the dataset
        self.index: List[Tuple[int, int]] = []
        n_dropped = 0
        for line_idx in range(len(self.base)):
            try:
                _, traj_tensor, _, _ = self.base[line_idx]
            except Exception:
                continue
            strokes = extract_pen_down_strokes(traj_tensor.numpy())
            for s_idx, stroke in enumerate(strokes):
                if stroke.shape[0] < self.min_stroke_points:
                    continue
                if (line_idx, s_idx) not in self.lookup:
                    n_dropped += 1
                    continue
                self.index.append((line_idx, s_idx))

        print(
            f"[CleanStrokeDataset] kept={len(self.index)}  "
            f"dropped_unlabeled={n_dropped}  "
            f"diacritic_class_id={self.diacritic_class_id}"
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        line_idx, stroke_idx = self.index[i]
        _, traj_tensor, _, _ = self.base[line_idx]
        strokes = extract_pen_down_strokes(traj_tensor.numpy())
        stroke = strokes[stroke_idx]
        stroke_xy = stroke[:, :2]
        if stroke_xy.shape[0] > self.max_stroke_points:
            stroke_xy = stroke_xy[: self.max_stroke_points]

        # Render clean patch
        patch = render_clean_stroke(
            stroke_xy,
            crop_size=self.crop_size,
            pen_width=self.pen_width,
            padding_frac=self.padding_frac,
        )
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()  # (1, H, W)

        # Same local-coord trajectory the visualizer expects, but computed
        # using the stroke's own bbox (no canvas-relative crop needed since
        # the patch is rendered cleanly).
        x_min, x_max = float(stroke_xy[:, 0].min()), float(stroke_xy[:, 0].max())
        y_min, y_max = float(stroke_xy[:, 1].min()), float(stroke_xy[:, 1].max())
        w_src = max(x_max - x_min, 1.0)
        h_src = max(y_max - y_min, 1.0)
        local = np.empty_like(stroke_xy, dtype=np.float32)
        local[:, 0] = (stroke_xy[:, 0] - x_min) / w_src
        local[:, 1] = (stroke_xy[:, 1] - y_min) / h_src

        label, is_diac = self.lookup[(line_idx, stroke_idx)]
        meta = {
            "line_idx": line_idx,
            "stroke_idx": stroke_idx,
            "label": label,
            "is_diacritic": bool(is_diac),
            "gt_local_traj": local,
            "gt_local_len": int(local.shape[0]),
            "stroke_len_raw": int(stroke.shape[0]),
            "crop_size": self.crop_size,
        }
        return patch_tensor, torch.tensor(label, dtype=torch.long), meta


def clean_collate(batch):
    patches, labels, metas = zip(*batch)
    patches = torch.stack(list(patches), dim=0)
    labels = torch.stack(list(labels), dim=0)
    return patches, labels, list(metas)


def line_level_split(dataset: CleanStrokeDataset, val_frac: float = 0.1,
                     seed: int = 0):
    """
    Split sample indices by LINE so no line appears in both train and val.

    Returns (train_idx, val_idx) as lists of dataset indices.

    Important: this differs from the existing stroke-level split (which has
    100% line leakage) used in train_local_shape.py and train_classifier.py.
    """
    line_to_samples: Dict[int, List[int]] = {}
    for sample_i, (line_idx, _) in enumerate(dataset.index):
        line_to_samples.setdefault(line_idx, []).append(sample_i)

    unique_lines = sorted(line_to_samples.keys())
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique_lines))
    n_val_lines = int(len(unique_lines) * val_frac)
    val_lines = {unique_lines[i] for i in perm[:n_val_lines]}
    train_lines = {unique_lines[i] for i in perm[n_val_lines:]}

    train_idx, val_idx = [], []
    for line, samples in line_to_samples.items():
        if line in val_lines:
            val_idx.extend(samples)
        else:
            train_idx.extend(samples)
    print(f"[line_level_split] lines: train={len(train_lines)}  val={len(val_lines)}  "
          f"samples: train={len(train_idx)}  val={len(val_idx)}")
    return train_idx, val_idx
