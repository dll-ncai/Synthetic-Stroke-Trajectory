"""
Phase 10 - stroke instance segmentation dataset.

Pure offline-to-online learning: input is an image, output is per-stroke
pixel labels. We render the image ourselves from the labelled trajectory
so labels are pixel-perfect. Augmentation simulates scanned-paper conditions
to narrow the gap to UHWR.

Output per sample:
    image          (1, H, W) float32 in [0, 1]
    semantic       (H, W) int64 — 0=background, 1=stroke-interior, 2=stroke-boundary
    instance_mask  (H, W) int32 — 0=background, 1..N=stroke index (in pen order)
    n_strokes      int
    meta           dict
"""

from __future__ import annotations
import config

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

from trajectory_dataset import TrajectoryDataset  # noqa: E402
from stroke_sequence_utils import extract_pen_down_strokes  # noqa: E402


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def _rasterize_strokes(
    strokes: List[np.ndarray],
    canvas_h: int,
    canvas_w: int,
    thickness: int,
) -> np.ndarray:
    """Rasterize ordered pen-down strokes into an integer instance mask.

    Pixel labels are 1-indexed so 0 stays as "background". Later strokes
    overwrite earlier ones at overlap pixels (painter's algorithm),
    matching how pen ink physically covers previous ink.
    """
    inst = np.zeros((canvas_h, canvas_w), dtype=np.int32)
    for k, stroke in enumerate(strokes, start=1):
        pts = stroke[:, :2].astype(np.int32)
        if len(pts) == 0:
            continue
        if len(pts) == 1:
            cv2.circle(inst, tuple(pts[0]), max(1, thickness // 2), color=k, thickness=-1)
            continue
        for i in range(len(pts) - 1):
            cv2.line(inst, tuple(pts[i]), tuple(pts[i + 1]),
                     color=int(k), thickness=thickness, lineType=cv2.LINE_8)
    return inst


def _build_semantic_from_instance(instance_mask: np.ndarray) -> np.ndarray:
    """Derive 3-class semantic mask: 0=bg, 1=interior, 2=boundary.

    A boundary pixel is an ink pixel whose 3x3 neighbourhood contains
    a *different* positive instance label — i.e., the place where two
    strokes touch. This is what the U-Net learns to mark so connected
    components can split touching strokes at inference.
    """
    inst = instance_mask
    # Dilate the instance label - a pixel sees neighbours via max in 3x3
    # We want: pixel is boundary iff it is ink AND there exists another
    # instance label != current within 3x3.
    h, w = inst.shape
    # Pad with zeros for safe neighbourhood lookup
    padded = np.pad(inst, 1, mode="constant", constant_values=0)
    # Stack 8 neighbours
    neigh = np.stack([
        padded[0:h, 0:w], padded[0:h, 1:w + 1], padded[0:h, 2:w + 2],
        padded[1:h + 1, 0:w],                    padded[1:h + 1, 2:w + 2],
        padded[2:h + 2, 0:w], padded[2:h + 2, 1:w + 1], padded[2:h + 2, 2:w + 2],
    ], axis=0)  # (8, H, W)
    # boundary mask: any neighbour has a non-zero label different from this pixel's label
    is_ink = inst > 0
    diff_neigh = (neigh > 0) & (neigh != inst[None])
    has_diff = np.any(diff_neigh, axis=0)
    boundary = is_ink & has_diff

    semantic = np.zeros_like(inst, dtype=np.int64)
    semantic[is_ink] = 1
    semantic[boundary] = 2
    return semantic


# ----------------------------------------------------------------------
# Augmentation (simulate offline / scanned conditions)
# ----------------------------------------------------------------------

def _apply_offline_augmentation(
    image: np.ndarray,
    instance_mask: np.ndarray,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random augmentations that move synthetic renderings closer
    to real scanned offline-paper conditions.

    Geometric augs touch BOTH image and instance mask (with nearest
    interpolation on the mask). Photometric augs only touch the image.
    """
    h, w = image.shape

    # Vertical translation (most lines have slack at top/bottom)
    if rng.rand() < 0.5:
        shift = rng.randint(-3, 4)
        if shift != 0:
            M = np.float32([[1, 0, 0], [0, 1, shift]])
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            instance_mask = cv2.warpAffine(
                instance_mask.astype(np.int32), M, (w, h),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ).astype(np.int32)

    # Tiny rotation
    if rng.rand() < 0.3:
        angle = rng.uniform(-2.0, 2.0)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        instance_mask = cv2.warpAffine(
            instance_mask.astype(np.int32), M, (w, h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ).astype(np.int32)

    # Gaussian blur (simulate ink bleed / scan softness)
    if rng.rand() < 0.5:
        sigma = rng.uniform(0.3, 1.2)
        ksize = int(2 * round(2 * sigma) + 1)
        image = cv2.GaussianBlur(image, (ksize, ksize), sigma)

    # Light salt-and-pepper noise (paper specks)
    if rng.rand() < 0.4:
        noise = rng.rand(h, w)
        salt = noise < 0.002
        pepper = noise > 0.998
        image = image.copy()
        image[salt] = np.maximum(image[salt], rng.uniform(0.3, 0.8))
        image[pepper] = np.minimum(image[pepper], rng.uniform(0.0, 0.3))

    # Contrast / brightness jitter
    if rng.rand() < 0.5:
        gain = rng.uniform(0.7, 1.3)
        bias = rng.uniform(-0.05, 0.05)
        image = np.clip(image * gain + bias, 0.0, 1.0)

    return image, instance_mask


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

class StrokeSegDataset(Dataset):
    """Wrap a TrajectoryDataset (canvas-normalized) into a stroke-instance
    segmentation dataset by rendering trajectories to images + per-stroke
    masks. Optionally applies offline-style augmentation.
    """

    def __init__(
        self,
        traj_dataset: TrajectoryDataset,
        thickness_range: Tuple[int, int] = (3, 7),
        augment: bool = True,
        seed: int = 0,
    ):
        assert traj_dataset.normalization_mode == "canvas", (
            "StrokeSegDataset needs trajectory in canvas-pixel space; "
            "instantiate TrajectoryDataset with normalization_mode='canvas'."
        )
        self.traj = traj_dataset
        self.thickness_range = thickness_range
        self.augment = augment
        self.canvas_h = int(traj_dataset.img_height)
        self.canvas_w = int(traj_dataset.img_width)
        self.rng_master = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.traj)

    def __getitem__(self, idx: int) -> Dict:
        _, traj_tensor, traj_len, meta = self.traj[idx]
        traj = traj_tensor.numpy()[:traj_len]

        strokes = extract_pen_down_strokes(traj)
        n_strokes = len(strokes)

        rng = np.random.RandomState(self.rng_master.randint(0, 2**31 - 1))
        if self.augment:
            thickness = int(rng.randint(self.thickness_range[0],
                                         self.thickness_range[1] + 1))
        else:
            thickness = int(round(0.5 * (self.thickness_range[0] +
                                         self.thickness_range[1])))

        instance_mask = _rasterize_strokes(
            strokes, self.canvas_h, self.canvas_w, thickness=thickness,
        )
        image = (instance_mask > 0).astype(np.float32)

        if self.augment:
            image, instance_mask = _apply_offline_augmentation(
                image, instance_mask, rng,
            )

        semantic = _build_semantic_from_instance(instance_mask)

        sample = {
            "image": torch.from_numpy(image).unsqueeze(0).float(),
            "semantic": torch.from_numpy(semantic).long(),
            "instance_mask": torch.from_numpy(instance_mask.astype(np.int32)),
            "n_strokes": int(n_strokes),
            "meta": meta,
        }
        return sample


def build_canvas_traj_dataset(
    meta_csv_path: str,
    img_size: Tuple[int, int] = (64, 1600),
    base_path: str = config.DATA_ROOT_STR,
) -> TrajectoryDataset:
    """Convenience: build a TrajectoryDataset in canvas mode for use with
    StrokeSegDataset. Distance-thinning is on (we want clean rasterization).
    """
    return TrajectoryDataset(
        meta_csv_path=meta_csv_path,
        img_size=img_size,
        base_path=base_path,
        distance_thin=True,
        pen_down_min_dist_px=4.0,    # finer than default - we WANT dense points for rasterization
        pen_up_min_dist_px=100.0,
        normalize_coords=True,
        normalization_mode="canvas",
    )


def line_level_split_indices(
    n_samples: int, val_frac: float = 0.1, seed: int = 0,
) -> Tuple[List[int], List[int]]:
    """Simple line-level split: each row in the meta csv is one line, so
    permuting row indices is already a line-level split.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    n_val = int(n_samples * val_frac)
    val = sorted(perm[:n_val].tolist())
    train = sorted(perm[n_val:].tolist())
    return train, val


if __name__ == "__main__":
    # Quick sanity check
    csv_path = config.MASTER_RAW_CSV
    base = build_canvas_traj_dataset(csv_path)
    ds = StrokeSegDataset(base, augment=False)
    s = ds[0]
    print(f"image: {s['image'].shape} dtype={s['image'].dtype}")
    print(f"semantic: {s['semantic'].shape} unique={s['semantic'].unique().tolist()}")
    print(f"instance: {s['instance_mask'].shape} n_strokes={s['n_strokes']}")
