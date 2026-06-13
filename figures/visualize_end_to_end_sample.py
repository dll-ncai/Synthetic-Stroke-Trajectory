"""
End-to-end sample visualization for the thesis pipeline.

This script runs one offline line image through:
    1. U-Net stroke segmentation
    2. Instance extraction
    3. KNN stroke retrieval + skeleton refinement
    4. Learned stroke ordering
    5. BiLSTM-CTC recognition

It writes a single multi-panel figure that is easier to drop into a thesis
or slide than the per-stage debug plots.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")
warnings.filterwarnings("ignore", message="Matplotlib currently does not support Arabic natively.*")

sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARENT_DIR / "common"))
sys.path.insert(0, str(PARENT_DIR / "stage1_unet_segmentation"))
sys.path.insert(0, str(PARENT_DIR / "stage2_trajectory_recovery"))
sys.path.insert(0, str(PARENT_DIR / "stage3_order_predictor"))
sys.path.insert(0, str(PARENT_DIR / "stage4_recognizer"))
sys.path.insert(0, str(THIS_DIR))

import config

from infer_unet import colorize_instance, extract_instances, predict_semantic
from order_predictor import extract_spatial_features, load_order_predictor
from recognizer import TrajRecognizer
from template_stitcher import extract_skeleton, refine_with_skeleton, stitch_template_into_bbox
from traj_tokenizer import (
    cap_sequence_length,
    drop_near_duplicates,
    path_b_polylines_to_tokens,
    simplify_trajectory,
)
from unet import StrokeUNet
from vocab_loader import decode_ids, load_vocab
from test_uhwr import preprocess_uhwr_image


DEFAULT_IMAGE_DIR = Path(config.TEST_IMAGE_DIR)
DEFAULT_IMAGE = DEFAULT_IMAGE_DIR / "00063.jpg"
DEFAULT_VOCAB = Path(config.RECOG_VOCAB_JSON)
VOCAB_NPZ = Path(config.VOCAB_DIR_K100) / "vocab.npz"

UNET_CKPT = config.UNET_DEPLOY
CLASSIFIER_CKPT = config.SHAPE_CNN
KNN_INDEX = config.KNN_INDEX
ORDER_CKPT = config.ORDER_PREDICTOR
RECOGNIZER_CKPT = config.RECOG_FINETUNED

_urdu_font_path = next(
    (
        f.fname
        for f in font_manager.fontManager.ttflist
        if "NotoNastaliq" in Path(f.fname).name
    ),
    None,
)
URDU_FONT = font_manager.FontProperties(fname=_urdu_font_path) if _urdu_font_path else None


def greedy_ctc_decode(logits: torch.Tensor, blank_idx: int = 0) -> List[List[int]]:
    log_probs = F.log_softmax(logits, dim=-1)
    preds = log_probs.argmax(dim=-1)  # (B, T)
    decoded: List[List[int]] = []
    for seq in preds:
        prev = None
        out = []
        for tok in seq.tolist():
            if tok != blank_idx and tok != prev:
                out.append(tok)
            prev = tok
        decoded.append(out)
    return decoded


def load_checkpoint(path: str | Path, device: torch.device):
    return torch.load(path, map_location=device, weights_only=False)


def make_embedding_extractor(classifier: torch.nn.Module) -> torch.nn.Module:
    """Return the 256-d penultimate embedding from the CNN classifier."""
    classifier.eval()

    class Extractor(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.features = m.features
            self.pool = m.pool
            self.flatten = m.head[0]
            self.linear1 = m.head[1]
            self.relu = m.head[2]

        @torch.no_grad()
        def forward(self, x):
            feats = self.features(x)
            pooled = self.pool(feats)
            flat = self.flatten(pooled)
            return self.relu(self.linear1(flat))

    return Extractor(classifier)


def knn_retrieve(query_embeddings: np.ndarray,
                 index_embeddings: np.ndarray,
                 k: int = 1) -> np.ndarray:
    """Vectorized L2 nearest-neighbor search."""
    q_norm = (query_embeddings ** 2).sum(axis=1, keepdims=True)
    i_norm = (index_embeddings ** 2).sum(axis=1, keepdims=True).T
    dot = query_embeddings @ index_embeddings.T
    dists = q_norm + i_norm - 2.0 * dot
    if k == 1:
        return dists.argmin(axis=1, keepdims=True)
    idx = np.argpartition(dists, k, axis=1)[:, :k]
    row_dists = np.take_along_axis(dists, idx, axis=1)
    order = np.argsort(row_dists, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def order_skeleton_multi_pass(skeleton_mask: np.ndarray) -> List[np.ndarray]:
    """Lightweight local replacement for the missing legacy skeleton walker."""
    ys, xs = np.where(skeleton_mask > 0)
    if len(xs) == 0:
        return []

    points = {(int(x), int(y)) for x, y in zip(xs, ys)}
    nbrs: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for x, y in points:
        cur = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                q = (x + dx, y + dy)
                if q in points:
                    cur.append(q)
        nbrs[(x, y)] = cur

    remaining = set(points)
    paths: List[np.ndarray] = []

    while remaining:
        endpoints = [
            p for p in remaining
            if sum((q in remaining) for q in nbrs[p]) <= 1
        ]
        start = min(endpoints or list(remaining), key=lambda p: (p[0], p[1]))
        path = [start]
        remaining.remove(start)
        prev = None
        curr = start

        while True:
            candidates = [q for q in nbrs[curr] if q in remaining]
            if not candidates:
                break

            if prev is None:
                nxt = min(candidates, key=lambda p: (math.dist(curr, p), p[0], p[1]))
            else:
                vx0, vy0 = curr[0] - prev[0], curr[1] - prev[1]

                def score(p):
                    vx1, vy1 = p[0] - curr[0], p[1] - curr[1]
                    n0 = math.hypot(vx0, vy0) + 1e-6
                    n1 = math.hypot(vx1, vy1) + 1e-6
                    cos = (vx0 * vx1 + vy0 * vy1) / (n0 * n1)
                    turn_penalty = 1.0 - cos
                    step_penalty = math.hypot(vx1, vy1)
                    return (turn_penalty, step_penalty, p[0], p[1])

                nxt = min(candidates, key=score)

            path.append(nxt)
            remaining.remove(nxt)
            prev, curr = curr, nxt

        paths.append(np.asarray(path, dtype=np.float32))

    return paths


def smooth_polyline(points: np.ndarray, sigma: float) -> np.ndarray:
    """1-D Gaussian smoothing of x and y independently (removes skeleton staircase)."""
    if sigma <= 0 or len(points) < 3:
        return points
    from scipy.ndimage import gaussian_filter1d
    out = np.asarray(points, dtype=np.float32).copy()
    out[:, 0] = gaussian_filter1d(out[:, 0], sigma=sigma, mode="nearest")
    out[:, 1] = gaussian_filter1d(out[:, 1], sigma=sigma, mode="nearest")
    return out


def render_component_to_patch(component: dict,
                              crop_size: Tuple[int, int] = (64, 256),
                              padding_frac: float = 0.06) -> np.ndarray:
    """Render a detected component into the classifier's stroke-patch format."""
    H, W = crop_size
    x0, y0, x1, y1 = component["bbox"]
    blob = component["mask"][y0:y1, x0:x1]
    if blob.sum() == 0:
        return np.zeros(crop_size, dtype=np.float32)

    # Small diacritic-like components: just stamp a dot at the centroid.
    if max(x1 - x0, y1 - y0) < 5:
        canvas = np.zeros(crop_size, dtype=np.float32)
        cx = int(round((x1 - x0) * 0.5))
        cy = int(round((y1 - y0) * 0.5))
        cv2.circle(canvas, (cx, cy), 2, 1.0, thickness=-1)
        return canvas

    skeleton = extract_skeleton(blob)
    paths = order_skeleton_multi_pass(skeleton)
    if not paths:
        # Fallback: resize the mask itself if skeletonization fails.
        resized = cv2.resize(blob.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        return np.clip(resized, 0.0, 1.0)

    all_pts = np.vstack(paths)
    x_min, x_max = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
    y_min, y_max = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())
    src_w = max(x_max - x_min, 1.0)
    src_h = max(y_max - y_min, 1.0)
    avail_w = W * (1.0 - 2.0 * padding_frac)
    avail_h = H * (1.0 - 2.0 * padding_frac)
    scale = min(avail_w / src_w, avail_h / src_h)
    off_x = (W - src_w * scale) * 0.5
    off_y = (H - src_h * scale) * 0.5

    canvas = np.zeros((H, W), dtype=np.float32)
    for path in paths:
        pts = np.empty_like(path, dtype=np.float32)
        pts[:, 0] = (path[:, 0] - x_min) * scale + off_x
        pts[:, 1] = (path[:, 1] - y_min) * scale + off_y
        if len(pts) == 1:
            cv2.circle(canvas, tuple(pts[0].astype(int)), 2, 1.0, thickness=-1)
        else:
            cv2.polylines(
                canvas,
                [pts.astype(np.int32).reshape(-1, 1, 2)],
                isClosed=False,
                color=1.0,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
    return canvas


def make_patch_montage(patches: Sequence[np.ndarray],
                       captions: Sequence[str],
                       tile_size: Tuple[int, int] = (64, 256),
                       max_tiles: int = 6) -> np.ndarray:
    """Create a simple montage for a small number of stroke patches."""
    if not patches:
        return np.ones((tile_size[0], tile_size[1], 3), dtype=np.float32)

    n = min(len(patches), max_tiles)
    cols = 2
    rows = int(math.ceil(n / cols))
    pad = 8
    th, tw = tile_size
    canvas = np.ones((rows * (th + pad) + pad, cols * (tw + pad) + pad, 3), dtype=np.float32)

    for i in range(n):
        r = i // cols
        c = i % cols
        y0 = pad + r * (th + pad)
        x0 = pad + c * (tw + pad)
        patch = patches[i]
        if patch.ndim == 2:
            rgb = np.repeat(patch[:, :, None], 3, axis=2)
        else:
            rgb = patch
        rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
        canvas[y0:y0 + th, x0:x0 + tw] = rgb
        cv2.rectangle(canvas, (x0, y0), (x0 + tw, y0 + th), (0.82, 0.82, 0.82), 1)
        label = captions[i]
        cv2.putText(canvas, label, (x0 + 5, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0.1, 0.1, 0.1), 1, cv2.LINE_AA)
    return canvas


def render_strokes_canvas(strokes: Sequence[np.ndarray],
                          canvas_size: Tuple[int, int],
                          offset_xy: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Render ordered strokes as a raster image, one color per stroke."""
    H, W = canvas_size
    canvas = np.ones((H, W, 3), dtype=np.float32)
    if not strokes:
        return canvas
    cmap = plt.get_cmap("turbo", max(len(strokes), 1))
    for i, stroke in enumerate(strokes):
        if stroke is None or len(stroke) == 0:
            continue
        pts = np.asarray(stroke, dtype=np.float32).copy()
        pts[:, 0] -= float(offset_xy[0])
        pts[:, 1] -= float(offset_xy[1])
        color = tuple(float(c) for c in cmap(i)[:3])
        if len(pts) == 1:
            cv2.circle(canvas, tuple(pts[0].astype(int)), 2, color, thickness=-1)
        else:
            cv2.polylines(
                canvas,
                [pts.astype(np.int32).reshape(-1, 1, 2)],
                isClosed=False,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
    return canvas


def content_bbox(gray: np.ndarray, pad: int = 24, threshold: int = 245) -> Tuple[int, int, int, int]:
    """Tight crop around the dark ink pixels, expanded by `pad`."""
    ys, xs = np.where(gray < threshold)
    H, W = gray.shape
    if len(xs) == 0:
        return 0, 0, W, H
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, W)
    y1 = min(int(ys.max()) + pad + 1, H)
    return x0, y0, x1, y1


def component_bbox(components: Sequence[Dict], canvas_size: Tuple[int, int], pad: int = 24) -> Tuple[int, int, int, int]:
    """Tight crop around all detected component boxes."""
    H, W = canvas_size
    if not components:
        return 0, 0, W, H
    x0 = min(int(c["bbox"][0]) for c in components)
    y0 = min(int(c["bbox"][1]) for c in components)
    x1 = max(int(c["bbox"][2]) for c in components)
    y1 = max(int(c["bbox"][3]) for c in components)
    return (
        max(x0 - pad, 0),
        max(y0 - pad, 0),
        min(x1 + pad, W),
        min(y1 + pad, H),
    )


def crop2d(arr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return arr[y0:y1, x0:x1]


def run_pipeline(
    image_path: Path,
    device: torch.device,
    unet: StrokeUNet,
    classifier: torch.nn.Module,
    extractor,
    index_embeddings: np.ndarray,
    index_trajectories: np.ndarray,
    order_model,
    recognizer: TrajRecognizer,
    char_to_idx: Dict[str, int],
    thresh_interior: float,
    thresh_boundary: float,
    min_area: int,
    smooth_sigma: float,
    max_points: int,
) -> Dict:
    raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(image_path)

    image, meta = preprocess_uhwr_image(raw)
    probs = predict_semantic(unet, image, device)
    inst = extract_instances(
        probs,
        thresh_interior=thresh_interior,
        thresh_boundary=thresh_boundary,
        min_area=min_area,
    )

    components = []
    for k in range(1, int(inst.max()) + 1):
        mask_k = (inst == k).astype(np.uint8)
        area = int(mask_k.sum())
        if area < min_area:
            continue
        ys, xs = np.where(mask_k > 0)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        components.append({
            "mask": mask_k,
            "bbox": (x0, y0, x1, y1),
            "area": area,
            "x_center": float(xs.mean()),
            "y_center": float(ys.mean()),
            "extent": (int(x1 - x0), int(y1 - y0)),
        })

    if not components:
        return {
            "raw": raw,
            "image": image,
            "meta": meta,
            "probs": probs,
            "instance_mask": inst,
            "components": [],
            "patches": [],
            "nn_idx": np.zeros(0, dtype=np.int64),
            "retrieved": [],
            "refined": [],
            "order_scores": np.zeros(0, dtype=np.float32),
            "sort_idx": np.zeros(0, dtype=np.int64),
            "tokens": np.zeros((0, 3), dtype=np.float32),
            "pred_text": "",
            "final_strokes": [],
            "thresh_interior": thresh_interior,
            "thresh_boundary": thresh_boundary,
        }

    # Stage 3: render each component into the classifier's patch format.
    patches = np.stack([render_component_to_patch(c) for c in components], axis=0)
    batch = torch.from_numpy(patches).unsqueeze(1).float().to(device)
    with torch.no_grad():
        embeddings = extractor(batch).cpu().numpy()

    # Learned stroke order (shape embedding + spatial features).
    spatial = np.zeros((len(components), 6), dtype=np.float32)
    H, W = image.shape
    for i, c in enumerate(components):
        spatial[i] = extract_spatial_features(c["bbox"], len(components), canvas_size=(H, W))

    with torch.no_grad():
        emb_t = torch.from_numpy(embeddings).float().to(device)
        sp_t = torch.from_numpy(spatial).float().to(device)
        order_scores = order_model(emb_t, sp_t).cpu().numpy()
    sort_idx = np.argsort(order_scores)

    # Trajectory recovery: skeleton-walk each instance in writer order, exactly
    # as the canonical PathBExtractor pipeline does (thesis 3.4). This is what
    # the recognizer was trained on -- the previous template-retrieval path here
    # produced distorted strokes that didn't match the recognizer's input.
    polylines_per_stroke: List[List[np.ndarray]] = []
    final_strokes: List[np.ndarray] = []
    for ci in sort_idx:
        c = components[int(ci)]
        bx0, by0, bx1, by1 = c["bbox"]
        blob = c["mask"][by0:by1, bx0:bx1]
        if blob.sum() < 3:
            pt = np.array([[c["x_center"], c["y_center"]]], dtype=np.float32)
            polylines_per_stroke.append([pt]); final_strokes.append(pt); continue
        skel = extract_skeleton(blob)
        if skel.sum() < 2:
            pt = np.array([[c["x_center"], c["y_center"]]], dtype=np.float32)
            polylines_per_stroke.append([pt]); final_strokes.append(pt); continue
        subs = []
        for sp in order_skeleton_multi_pass(skel):
            sp = smooth_polyline(np.asarray(sp, dtype=np.float32), sigma=smooth_sigma)
            canvas = sp.copy()
            canvas[:, 0] += bx0
            canvas[:, 1] += by0
            subs.append(canvas)
            final_strokes.append(canvas)
        polylines_per_stroke.append(subs)

    # Tokenize for recognizer.
    tokens = path_b_polylines_to_tokens(polylines_per_stroke)
    tokens = simplify_trajectory(tokens, target_step=0.117, smooth_sigma=smooth_sigma)
    tokens = drop_near_duplicates(tokens)
    tokens = cap_sequence_length(tokens, max_T=max_points)

    # Recognition.
    if tokens.shape[0] > 0:
        x = torch.from_numpy(tokens).unsqueeze(0).float().to(device)
        lens = torch.tensor([tokens.shape[0]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = recognizer(x, lens)
        decoded = greedy_ctc_decode(logits, blank_idx=0)[0]
        pred_text = decode_ids(decoded, char_to_idx)
    else:
        pred_text = ""

    return {
        "raw": raw,
        "image": image,
        "meta": meta,
        "probs": probs,
        "instance_mask": inst,
        "components": components,
        "patches": patches,
        "final_strokes": final_strokes,
        "order_scores": order_scores,
        "sort_idx": sort_idx,
        "tokens": tokens,
        "pred_text": pred_text,
        "thresh_interior": thresh_interior,
        "thresh_boundary": thresh_boundary,
    }


def make_figure(out_path: Path, image_name: str, result: Dict) -> None:
    raw = result["raw"]
    image = result["image"]
    probs = result["probs"]
    inst = result["instance_mask"]
    comps = result["components"]
    final_strokes = result["final_strokes"]

    H, W = image.shape
    n = len(comps)
    bbox = component_bbox(comps, canvas_size=(H, W), pad=24)
    if bbox[2] - bbox[0] <= 1 or bbox[3] - bbox[1] <= 1:
        bbox = content_bbox(image, pad=20)
    x0, y0, x1, y1 = bbox
    crop_h, crop_w = y1 - y0, x1 - x0
    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=False)
    fig.subplots_adjust(hspace=0.36, top=0.95, bottom=0.05)

    ax = axes[0]
    ax.imshow(crop2d(image, bbox), cmap="gray_r", aspect="auto")
    ax.set_title("Input  (preprocessed to 64x1600)", fontsize=18, pad=14)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(False)

    ax = axes[1]
    # Render the actual predicted 3-class mask, using the same thresholds the
    # downstream cut uses (so this panel is consistent with the instance panel).
    # `image`/`probs` are float in [0, 1]; do NOT divide by 255.
    ti = result.get("thresh_interior", 0.4)
    tb = result.get("thresh_boundary", 0.7)
    interior_p = crop2d(probs[1], bbox)
    boundary_p = crop2d(probs[2], bbox)
    ink = (interior_p + boundary_p) > ti          # predicted stroke pixels
    bnd = boundary_p > tb                          # predicted inter-stroke boundary
    seg = np.ones((*interior_p.shape, 3), dtype=np.float32)  # background = white
    seg[ink] = [0.15, 0.15, 0.15]                  # interior = dark grey
    seg[bnd] = [1.0, 0.0, 0.0]                      # boundary = red (drawn on top)
    ax.imshow(seg, aspect="auto")
    ax.set_title("U-Net 3-class output  (grey = interior, red = predicted boundary)",
                 fontsize=18, pad=14)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(False)

    ax = axes[2]
    ax.imshow(crop2d(colorize_instance(inst), bbox), aspect="auto")
    ax.set_title(f"Instance segmentation -- {n} predicted", fontsize=18, pad=14)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(False)

    ax = axes[3]
    recovered = render_strokes_canvas(final_strokes, canvas_size=(crop_h, crop_w), offset_xy=(x0, y0))
    ax.imshow(recovered, aspect="auto")
    ax.set_title("Recovered trajectory  (ordered, rasterized at 2 px thickness)", fontsize=18, pad=14)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(False)

    fig.suptitle(
        f"{image_name}   |   recovered strokes={n}",
        fontsize=13,
        y=0.995,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE),
                        help="path to a single grayscale Urdu line image")
    parser.add_argument("--image-dir", type=str, default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--out", type=str,
                        default=str(THIS_DIR / "end_to_end_sample.png"))
    parser.add_argument("--vocab", type=str, default=str(DEFAULT_VOCAB))
    parser.add_argument("--unet-ckpt", type=str, default=str(UNET_CKPT))
    parser.add_argument("--classifier-ckpt", type=str, default=str(CLASSIFIER_CKPT))
    parser.add_argument("--knn-index", type=str, default=str(KNN_INDEX))
    parser.add_argument("--order-ckpt", type=str, default=str(ORDER_CKPT))
    parser.add_argument("--recognizer-ckpt", type=str, default=str(RECOGNIZER_CKPT))
    parser.add_argument("--thresh-interior", type=float, default=0.4)
    parser.add_argument("--thresh-boundary", type=float, default=0.7)
    parser.add_argument("--min-area", type=int, default=2)
    parser.add_argument("--smooth-sigma", type=float, default=0.6)
    parser.add_argument("--max-points", type=int, default=2000)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "mps":
        device = torch.device("mps")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    image_path = Path(args.image)
    if not image_path.exists():
        image_dir = Path(args.image_dir)
        candidates = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
        if not candidates:
            raise FileNotFoundError(f"no images found in {image_dir}")
        image_path = candidates[0]

    char_to_idx = load_vocab(args.vocab)
    vocab_size = len(char_to_idx) + 1

    unet = StrokeUNet(n_classes=3, base_ch=16).to(device)
    unet.load_state_dict(load_checkpoint(args.unet_ckpt, device)["state_dict"])
    unet.eval()

    idx = np.load(args.knn_index)
    index_embeddings = idx["embeddings"]
    index_trajectories = idx["trajectories"]

    # Load the classifier used for KNN embeddings.
    vocab_npz = np.load(VOCAB_NPZ)
    K_total = int(vocab_npz["k_total"]) if "k_total" in vocab_npz.files else int(vocab_npz["k"]) + 1
    from cnn_classifier import StrokeCNNClassifier

    classifier = StrokeCNNClassifier(n_classes=K_total, hidden=256, dropout=0.0).to(device)
    classifier.load_state_dict(load_checkpoint(args.classifier_ckpt, device))
    classifier.eval()
    extractor = make_embedding_extractor(classifier)

    order_model = load_order_predictor(Path(args.order_ckpt), embed_dim=index_embeddings.shape[1]).to(device)
    order_model.eval()

    recognizer = TrajRecognizer(vocab_size_with_blank=vocab_size, input_dim=3).to(device)
    recognizer.load_state_dict(load_checkpoint(args.recognizer_ckpt, device)["state_dict"])
    recognizer.eval()

    result = run_pipeline(
        image_path=image_path,
        device=device,
        unet=unet,
        classifier=classifier,
        extractor=extractor,
        index_embeddings=index_embeddings,
        index_trajectories=index_trajectories,
        order_model=order_model,
        recognizer=recognizer,
        char_to_idx=char_to_idx,
        thresh_interior=args.thresh_interior,
        thresh_boundary=args.thresh_boundary,
        min_area=args.min_area,
        smooth_sigma=args.smooth_sigma,
        max_points=args.max_points,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_figure(out_path, image_path.name, result)

    print(f"[viz] image: {image_path}")
    print(f"[viz] components: {len(result['components'])}")
    print(f"[viz] predicted text: {result['pred_text']}")
    print(f"[viz] wrote: {out_path}")


if __name__ == "__main__":
    main()
