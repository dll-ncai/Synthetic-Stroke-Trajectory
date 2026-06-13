"""
Phase 10 - UHWR pipeline using SKELETON of U-Net predicted masks
            as the trajectory (no KNN, no template retrieval).

Why this is right for UHWR:
    The U-Net's per-instance mask already follows the writer's actual ink.
    Skeletonizing that mask gives the writer's actual stroke centerline.
    No need to retrieve a "canonical" template from online training data
    when we have the real geometry right there in the predicted mask.

Pipeline:
    UHWR scan
        -> preprocess (auto-polarity, letterbox 64x1600)
        -> U-Net -> per-stroke instance masks
        -> for each instance:
              skeletonize the mask
              order skeleton pixels along the curve  (graph-based traversal)
              that IS this stroke's trajectory
        -> embed each stroke (for the order predictor only)
        -> learned order predictor sorts strokes by predicted temporal order

Output: ordered list of (T_i, 2) point arrays - the recovered trajectory.

Usage:
    python3 phase10/full_pipeline_unet_uhwr_skeleton.py --n-samples 6
"""

from __future__ import annotations
import config

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARENT_DIR / "phase5b"))
sys.path.insert(0, str(PARENT_DIR / "phase7"))
sys.path.insert(0, str(PARENT_DIR / "phase9"))
sys.path.insert(0, str(THIS_DIR))

from cnn_classifier import StrokeCNNClassifier
from template_stitcher import extract_skeleton
from hybrid_inference import render_component_to_patch
from knn_retrieval import make_embedding_extractor
from order_predictor import load_order_predictor, extract_spatial_features

from unet import StrokeUNet
from infer_unet import predict_semantic, extract_instances
from test_uhwr import preprocess_uhwr_image
from full_pipeline_unet_uhwr import instance_mask_to_components


DEFAULT_UHWR_DIR = config.UHWR_IMAGES_DIR
UNET_CKPT_DEFAULT = config.UNET_DEPLOY
CLASSIFIER_CKPT = config.SHAPE_CNN
VOCAB_DIR = config.VOCAB_DIR_K100
KNN_INDEX = config.KNN_INDEX
ORDER_CKPT = config.ORDER_PREDICTOR
OUT_DIR = THIS_DIR / "uhwr_skeleton_pipeline"


# ----------------------------------------------------------------------
# Skeleton-to-ordered-polyline conversion
# ----------------------------------------------------------------------

def order_skeleton_points(skeleton_mask: np.ndarray) -> np.ndarray:
    """Convert a 1px-thick skeleton into an ordered polyline.

    Strategy:
        1. Find all skeleton pixels.
        2. Build adjacency on 8-connectivity.
        3. Find endpoints (degree-1 nodes) - if any, pick the leftmost as start.
           If none (closed loop), pick any pixel as start.
        4. Greedy traversal: walk along neighbours, never revisiting.

    For a clean stroke without branches, this gives the natural ordering.
    For branched skeletons (T-junctions where two strokes still touch
    after U-Net cuts), the greedy walk picks one branch and may strand
    the other; acceptable for visualization, not ideal long-term.
    """
    ys, xs = np.where(skeleton_mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(xs) == 1:
        return np.array([[xs[0], ys[0]]], dtype=np.float32)

    pts = list(zip(xs.tolist(), ys.tolist()))
    pt_set = set(pts)
    # Neighbour offsets (8-connectivity)
    NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def neighbours(p):
        return [(p[0] + dx, p[1] + dy) for dx, dy in NB
                if (p[0] + dx, p[1] + dy) in pt_set]

    # Find endpoints (degree-1 pixels)
    endpoints = [p for p in pts if len(neighbours(p)) == 1]
    if endpoints:
        # Use leftmost endpoint as start
        endpoints.sort(key=lambda p: (p[0], p[1]))
        start = endpoints[0]
    else:
        # Closed loop / branched - just start at leftmost pixel
        start = min(pts, key=lambda p: (p[0], p[1]))

    # Greedy walk: at each step pick the un-visited neighbour with the
    # smallest angle change from the current direction. Falls back to any
    # unvisited neighbour if none aligned.
    visited = {start}
    path = [start]
    prev_dir = (1.0, 0.0)  # arbitrary initial direction

    while True:
        cur = path[-1]
        cands = [n for n in neighbours(cur) if n not in visited]
        if not cands:
            break
        # Pick neighbour minimizing direction change
        best = None
        best_cost = float("inf")
        for c in cands:
            d = (c[0] - cur[0], c[1] - cur[1])
            mag = (d[0] ** 2 + d[1] ** 2) ** 0.5 or 1.0
            d_norm = (d[0] / mag, d[1] / mag)
            # Cost = -dot(prev_dir, d_norm)
            cost = -(prev_dir[0] * d_norm[0] + prev_dir[1] * d_norm[1])
            if cost < best_cost:
                best_cost = cost
                best = c
                prev_dir = d_norm
        path.append(best)
        visited.add(best)

    return np.asarray(path, dtype=np.float32)


def component_to_skeleton_trajectory(
    component: Dict,
    canvas_h: int,
    canvas_w: int,
) -> np.ndarray:
    """Build a canvas-coord polyline trajectory from a component's mask."""
    bbox = component["bbox"]
    bx0, by0, bx1, by1 = bbox
    blob = component["mask"][by0:by1, bx0:bx1]
    if blob.sum() < 3:
        # Tiny - return centroid
        cx = 0.5 * (bx0 + bx1); cy = 0.5 * (by0 + by1)
        return np.array([[cx, cy]], dtype=np.float32)
    skel = extract_skeleton(blob)
    if skel.sum() < 2:
        # Skeletonization failed; fall back to mask centerline along x
        return np.array([[component["x_center"], component["y_center"]]],
                         dtype=np.float32)
    local = order_skeleton_points(skel)
    # Translate from local crop coords back to canvas coords
    canvas_pts = local.copy()
    canvas_pts[:, 0] += bx0
    canvas_pts[:, 1] += by0
    return canvas_pts


# ----------------------------------------------------------------------
# Full skeleton pipeline on one UHWR image
# ----------------------------------------------------------------------

def run_skeleton_pipeline(
    image_path: Path,
    unet: StrokeUNet,
    classifier: StrokeCNNClassifier,
    extractor,
    order_model,
    device: torch.device,
    thresh_interior: float = 0.4,
    thresh_boundary: float = 0.5,
    min_area: int = 15,
) -> Dict:
    raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return {"error": f"could not read {image_path}"}

    image, meta = preprocess_uhwr_image(raw)
    H, W = image.shape

    # 1. U-Net detection
    probs = predict_semantic(unet, image, device)
    instance_mask = extract_instances(
        probs, thresh_interior, thresh_boundary, min_area,
    )
    components = instance_mask_to_components(instance_mask, min_area)
    n_blobs = len(components)
    if n_blobs == 0:
        return {
            "raw": raw, "image": image, "instance_mask": instance_mask,
            "components": [], "final_strokes": [],
            "order_scores": np.zeros(0),
        }

    # 2. Skeletonize each instance -> ordered polyline
    skel_trajs = [component_to_skeleton_trajectory(c, H, W) for c in components]

    # 3. Order: use the existing learned order predictor (needs CNN embeddings)
    patches = np.stack([render_component_to_patch(c) for c in components])
    batch = torch.from_numpy(patches).unsqueeze(1).float().to(device)
    with torch.no_grad():
        embeddings = extractor(batch).cpu().numpy()

    spatial = np.zeros((n_blobs, 6), dtype=np.float32)
    for i, c in enumerate(components):
        spatial[i] = extract_spatial_features(
            c["bbox"], n_blobs, canvas_size=(H, W),
        )
    if order_model is not None:
        with torch.no_grad():
            emb_t = torch.from_numpy(embeddings).float().to(device)
            sp_t = torch.from_numpy(spatial).float().to(device)
            order_scores = order_model(emb_t, sp_t).cpu().numpy()
    else:
        # Fallback: right-to-left by x-center
        xs = np.array([c["x_center"] for c in components], dtype=np.float32)
        order_scores = -xs
        order_scores = (order_scores - order_scores.min()) / max(
            order_scores.max() - order_scores.min(), 1e-6
        )

    sort_idx = np.argsort(order_scores)
    final_strokes = [skel_trajs[i] for i in sort_idx]

    return {
        "raw": raw, "image": image, "meta": meta, "probs": probs,
        "instance_mask": instance_mask, "components": components,
        "skel_trajs": skel_trajs, "final_strokes": final_strokes,
        "order_scores": order_scores, "sort_idx": sort_idx,
    }


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------

def colorize_instance(instance_mask: np.ndarray) -> np.ndarray:
    h, w = instance_mask.shape
    n_inst = int(instance_mask.max())
    rng = np.random.RandomState(0)
    palette = rng.uniform(0.2, 1.0, size=(n_inst + 1, 3))
    palette[0] = [1.0, 1.0, 1.0]
    return palette[instance_mask.reshape(-1)].reshape(h, w, 3)


def make_figure(out_path, image_name, result):
    if "error" in result:
        return
    image = result["image"]
    H, W = image.shape
    n_pred = len(result["final_strokes"])
    cmap = plt.get_cmap("turbo", max(n_pred, 1))

    fig, axes = plt.subplots(3, 1, figsize=(15, 7), sharex=True)

    ax = axes[0]
    ax.imshow(image, cmap="gray_r", aspect="auto", extent=[0, W, H, 0])
    ax.set_title(f"{image_name}: preprocessed input ({n_pred} strokes recovered)",
                  fontsize=10)
    ax.set_ylabel("y (px)")

    ax = axes[1]
    ax.imshow(colorize_instance(result["instance_mask"]), aspect="auto",
              extent=[0, W, H, 0])
    ax.set_title("U-Net per-stroke instance masks", fontsize=10)
    ax.set_ylabel("y (px)")

    ax = axes[2]
    ax.set_facecolor("white")
    for i, s in enumerate(result["final_strokes"]):
        if s is None or s.size == 0:
            continue
        ax.plot(s[:, 0], s[:, 1], "-", color=cmap(i), lw=2.0, alpha=0.95)
    ax.set_title(
        "RECOVERED TRAJECTORY: skeleton-of-U-Net-mask + learned order  "
        "(colour = predicted stroke order)",
        fontsize=10,
    )
    ax.set_xlabel("x (canvas px)")
    ax.set_ylabel("y (px)")

    for a in axes:
        a.set_xlim(0, W); a.set_ylim(H, 0)
        a.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet-ckpt", type=str, default=str(UNET_CKPT_DEFAULT))
    parser.add_argument("--classifier-ckpt", type=str, default=str(CLASSIFIER_CKPT))
    parser.add_argument("--vocab-dir", type=str, default=str(VOCAB_DIR))
    parser.add_argument("--knn-index", type=str, default=str(KNN_INDEX))
    parser.add_argument("--order-ckpt", type=str, default=str(ORDER_CKPT))
    parser.add_argument("--image-dir", type=str, default=DEFAULT_UHWR_DIR)
    parser.add_argument("--image-files", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thresh-interior", type=float, default=0.4)
    parser.add_argument("--thresh-boundary", type=float, default=0.7)
    parser.add_argument("--min-area", type=int, default=2)
    parser.add_argument("--no-learned-order", action="store_true")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[uhwr-skel] device={device}")

    unet = StrokeUNet(n_classes=3, base_ch=16).to(device)
    ck = torch.load(args.unet_ckpt, map_location=device)
    unet.load_state_dict(ck["state_dict"])
    unet.eval()
    print(f"[uhwr-skel] U-Net loaded from {args.unet_ckpt}")

    vocab = np.load(Path(args.vocab_dir) / "vocab.npz")
    K_total = int(vocab["k_total"]) if "k_total" in vocab.files else int(vocab["k"]) + 1
    classifier = StrokeCNNClassifier(n_classes=K_total, hidden=256, dropout=0.0).to(device)
    classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location=device))
    classifier.eval()
    extractor = make_embedding_extractor(classifier)

    idx = np.load(args.knn_index)
    embed_dim = int(idx["embeddings"].shape[1])

    order_model = None
    order_path = Path(args.order_ckpt)
    if (not args.no_learned_order) and order_path.exists():
        order_model = load_order_predictor(order_path, embed_dim=embed_dim)
        order_model = order_model.to(device).eval()
        print(f"[uhwr-skel] order predictor loaded")

    img_dir = Path(args.image_dir)
    all_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not all_files:
        print(f"[error] no images in {img_dir}")
        return
    if args.image_files is not None:
        names = [s.strip() for s in args.image_files.split(",")]
        picks = [img_dir / n for n in names if (img_dir / n).exists()]
    else:
        rng = np.random.RandomState(args.seed)
        picks = rng.choice(all_files,
                            size=min(args.n_samples, len(all_files)),
                            replace=False).tolist()
    print(f"[uhwr-skel] running on {len(picks)} images")
    print(f"[uhwr-skel] thresh_interior={args.thresh_interior}  "
          f"thresh_boundary={args.thresh_boundary}  min_area={args.min_area}")

    counts = []
    for p in picks:
        result = run_skeleton_pipeline(
            p, unet, classifier, extractor, order_model, device,
            thresh_interior=args.thresh_interior,
            thresh_boundary=args.thresh_boundary,
            min_area=args.min_area,
        )
        if "error" in result:
            print(f"  [skip] {p.name}: {result['error']}")
            continue
        out_path = out_dir / f"skel_{p.stem}.png"
        make_figure(out_path, p.name, result)
        n_strokes = len(result["final_strokes"])
        counts.append(n_strokes)
        print(f"  {p.name}: {n_strokes} strokes recovered  -> {out_path.name}")

    print()
    print("=" * 72)
    if counts:
        arr = np.asarray(counts)
        print(f"[uhwr-skel] stroke counts: mean={arr.mean():.1f}  "
              f"median={np.median(arr):.1f}  min={arr.min()}  max={arr.max()}")
    print(f"[uhwr-skel] outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
