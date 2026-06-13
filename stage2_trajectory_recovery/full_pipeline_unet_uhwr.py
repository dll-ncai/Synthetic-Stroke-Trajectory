"""
Phase 10 - end-to-end offline-to-online recovery on UHWR scans.

Replaces classical CC detection with the learned U-Net stroke segmenter.
Everything downstream (classifier embeddings, KNN retrieval, skeleton refine,
learned stroke order) reuses the Phase 9 modules — this file just wires
them together with U-Net detection. NO EDITS to existing files.

Pipeline:
    UHWR scan (any size)
        -> preprocess (auto-polarity, letterbox 64x1600)
        -> U-Net 3-class softmax (bg / interior / boundary)
        -> threshold + cut boundaries -> per-stroke instance masks
        -> for each instance: bbox, mask, render-to-patch
        -> CNN embedding (Phase 7 classifier, penultimate layer)
        -> KNN retrieval over 83,711-stroke training index
        -> stitch retrieved template into stroke bbox
        -> skeleton-snap template to actual ink centerline
        -> learned order predictor sorts strokes
    Output: ordered list of trajectories on the canvas.

Usage:
    python3 phase10/full_pipeline_unet_uhwr.py
    python3 phase10/full_pipeline_unet_uhwr.py --n-samples 6
    python3 phase10/full_pipeline_unet_uhwr.py --image-files 03461.jpg,05350.jpg
"""

from __future__ import annotations
import config

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
# Reuse existing pipeline modules
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARENT_DIR / "phase5b"))
sys.path.insert(0, str(PARENT_DIR / "phase7"))
sys.path.insert(0, str(PARENT_DIR / "phase9"))
sys.path.insert(0, str(THIS_DIR))

from cnn_classifier import StrokeCNNClassifier
from template_stitcher import (
    stitch_template_into_bbox, refine_with_skeleton,
    diacritic_template_at_center,
)
from hybrid_inference import render_component_to_patch
from knn_retrieval import make_embedding_extractor, knn_retrieve
from order_predictor import load_order_predictor, extract_spatial_features

# Phase 10 detector
from unet import StrokeUNet
from infer_unet import predict_semantic, extract_instances
from test_uhwr import preprocess_uhwr_image


DEFAULT_UHWR_DIR = config.UHWR_IMAGES_DIR
UNET_CKPT_DEFAULT = config.UNET_DEPLOY
CLASSIFIER_CKPT = config.SHAPE_CNN
VOCAB_DIR = config.VOCAB_DIR_K100
KNN_INDEX = config.KNN_INDEX
ORDER_CKPT = config.ORDER_PREDICTOR
OUT_DIR = THIS_DIR / "uhwr_full_pipeline"


# ----------------------------------------------------------------------
# Instance mask -> component dict (matches Phase 9 expected schema)
# ----------------------------------------------------------------------

def instance_mask_to_components(
    instance_mask: np.ndarray,
    min_area: int = 15,
) -> List[Dict]:
    """Convert U-Net (H, W) instance mask into a list of Component dicts.

    Each dict matches the schema used by Phase 9's pipeline:
        mask:     (H, W) uint8 with 1 = this stroke's ink
        bbox:     (x0, y0, x1, y1) inclusive bbox in canvas pixels
        area:     int — ink pixel count
        x_center: float — centroid x
        y_center: float — centroid y
        extent:   (w, h)
    """
    components: List[Dict] = []
    H, W = instance_mask.shape
    n_inst = int(instance_mask.max())
    for k in range(1, n_inst + 1):
        mask_k = (instance_mask == k).astype(np.uint8)
        area = int(mask_k.sum())
        if area < min_area:
            continue
        ys, xs = np.where(mask_k > 0)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        cx = float(xs.mean())
        cy = float(ys.mean())
        components.append({
            "mask": mask_k,
            "bbox": (x0, y0, x1, y1),
            "area": area,
            "x_center": cx,
            "y_center": cy,
            "extent": (int(x1 - x0), int(y1 - y0)),
        })
    return components


# ----------------------------------------------------------------------
# Full pipeline on one UHWR image
# ----------------------------------------------------------------------

def run_pipeline_on_uhwr(
    image_path: Path,
    unet: StrokeUNet,
    classifier: StrokeCNNClassifier,
    extractor,
    index_embeddings: np.ndarray,
    index_trajectories: np.ndarray,
    order_model,
    device: torch.device,
    thresh_interior: float = 0.4,
    thresh_boundary: float = 0.5,
    min_area: int = 15,
    use_skeleton_refine: bool = True,
    snap_radius_px: float = 10.0,
) -> Dict:
    """Full offline-to-online pipeline starting from a raw UHWR file."""
    raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return {"error": f"could not read {image_path}"}

    # 1. Preprocess (auto-polarity, letterbox 64x1600)
    image, meta = preprocess_uhwr_image(raw)
    H, W = image.shape

    # 2. U-Net detection
    probs = predict_semantic(unet, image, device)
    instance_mask = extract_instances(
        probs,
        thresh_interior=thresh_interior,
        thresh_boundary=thresh_boundary,
        min_area=min_area,
    )
    components = instance_mask_to_components(instance_mask, min_area=min_area)
    n_blobs = len(components)
    if n_blobs == 0:
        return {
            "raw": raw, "image": image, "instance_mask": instance_mask,
            "components": [], "final_strokes": [], "order_scores": np.zeros(0),
        }

    # 3. Render each blob to a clean patch + embed
    patches = np.stack([render_component_to_patch(c) for c in components])
    batch = torch.from_numpy(patches).unsqueeze(1).float().to(device)
    with torch.no_grad():
        embeddings = extractor(batch).cpu().numpy()

    # 4. KNN top-1
    nn_idx = knn_retrieve(embeddings, index_embeddings, k=1).squeeze(1)

    # 5. Stitch + (optionally) skeleton-refine each retrieved trajectory
    retrieved_trajs = []
    refined_trajs = []
    for i, c in enumerate(components):
        template = index_trajectories[int(nn_idx[i])]
        stitched = stitch_template_into_bbox(template, c["bbox"])
        retrieved_trajs.append(stitched)
        if use_skeleton_refine:
            refined = refine_with_skeleton(
                stitched, c["mask"], c["bbox"],
                snap_radius_px=snap_radius_px,
            )
        else:
            refined = stitched
        refined_trajs.append(refined)

    # 6. Learned order
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
        # fallback: right-to-left detection order
        order_scores = -np.array([c["x_center"] for c in components],
                                  dtype=np.float32)
        order_scores = (order_scores - order_scores.min()) / max(
            order_scores.max() - order_scores.min(), 1e-6
        )

    sort_idx = np.argsort(order_scores)
    final_strokes = [refined_trajs[i] for i in sort_idx]

    return {
        "raw": raw, "image": image, "meta": meta, "probs": probs,
        "instance_mask": instance_mask, "components": components,
        "retrieved_trajs": retrieved_trajs, "refined_trajs": refined_trajs,
        "final_strokes": final_strokes,
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


def make_uhwr_figure(out_path, image_name, result):
    """4-row figure: input / U-Net instances / KNN-only / full pipeline."""
    if "error" in result:
        return
    image = result["image"]
    H, W = image.shape
    n_pred = len(result["final_strokes"])
    cmap_pred = plt.get_cmap("turbo", max(n_pred, 1))

    fig, axes = plt.subplots(4, 1, figsize=(15, 9), sharex=True)

    ax = axes[0]
    ax.imshow(image, cmap="gray_r", aspect="auto", extent=[0, W, H, 0])
    ax.set_title(f"{image_name}: preprocessed input (U-Net found {n_pred} strokes)",
                  fontsize=10)
    ax.set_ylabel("y (px)")

    ax = axes[1]
    ax.imshow(colorize_instance(result["instance_mask"]), aspect="auto",
              extent=[0, W, H, 0])
    ax.set_title("U-Net predicted stroke instances (colour = instance id)",
                  fontsize=10)
    ax.set_ylabel("y (px)")

    ax = axes[2]
    ax.set_facecolor("white")
    for i, s in enumerate(result["retrieved_trajs"]):
        if s is None or s.size == 0:
            continue
        ax.plot(s[:, 0], s[:, 1], "-", color=cmap_pred(i), lw=1.8, alpha=0.85)
    ax.set_title(
        "KNN-only ablation: retrieved trajectories in detection order, no refine",
        fontsize=10,
    )
    ax.set_ylabel("y (px)")

    ax = axes[3]
    ax.set_facecolor("white")
    for i, s in enumerate(result["final_strokes"]):
        if s is None or s.size == 0:
            continue
        ax.plot(s[:, 0], s[:, 1], "-", color=cmap_pred(i), lw=2.0, alpha=0.95)
    ax.set_title(
        "FULL PIPELINE: U-Net detect + KNN + skeleton-refine + learned order  "
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
    parser.add_argument("--image-files", type=str, default=None,
                        help="comma-separated filenames in image-dir")
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thresh-interior", type=float, default=0.4)
    parser.add_argument("--thresh-boundary", type=float, default=0.7)
    parser.add_argument("--min-area", type=int, default=2)
    parser.add_argument("--snap-radius-px", type=float, default=10.0)
    parser.add_argument("--no-skeleton-refine", action="store_true")
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
    print(f"[uhwr-pipeline] device={device}")

    # ---- Load U-Net ----
    unet = StrokeUNet(n_classes=3, base_ch=16).to(device)
    ck = torch.load(args.unet_ckpt, map_location=device)
    unet.load_state_dict(ck["state_dict"])
    unet.eval()
    print(f"[uhwr-pipeline] U-Net loaded from {args.unet_ckpt}")

    # ---- Load classifier (for embeddings only) ----
    vocab = np.load(Path(args.vocab_dir) / "vocab.npz")
    K_total = int(vocab["k_total"]) if "k_total" in vocab.files else int(vocab["k"]) + 1
    classifier = StrokeCNNClassifier(n_classes=K_total, hidden=256, dropout=0.0).to(device)
    classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location=device))
    classifier.eval()
    extractor = make_embedding_extractor(classifier)
    print(f"[uhwr-pipeline] classifier loaded from {args.classifier_ckpt}  K_total={K_total}")

    # ---- Load KNN index ----
    idx = np.load(args.knn_index)
    index_embeddings = idx["embeddings"]
    index_trajectories = idx["trajectories"]
    print(f"[uhwr-pipeline] KNN index: {len(index_embeddings)} strokes  "
          f"embed_dim={index_embeddings.shape[1]}")

    # ---- Load order predictor ----
    order_model = None
    order_path = Path(args.order_ckpt)
    if (not args.no_learned_order) and order_path.exists():
        order_model = load_order_predictor(order_path,
                                            embed_dim=index_embeddings.shape[1])
        order_model = order_model.to(device)
        order_model.eval()
        print(f"[uhwr-pipeline] order predictor loaded from {order_path}")
    else:
        print(f"[uhwr-pipeline] using fallback (R-to-L) order")

    # ---- Pick images ----
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
    print(f"[uhwr-pipeline] running on {len(picks)} images")
    print(f"[uhwr-pipeline] thresh_interior={args.thresh_interior}  "
          f"thresh_boundary={args.thresh_boundary}  min_area={args.min_area}  "
          f"skeleton_refine={'OFF' if args.no_skeleton_refine else 'ON'}")

    n_strokes_log = []
    for p in picks:
        result = run_pipeline_on_uhwr(
            p, unet, classifier, extractor,
            index_embeddings, index_trajectories,
            order_model, device,
            thresh_interior=args.thresh_interior,
            thresh_boundary=args.thresh_boundary,
            min_area=args.min_area,
            use_skeleton_refine=not args.no_skeleton_refine,
            snap_radius_px=args.snap_radius_px,
        )
        if "error" in result:
            print(f"  [skip] {p.name}: {result['error']}")
            continue
        out_path = out_dir / f"uhwr_{p.stem}.png"
        make_uhwr_figure(out_path, p.name, result)
        n_strokes = len(result["final_strokes"])
        n_strokes_log.append(n_strokes)
        print(f"  {p.name}: {n_strokes} strokes recovered  -> {out_path.name}")

    print()
    print("=" * 72)
    if n_strokes_log:
        arr = np.asarray(n_strokes_log)
        print(f"[uhwr-pipeline] stroke counts:  mean={arr.mean():.1f}  "
              f"median={np.median(arr):.1f}  min={arr.min()}  max={arr.max()}")
    print(f"[uhwr-pipeline] outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
