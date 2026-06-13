"""
Phase 9 - KNN retrieval-based trajectory recovery (no medoid averaging).

Key idea:
    Don't use K-means cluster medoids (generic averages). Instead, for each
    input stroke, find the NEAREST TRAINING STROKE in the 92k training set
    and use ITS actual trajectory. The output is always a real stroke that
    a real writer drew, with full writer-specific detail preserved.

Why this should help for Urdu Nastaleeq:
    Same letter shapes appear across many words. With 92k training strokes
    from 240 writers, almost every common shape has many real examples.
    KNN returns one of those real examples instead of a smoothed-out medoid.

Pipeline:
    ONE-TIME (precompute):
        For every training stroke:
            - Render clean per-stroke image (same as Phase 7 trained on)
            - Feed through the trained CNN classifier
            - Extract the penultimate-layer 256-dim embedding
            - Cache (embedding, normalized_trajectory, line_idx, stroke_idx)
        Saves to phase9/knn_index.npz.

    INFERENCE (per stroke):
        - Render clean stroke patch
        - Compute embedding through same network
        - L2 nearest-neighbor search in cached embeddings (vectorized)
        - Retrieve that training stroke's actual normalized trajectory
        - Scale into the target bbox (same stitching as before)

Output: 3-panel figure: image / GT trajectory / KNN-retrieved trajectory.

Usage:
    # Step 1: build the index (one-time, ~5 min)
    python3 phase9/knn_retrieval.py --build-index

    # Step 2: run the demo
    python3 phase9/knn_retrieval.py --random-lines 8
    python3 phase9/knn_retrieval.py --random-lines 8 \\
        --ckpt phase9/classifier_k300_out/model_best_noDIAC.pt \\
        --vocab-dir phase5b/vocab_out_k300
"""

from __future__ import annotations
import config

import argparse
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARENT_DIR / "phase5b"))
sys.path.insert(0, str(PARENT_DIR / "phase7"))
sys.path.insert(0, str(THIS_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from cnn_classifier import StrokeCNNClassifier
from clean_stroke_dataset import render_clean_stroke
from template_stitcher import (
    stitch_template_into_bbox,
    refine_with_skeleton,
    diacritic_template_at_center,
)
from stroke_detector import binarize_image

from trajectory_dataset import TrajectoryDataset
from stroke_sequence_utils import extract_pen_down_strokes


CSV_PATH = config.MAIN_ALL_CSV
CKPT_DEFAULT = (
    config.SHAPE_CNN
)
VOCAB_DIR_DEFAULT = config.VOCAB_DIR_K100
OUT_DIR = THIS_DIR / "knn_demos"
INDEX_DEFAULT = config.KNN_INDEX

EMBED_DIM = 256
TARGET_TRAJ_LEN = 50


def make_embedding_extractor(classifier: StrokeCNNClassifier) -> nn.Module:
    """
    Returns a callable that takes a (B, 1, H, W) batch and returns the
    256-dim penultimate-layer embedding (output of the first Linear+ReLU
    in the MLP head, before final classification).
    """
    classifier.eval()

    class Extractor(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.features = m.features
            self.pool = m.pool
            # First two layers of head: Flatten, Linear, ReLU
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


def resample_trajectory(xy: np.ndarray, target_len: int) -> np.ndarray:
    """Arc-length resampling to fixed length, same as build_vocab.py."""
    T = xy.shape[0]
    if T == 0:
        return np.zeros((target_len, 2), dtype=np.float32)
    if T == 1:
        return np.tile(xy.astype(np.float32), (target_len, 1))
    diffs = np.diff(xy, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg_lens))).astype(np.float64)
    total = float(cum[-1])
    if total < 1e-9:
        return np.tile(xy[:1].astype(np.float32), (target_len, 1))
    targets = np.linspace(0, total, target_len)
    x_new = np.interp(targets, cum, xy[:, 0])
    y_new = np.interp(targets, cum, xy[:, 1])
    return np.stack([x_new, y_new], axis=1).astype(np.float32)


def normalize_stroke_local(xy: np.ndarray) -> np.ndarray:
    """Shared-span normalization (same as build_vocab.py)."""
    x_min, x_max = float(xy[:, 0].min()), float(xy[:, 0].max())
    y_min, y_max = float(xy[:, 1].min()), float(xy[:, 1].max())
    x_range = x_max - x_min
    y_range = y_max - y_min
    span = max(x_range, y_range, 1e-9)
    out = np.empty_like(xy, dtype=np.float32)
    out[:, 0] = (xy[:, 0] - x_min) / span
    out[:, 1] = (xy[:, 1] - y_min) / span
    extent_x = x_range / span
    extent_y = y_range / span
    out[:, 0] += (1.0 - extent_x) / 2.0
    out[:, 1] += (1.0 - extent_y) / 2.0
    return out


def build_knn_index(
    ckpt_path: Path,
    vocab_dir: Path,
    out_path: Path,
    crop_size=(64, 256),
    pen_width: int = 2,
    padding_frac: float = 0.05,
    batch_size: int = 32,
    train_only: bool = True,
):
    """
    Precompute embeddings + normalized trajectories for all (training-side)
    strokes. Saves to out_path as an npz.

    Only TRAIN strokes are included by default — val strokes are excluded
    so retrieval can't accidentally return a near-identical val sample at
    inference, which would inflate quality estimates.
    """
    device = torch.device("cpu")
    print(f"[knn-index] loading vocab from {vocab_dir}")
    vocab = np.load(vocab_dir / "vocab.npz")
    K_total = int(vocab["k_total"]) if "k_total" in vocab.files else int(vocab["k"]) + 1

    print(f"[knn-index] loading classifier from {ckpt_path}")
    classifier = StrokeCNNClassifier(
        n_classes=K_total, hidden=256, dropout=0.0,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    classifier.load_state_dict(state)
    extractor = make_embedding_extractor(classifier)

    assigns = np.load(vocab_dir / "label_assignments.npz")
    line_idx_arr = assigns["line_idx"]
    stroke_idx_arr = assigns["stroke_idx"]

    # Same line-level val split as phase7
    all_lines = sorted({int(li) for li in line_idx_arr})
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(all_lines))
    n_val = int(len(all_lines) * 0.1)
    val_lines_set = {all_lines[i] for i in perm[:n_val]}
    is_train = np.array([int(li) not in val_lines_set for li in line_idx_arr])
    if train_only:
        keep_idx = np.where(is_train)[0]
    else:
        keep_idx = np.arange(len(line_idx_arr))
    print(f"[knn-index] indexing {len(keep_idx)} strokes "
          f"(train_only={train_only}; train pool={int(is_train.sum())}, "
          f"val pool={int((~is_train).sum())})")

    print(f"[knn-index] loading trajectory dataset...")
    ds = TrajectoryDataset(
        meta_csv_path=CSV_PATH,
        normalize_coords=True,
        normalization_mode="canvas",
    )

    embeddings = np.zeros((len(keep_idx), EMBED_DIM), dtype=np.float32)
    trajectories = np.zeros((len(keep_idx), TARGET_TRAJ_LEN, 2), dtype=np.float32)
    out_line_idx = np.zeros(len(keep_idx), dtype=np.int32)
    out_stroke_idx = np.zeros(len(keep_idx), dtype=np.int32)
    diac_flag = np.zeros(len(keep_idx), dtype=np.uint8)

    # Process line-by-line for efficient stroke extraction
    line_to_keep_positions = {}
    for pos, idx in enumerate(keep_idx):
        li = int(line_idx_arr[idx])
        line_to_keep_positions.setdefault(li, []).append((pos, int(stroke_idx_arr[idx])))

    t0 = time.time()
    n_done = 0
    # Buffer patches in batches across lines for speed
    patch_buffer = []
    pos_buffer = []
    for li, items in line_to_keep_positions.items():
        try:
            _, traj_tensor, _, _ = ds[li]
        except Exception:
            continue
        strokes = extract_pen_down_strokes(traj_tensor.numpy())
        for pos, s_idx in items:
            if s_idx >= len(strokes):
                continue
            stroke_xy = strokes[s_idx][:, :2].astype(np.float32)
            patch = render_clean_stroke(
                stroke_xy, crop_size=crop_size, pen_width=pen_width,
                padding_frac=padding_frac,
            )
            patch_buffer.append(patch)
            pos_buffer.append(pos)
            # Save trajectory in normalized [0,1]^2 form (so we can scale
            # into any target bbox at retrieval time)
            normed = normalize_stroke_local(stroke_xy)
            resampled = resample_trajectory(normed, TARGET_TRAJ_LEN)
            trajectories[pos] = resampled
            out_line_idx[pos] = li
            out_stroke_idx[pos] = s_idx

            # Quick DIAC flag using raw extent (matches Phase 5b convention)
            x_range = float(stroke_xy[:, 0].max() - stroke_xy[:, 0].min())
            y_range = float(stroke_xy[:, 1].max() - stroke_xy[:, 1].min())
            if max(x_range, y_range) < 5.0:
                diac_flag[pos] = 1

            if len(patch_buffer) >= batch_size:
                batch = torch.from_numpy(np.stack(patch_buffer)).unsqueeze(1).float()
                emb = extractor(batch).cpu().numpy()
                for i, p in enumerate(pos_buffer):
                    embeddings[p] = emb[i]
                patch_buffer = []
                pos_buffer = []
                n_done += batch_size
                if n_done % 1024 == 0:
                    elapsed = time.time() - t0
                    print(f"  [knn-index] embedded {n_done}/{len(keep_idx)}  "
                          f"({elapsed:.1f}s, {n_done/elapsed:.0f}/s)")

    # Flush remaining
    if patch_buffer:
        batch = torch.from_numpy(np.stack(patch_buffer)).unsqueeze(1).float()
        emb = extractor(batch).cpu().numpy()
        for i, p in enumerate(pos_buffer):
            embeddings[p] = emb[i]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        trajectories=trajectories,
        line_idx=out_line_idx,
        stroke_idx=out_stroke_idx,
        is_diacritic=diac_flag,
    )
    print(f"[knn-index] saved {len(embeddings)} entries to {out_path} "
          f"in {time.time()-t0:.1f}s")


def knn_retrieve(query_embeddings: np.ndarray,
                 index_embeddings: np.ndarray,
                 k: int = 1) -> np.ndarray:
    """L2 nearest-neighbor search (vectorized). Returns (Q, k) indices."""
    # ||q - i||^2 = ||q||^2 + ||i||^2 - 2 q . i
    q_norm = (query_embeddings ** 2).sum(axis=1, keepdims=True)
    i_norm = (index_embeddings ** 2).sum(axis=1, keepdims=True).T
    dot = query_embeddings @ index_embeddings.T
    dists = q_norm + i_norm - 2 * dot
    # k smallest
    if k == 1:
        idx = dists.argmin(axis=1, keepdims=True)
    else:
        idx = np.argpartition(dists, k, axis=1)[:, :k]
        # Sort the top-k by actual distance
        row_dists = np.take_along_axis(dists, idx, axis=1)
        order = np.argsort(row_dists, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
    return idx


def chamfer_rerank(
    query_normalized_trajectories: np.ndarray,
    candidate_indices: np.ndarray,
    index_trajectories: np.ndarray,
) -> np.ndarray:
    """
    Re-rank top-K KNN candidates by Chamfer distance to the query's shape.

    Process per query:
        1. The K embedding-nearest candidate trajectories are passed in
        2. For each candidate, compute Chamfer distance between
           candidate trajectory and the query's normalized trajectory
        3. Return the candidate with smallest Chamfer distance

    This lets the embedding find SHAPE-CATEGORY-similar candidates and
    then geometric distance pick the best-fitting one within that set.

    Args:
        query_normalized_trajectories: (Q, T, 2) — query strokes' normalized trajs
                                        (used as Chamfer target — represents the
                                        actual shape we want to match)
        candidate_indices:              (Q, K) — top-K KNN candidates per query
        index_trajectories:             (N_index, T, 2) — normalized trajectories
                                        stored in KNN index

    Returns:
        best_idx: (Q,) — chosen candidate index per query (from index_trajectories)
    """
    from template_stitcher import chamfer_distance
    Q, K = candidate_indices.shape
    best = np.zeros(Q, dtype=np.int64)
    for q in range(Q):
        query_traj = query_normalized_trajectories[q]    # (T, 2)
        dists = np.zeros(K, dtype=np.float64)
        for ki in range(K):
            cand_idx = int(candidate_indices[q, ki])
            cand_traj = index_trajectories[cand_idx]       # (T, 2)
            dists[ki] = chamfer_distance(query_traj, cand_traj)
        best[q] = int(candidate_indices[q, int(np.argmin(dists))])
    return best


def make_3panel_figure(out_path, image, gt_strokes_canvas,
                       knn_retrieved_canvas, line_idx, n_diac):
    H, W = image.shape
    n_strokes = len(gt_strokes_canvas)
    cmap = plt.get_cmap("turbo", max(n_strokes, 1))

    fig, axes = plt.subplots(3, 1, figsize=(15, 7.0), sharex=True)

    ax = axes[0]
    ax.imshow(image, cmap="gray_r", aspect="auto", extent=[0, W, H, 0])
    ax.set_title(
        f"line {line_idx}: original image ({n_strokes} GT strokes, "
        f"{n_diac} DIAC)",
        fontsize=10,
    )
    ax.set_ylabel("y (px)")

    ax = axes[1]
    ax.set_facecolor("white")
    for i, s in enumerate(gt_strokes_canvas):
        if s.size == 0:
            continue
        ax.plot(s[:, 0], s[:, 1], "-", color=cmap(i), lw=2.2, alpha=0.95)
    ax.set_title("GROUND TRUTH trajectory", fontsize=10)
    ax.set_ylabel("y (px)")

    ax = axes[2]
    ax.set_facecolor("white")
    for i, s in enumerate(knn_retrieved_canvas):
        if s is None or s.size == 0:
            continue
        ax.plot(s[:, 0], s[:, 1], "-", color=cmap(i), lw=2.2, alpha=0.95)
    ax.set_title(
        "KNN-RETRIEVED reconstruction "
        "(nearest training stroke per query, scaled into GT bbox)",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", action="store_true",
                        help="build the KNN index and exit")
    parser.add_argument("--ckpt", type=str, default=str(CKPT_DEFAULT))
    parser.add_argument("--vocab-dir", type=str, default=str(VOCAB_DIR_DEFAULT))
    parser.add_argument("--index-path", type=str, default=str(INDEX_DEFAULT))
    parser.add_argument("--random-lines", type=int, default=8)
    parser.add_argument("--line", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1,
                        help="K nearest neighbors to retrieve (1 = no rerank)")
    parser.add_argument("--rerank", type=str, default="none",
                        choices=["none", "chamfer"],
                        help="reranking strategy applied to top-K candidates "
                             "(only used if --top-k > 1)")
    args = parser.parse_args()

    index_path = Path(args.index_path)

    if args.build_index:
        build_knn_index(Path(args.ckpt), Path(args.vocab_dir), index_path)
        return

    if not index_path.exists():
        print(f"[knn] index not found at {index_path}; building it now...")
        build_knn_index(Path(args.ckpt), Path(args.vocab_dir), index_path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print(f"[knn] loading index from {index_path}")
    idx_data = np.load(index_path)
    index_embeddings = idx_data["embeddings"]
    index_trajectories = idx_data["trajectories"]
    print(f"[knn] index: {len(index_embeddings)} strokes, "
          f"embed_dim={index_embeddings.shape[1]}")

    vocab_dir = Path(args.vocab_dir)
    vocab = np.load(vocab_dir / "vocab.npz")
    K_total = int(vocab["k_total"]) if "k_total" in vocab.files else int(vocab["k"]) + 1
    diac_id = (
        int(vocab["diacritic_class_id"])
        if "diacritic_class_id" in vocab.files else K_total - 1
    )

    print(f"[knn] loading classifier from {args.ckpt}")
    classifier = StrokeCNNClassifier(
        n_classes=K_total, hidden=256, dropout=0.0,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    classifier.load_state_dict(state)
    extractor = make_embedding_extractor(classifier)

    assigns = np.load(vocab_dir / "label_assignments.npz")
    all_lines = sorted({int(li) for li in assigns["line_idx"]})
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(all_lines))
    n_val = int(len(all_lines) * 0.1)
    val_lines = [all_lines[i] for i in perm[:n_val]]

    if args.line is not None:
        picks = [args.line]
    else:
        rng2 = np.random.RandomState(args.seed)
        picks = rng2.choice(val_lines,
                            size=min(args.random_lines, len(val_lines)),
                            replace=False).tolist()
    print(f"[knn] picks: {picks}")

    ds = TrajectoryDataset(
        meta_csv_path=CSV_PATH,
        normalize_coords=True,
        normalization_mode="canvas",
    )

    for line_idx in picks:
        image_tensor, traj_tensor, _, _ = ds[int(line_idx)]
        image = image_tensor[0].numpy()
        H, W = image.shape

        gt_strokes = extract_pen_down_strokes(traj_tensor.numpy())
        gt_strokes_canvas = [s[:, :2].astype(np.float32) for s in gt_strokes]

        # Build query patches + bboxes per stroke
        patches = []
        bboxes = []
        is_diac = []
        for s_xy in gt_strokes_canvas:
            patch = render_clean_stroke(s_xy)
            patches.append(patch)
            x0 = float(s_xy[:, 0].min()); x1 = float(s_xy[:, 0].max())
            y0 = float(s_xy[:, 1].min()); y1 = float(s_xy[:, 1].max())
            bboxes.append((x0, y0, x1, y1))
            ext = max(x1 - x0, y1 - y0)
            is_diac.append(ext < 5.0)

        batch = torch.from_numpy(np.stack(patches)).unsqueeze(1).float()
        with torch.no_grad():
            query_embeddings = extractor(batch).cpu().numpy()

        # KNN search — top-K candidates per query
        k_val = max(1, args.top_k)
        topk_idx = knn_retrieve(query_embeddings, index_embeddings, k=k_val)  # (Q, K)

        if k_val > 1 and args.rerank == "chamfer":
            # Build query normalized trajectories for Chamfer target.
            # The query trajectory IS the GT stroke trajectory (we have it
            # since this is the GT-bbox demo). At true inference, you'd
            # use the image-derived skeleton as the Chamfer target.
            from knn_retrieval import resample_trajectory, normalize_stroke_local
            query_normed = np.zeros((len(gt_strokes_canvas), TARGET_TRAJ_LEN, 2),
                                     dtype=np.float32)
            for qi, s_xy in enumerate(gt_strokes_canvas):
                normed = normalize_stroke_local(s_xy)
                query_normed[qi] = resample_trajectory(normed, TARGET_TRAJ_LEN)
            nn_idx = chamfer_rerank(query_normed, topk_idx, index_trajectories)
            print(f"  [rerank=chamfer k={k_val}] applied per stroke")
        else:
            nn_idx = topk_idx[:, 0]   # just top-1

        # Build retrieved trajectories scaled into target bboxes
        retrieved = []
        for q_i, bbox in zip(nn_idx, bboxes):
            template = index_trajectories[q_i]   # (50, 2) normalized
            x0, y0, x1, y1 = bbox
            w = max(x1 - x0, 1.0); h = max(y1 - y0, 1.0)
            xy = np.empty_like(template)
            xy[:, 0] = x0 + template[:, 0] * w
            xy[:, 1] = y0 + template[:, 1] * h
            retrieved.append(xy)

        out_path = OUT_DIR / f"knn_line{int(line_idx):05d}.png"
        make_3panel_figure(
            out_path, image, gt_strokes_canvas, retrieved,
            line_idx=int(line_idx), n_diac=int(sum(is_diac)),
        )
        print(f"  line {int(line_idx):5d}: {len(gt_strokes_canvas)} strokes  -> {out_path.name}")

    print()
    print("=" * 72)
    print(f"KNN demos saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
