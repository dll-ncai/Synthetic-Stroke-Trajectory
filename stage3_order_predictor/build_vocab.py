"""
Phase 5b - Step 1: build the stroke vocabulary.

Loads all strokes from the OUHD-L dataset (canvas mode, including
1-point diacritics), resamples each to a fixed length via arc-length
interpolation, normalizes to [0, 1]^2, and clusters with K-means to
produce a discrete vocabulary of stroke templates.

Outputs (saved to phase5b/vocab_out/):
    vocab.npz                   centers, fixed_length, K, label_count_per_class
    vocab_grid.png              visualization of all K cluster centers
    cluster_size_histogram.png  how many strokes fall in each cluster
    label_assignments.npz       (line_idx, stroke_idx) -> cluster_label for each stroke

This is foundation infrastructure. After running this, we have:
    - a learned vocabulary of K stroke templates
    - a label per training stroke (cluster ID)
    - a visual sanity check on whether the clusters look meaningful

Usage:
    python3 phase5b/build_vocab.py                      # K=200 default
    python3 phase5b/build_vocab.py --k 500               # try a larger vocab
    python3 phase5b/build_vocab.py --target-length 32    # shorter templates
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.cluster import KMeans

from trajectory_dataset import TrajectoryDataset
from stroke_sequence_utils import extract_pen_down_strokes


CSV_PATH = config.MAIN_ALL_CSV
DEFAULT_OUT_DIR = THIS_DIR / "vocab_out"


def resample_stroke(xy: np.ndarray, target_len: int) -> np.ndarray:
    """
    Arc-length parameterized resampling to a fixed number of points.

    Args:
        xy: (T, 2) float array of (x, y) points in any coord system
        target_len: number of output points

    Returns:
        (target_len, 2) array
    """
    T = xy.shape[0]
    if T == 0:
        return np.zeros((target_len, 2), dtype=np.float32)
    if T == 1:
        # Single-point stroke (dot/diacritic): replicate
        return np.tile(xy.astype(np.float32), (target_len, 1))

    # Cumulative arc length along the stroke
    diffs = np.diff(xy, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_lens = np.concatenate(([0.0], np.cumsum(seg_lens))).astype(np.float64)
    total_len = float(cum_lens[-1])

    if total_len < 1e-9:
        # All points at the same location
        return np.tile(xy[:1].astype(np.float32), (target_len, 1))

    # Uniform sampling along arc length
    targets = np.linspace(0.0, total_len, target_len)
    x_new = np.interp(targets, cum_lens, xy[:, 0])
    y_new = np.interp(targets, cum_lens, xy[:, 1])
    return np.stack([x_new, y_new], axis=1).astype(np.float32)


def normalize_stroke_local(xy: np.ndarray) -> np.ndarray:
    """
    Shared-span normalization: divide BOTH axes by the same span value
    (max of x_range and y_range). This preserves aspect ratio and prevents
    small-extent strokes (diacritics) from being stretched out across
    [0, 1]^2.

    Small-extent strokes are then centered in the [0, 1]^2 box so that a
    dot ends up near (0.5, 0.5) regardless of its raw position.

    A truly single-point stroke ends up at (0.5, 0.5).
    """
    x_min, x_max = float(xy[:, 0].min()), float(xy[:, 0].max())
    y_min, y_max = float(xy[:, 1].min()), float(xy[:, 1].max())
    x_range = x_max - x_min
    y_range = y_max - y_min
    span = max(x_range, y_range, 1e-9)

    out = np.empty_like(xy, dtype=np.float32)
    # Both axes divided by SAME span to preserve aspect ratio
    out[:, 0] = (xy[:, 0] - x_min) / span
    out[:, 1] = (xy[:, 1] - y_min) / span

    # Center small-extent strokes inside [0, 1]^2 so dots end up at (0.5, 0.5)
    extent_x = x_range / span
    extent_y = y_range / span
    out[:, 0] += (1.0 - extent_x) / 2.0
    out[:, 1] += (1.0 - extent_y) / 2.0

    return out


def collect_strokes(
    ds: TrajectoryDataset,
    target_len: int,
    min_stroke_points: int,
    diacritic_max_extent_px: float,
):
    """
    Iterate the dataset, extract all pen-down strokes, resample + normalize.

    Strokes whose RAW bounding-box max-extent is below
    `diacritic_max_extent_px` are routed into a separate "diacritic" pool
    (they are NOT included in the K-means matrix; they all share a single
    "DOT" class). Everything else is normalized + resampled normally.

    Returns:
        strokes_matrix: (N, target_len, 2) float32 -- normal strokes for K-means
        origin_index:   list of (line_idx, stroke_idx) for each normal stroke
        raw_lengths:    list of original lengths of normal strokes
        diacritic_origins: list of (line_idx, stroke_idx) for diacritic strokes
        diacritic_raw_lengths: list of raw lengths of diacritic strokes
    """
    rows = []
    origins = []
    raw_lengths = []
    diacritic_origins = []
    diacritic_raw_lengths = []

    n_lines = len(ds)
    t0 = time.time()
    print(f"[vocab] iterating {n_lines} lines to collect strokes...")
    print(f"[vocab] diacritic threshold: max(x_range, y_range) < {diacritic_max_extent_px}px")
    for line_idx in range(n_lines):
        try:
            _, traj_tensor, _, _ = ds[line_idx]
        except Exception as e:
            print(f"  [warn] line {line_idx}: {e}")
            continue
        strokes = extract_pen_down_strokes(traj_tensor.numpy())
        for s_idx, stroke in enumerate(strokes):
            if stroke.shape[0] < min_stroke_points:
                continue
            xy = stroke[:, :2].astype(np.float32)

            # Check raw extent in canvas pixels
            x_range_raw = float(xy[:, 0].max() - xy[:, 0].min())
            y_range_raw = float(xy[:, 1].max() - xy[:, 1].min())
            max_extent_raw = max(x_range_raw, y_range_raw)

            if max_extent_raw < diacritic_max_extent_px:
                # diacritic: tiny stroke; don't normalize (would stretch it)
                diacritic_origins.append((line_idx, s_idx))
                diacritic_raw_lengths.append(int(stroke.shape[0]))
                continue

            xy_norm = normalize_stroke_local(xy)
            xy_resampled = resample_stroke(xy_norm, target_len)
            rows.append(xy_resampled)
            origins.append((line_idx, s_idx))
            raw_lengths.append(int(stroke.shape[0]))
        if (line_idx + 1) % 500 == 0:
            print(f"  [vocab] processed {line_idx+1}/{n_lines} lines  "
                  f"normal={len(rows)}  diacritic={len(diacritic_origins)}  "
                  f"elapsed={time.time()-t0:.1f}s")
    print(f"[vocab] done collecting. normal_strokes={len(rows)}  "
          f"diacritic_strokes={len(diacritic_origins)}  "
          f"elapsed={time.time()-t0:.1f}s")
    matrix = np.stack(rows, axis=0)
    return matrix, origins, raw_lengths, diacritic_origins, diacritic_raw_lengths


def plot_vocab_grid(centers: np.ndarray, counts: np.ndarray,
                    grid_path: Path, cols: int = 10):
    """Plot all K cluster centers in a grid for visual inspection."""
    K = centers.shape[0]
    rows = int(np.ceil(K / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    if rows == 1:
        axes = axes[None, :]
    for k in range(rows * cols):
        ax = axes[k // cols, k % cols]
        ax.set_xticks([])
        ax.set_yticks([])
        if k < K:
            xy = centers[k]
            ax.plot(xy[:, 0], xy[:, 1], "-", color="tab:blue", lw=1.0)
            ax.plot(xy[:, 0], xy[:, 1], ".", color="tab:blue", ms=1.5)
            ax.scatter([xy[0, 0]], [xy[0, 1]], color="lime", marker="o",
                       s=20, zorder=5, edgecolors="black", linewidths=0.4)
            ax.scatter([xy[-1, 0]], [xy[-1, 1]], color="red", marker="X",
                       s=20, zorder=5, edgecolors="black", linewidths=0.4)
            ax.add_patch(Rectangle((0, 0), 1, 1, fill=False,
                                   edgecolor="black", linewidth=0.4, alpha=0.4))
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(1.05, -0.05)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"#{k}: {int(counts[k])} strokes", fontsize=6)
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_size_hist(counts: np.ndarray, hist_path: Path):
    fig, ax = plt.subplots(figsize=(9, 4))
    sorted_counts = np.sort(counts)[::-1]
    ax.bar(np.arange(len(sorted_counts)), sorted_counts,
           color="tab:blue", edgecolor="none")
    ax.set_title(f"cluster sizes (sorted descending) "
                 f"min={int(sorted_counts.min())} "
                 f"max={int(sorted_counts.max())} "
                 f"median={int(np.median(sorted_counts))}")
    ax.set_xlabel("cluster rank")
    ax.set_ylabel("# strokes in cluster")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(hist_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=200,
                        help="number of K-means vocabulary clusters "
                             "(plus 1 dedicated DIACRITIC class)")
    parser.add_argument("--target-length", type=int, default=50,
                        help="resample every stroke to this many points")
    parser.add_argument("--min-stroke-points", type=int, default=1,
                        help="ignore strokes shorter than this")
    parser.add_argument("--diacritic-max-extent-px", type=float, default=5.0,
                        help="strokes with max(x_range, y_range) below this "
                             "many canvas pixels are pre-classified as "
                             "diacritics (one dedicated class, no K-means)")
    parser.add_argument("--n-init", type=int, default=4,
                        help="K-means n_init (more = better, slower)")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                        help="output directory for vocab.npz, label_assignments.npz, plots")
    args = parser.parse_args()

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.random_state)

    # The full vocabulary is: K normal clusters (indices 0..K-1)
    # plus one diacritic class (index K). Total = K + 1 classes.
    DIACRITIC_CLASS_ID = args.k

    print("=" * 72)
    print(f"Phase 5b - Build stroke vocabulary")
    print(f"  K={args.k} (+ 1 dedicated diacritic class at index {DIACRITIC_CLASS_ID})")
    print(f"  target_length={args.target_length}  "
          f"min_stroke_pts={args.min_stroke_points}")
    print(f"  diacritic threshold: max(x_range, y_range) < "
          f"{args.diacritic_max_extent_px} canvas px")
    print(f"  output: {OUT_DIR}")
    print("=" * 72)

    print("[vocab] loading dataset (canvas mode)...")
    ds = TrajectoryDataset(
        meta_csv_path=CSV_PATH,
        normalize_coords=True,
        normalization_mode="canvas",
    )
    matrix, origins, raw_lengths, diac_origins, diac_raw_lengths = collect_strokes(
        ds,
        args.target_length,
        args.min_stroke_points,
        args.diacritic_max_extent_px,
    )
    N = matrix.shape[0]
    n_diacritic = len(diac_origins)
    n_total = N + n_diacritic
    print(f"[vocab] strokes_matrix shape={matrix.shape}  raw_len stats: "
          f"min={min(raw_lengths) if raw_lengths else 0} "
          f"median={int(np.median(raw_lengths)) if raw_lengths else 0} "
          f"max={max(raw_lengths) if raw_lengths else 0}")
    print(f"[vocab] diacritic strokes routed to class {DIACRITIC_CLASS_ID}: "
          f"{n_diacritic} ({100*n_diacritic/max(n_total,1):.1f}% of total)")

    X = matrix.reshape(N, -1).astype(np.float32)  # (N, target_length * 2)
    print(f"[vocab] running K-means K={args.k} on {N} strokes (dim={X.shape[1]})...")
    t0 = time.time()
    kmeans = KMeans(
        n_clusters=args.k,
        n_init=args.n_init,
        max_iter=args.max_iter,
        random_state=args.random_state,
        verbose=0,
    )
    labels = kmeans.fit_predict(X)
    centroids_flat = kmeans.cluster_centers_       # (K, target_length * 2)
    centroids = centroids_flat.reshape(args.k, args.target_length, 2)
    print(f"[vocab] K-means done in {time.time()-t0:.1f}s  "
          f"inertia={kmeans.inertia_:.2f}")

    counts = np.bincount(labels, minlength=args.k)
    print(f"[vocab] cluster size stats: min={counts.min()} "
          f"max={counts.max()} median={int(np.median(counts))}")
    n_empty = int((counts == 0).sum())
    if n_empty > 0:
        print(f"  [warn] {n_empty} clusters got 0 strokes assigned")
    n_singleton = int((counts == 1).sum())
    if n_singleton > 0:
        print(f"  [warn] {n_singleton} clusters have only 1 stroke")

    # ----------------------------------------------------------------------
    # Find MEDOIDS: for each cluster, pick the REAL stroke closest to the
    # centroid. This gives us templates that are actual training strokes
    # (natural shapes a real writer drew) instead of synthetic averages.
    # ----------------------------------------------------------------------
    print(f"[vocab] computing medoids (real stroke closest to each centroid)...")
    t0 = time.time()
    medoid_indices = np.full(args.k, -1, dtype=np.int64)
    medoid_centroid_distances = np.zeros(args.k, dtype=np.float32)
    for k in range(args.k):
        member_mask = labels == k
        if not member_mask.any():
            continue
        member_local = np.where(member_mask)[0]
        members = X[member_local]                       # (n_k, D)
        dists = np.linalg.norm(members - centroids_flat[k], axis=1)
        best_local = int(np.argmin(dists))
        medoid_indices[k] = member_local[best_local]
        medoid_centroid_distances[k] = float(dists[best_local])

    # Build medoid templates from real strokes; for empty clusters fall back
    # to the (synthetic) centroid so the array is fully populated.
    medoids = np.empty((args.k, args.target_length, 2), dtype=np.float32)
    for k in range(args.k):
        if medoid_indices[k] >= 0:
            medoids[k] = matrix[medoid_indices[k]]
        else:
            medoids[k] = centroids[k]

    valid_mask = medoid_indices >= 0
    if valid_mask.any():
        print(f"[vocab] medoid->centroid distance stats: "
              f"min={medoid_centroid_distances[valid_mask].min():.4f} "
              f"median={float(np.median(medoid_centroid_distances[valid_mask])):.4f} "
              f"max={medoid_centroid_distances[valid_mask].max():.4f} "
              f"(elapsed {time.time()-t0:.1f}s)")

    # ----------------------------------------------------------------------
    # Build the dedicated DIACRITIC class (index K). All diacritics share
    # a single canonical "dot" template at the center of the [0,1]^2 box,
    # since their tiny raw extent makes per-stroke shape meaningless.
    # ----------------------------------------------------------------------
    diacritic_template = np.full(
        (args.target_length, 2), 0.5, dtype=np.float32,
    )
    # Total class count = K + 1 (K K-means clusters + 1 diacritic class)
    K_total = args.k + 1
    medoids_full = np.empty((K_total, args.target_length, 2), dtype=np.float32)
    medoids_full[:args.k] = medoids
    medoids_full[DIACRITIC_CLASS_ID] = diacritic_template

    centroids_full = np.empty((K_total, args.target_length, 2), dtype=np.float32)
    centroids_full[:args.k] = centroids
    centroids_full[DIACRITIC_CLASS_ID] = diacritic_template

    counts_full = np.zeros(K_total, dtype=np.int32)
    counts_full[:args.k] = counts.astype(np.int32)
    counts_full[DIACRITIC_CLASS_ID] = n_diacritic

    print(f"[vocab] augmented vocab to {K_total} classes "
          f"(K={args.k} K-means + 1 diacritic at index {DIACRITIC_CLASS_ID}, "
          f"n={n_diacritic})")

    # ----------------------------------------------------------------------
    # Build combined label assignments: every stroke gets a class id in
    # [0, K]. Normal strokes keep their K-means label; diacritics all get
    # DIACRITIC_CLASS_ID.
    # ----------------------------------------------------------------------
    normal_origins_arr = np.array(origins, dtype=np.int32)              # (N, 2)
    diac_origins_arr = (
        np.array(diac_origins, dtype=np.int32)
        if n_diacritic > 0 else np.zeros((0, 2), dtype=np.int32)
    )
    all_line_idx = np.concatenate(
        [normal_origins_arr[:, 0], diac_origins_arr[:, 0]]
    ).astype(np.int32)
    all_stroke_idx = np.concatenate(
        [normal_origins_arr[:, 1], diac_origins_arr[:, 1]]
    ).astype(np.int32)
    all_labels = np.concatenate(
        [labels.astype(np.int32),
         np.full(n_diacritic, DIACRITIC_CLASS_ID, dtype=np.int32)]
    )
    all_raw_lengths = np.concatenate(
        [np.array(raw_lengths, dtype=np.int32),
         np.array(diac_raw_lengths, dtype=np.int32)]
    )
    # Flag which entries are diacritics (handy for analysis tooling)
    is_diacritic = np.concatenate(
        [np.zeros(N, dtype=np.uint8),
         np.ones(n_diacritic, dtype=np.uint8)]
    )

    # Save augmented vocab (K+1 classes) + augmented label assignments
    np.savez_compressed(
        OUT_DIR / "vocab.npz",
        centers=medoids_full,                         # (K+1, T, 2)
        centroids=centroids_full,                     # (K+1, T, 2)
        target_length=np.int32(args.target_length),
        k=np.int32(args.k),                           # K-means K (excludes diacritic)
        k_total=np.int32(K_total),                    # K + 1
        diacritic_class_id=np.int32(DIACRITIC_CLASS_ID),
        diacritic_max_extent_px=np.float32(args.diacritic_max_extent_px),
        counts=counts_full,
        medoid_indices=medoid_indices.astype(np.int32),
        medoid_centroid_distances=medoid_centroid_distances,
    )
    np.savez_compressed(
        OUT_DIR / "label_assignments.npz",
        labels=all_labels,
        line_idx=all_line_idx,
        stroke_idx=all_stroke_idx,
        raw_lengths=all_raw_lengths,
        is_diacritic=is_diacritic,
        diacritic_class_id=np.int32(DIACRITIC_CLASS_ID),
    )
    print(f"[vocab] saved vocab.npz and label_assignments.npz "
          f"(N_normal={N}  N_diacritic={n_diacritic}  total={N+n_diacritic})")

    # Visualizations (include diacritic class as last cell of the grid)
    print(f"[vocab] generating cluster-center grid (medoids = real strokes, "
          f"+ diacritic class)...")
    plot_vocab_grid(medoids_full, counts_full,
                    OUT_DIR / "vocab_grid_medoids.png", cols=10)
    print(f"[vocab] generating cluster-center grid (synthetic centroids, for comparison)...")
    plot_vocab_grid(centroids_full, counts_full,
                    OUT_DIR / "vocab_grid_centroids.png", cols=10)
    print(f"[vocab] generating cluster-size histogram...")
    plot_cluster_size_hist(counts_full, OUT_DIR / "cluster_size_histogram.png")
    print(f"[vocab] all outputs in {OUT_DIR}")
    print(f"[vocab] open {OUT_DIR / 'vocab_grid_medoids.png'} to see the real-stroke templates")
    print(f"[vocab] diacritic class is the LAST cell (#{DIACRITIC_CLASS_ID})")


if __name__ == "__main__":
    main()
