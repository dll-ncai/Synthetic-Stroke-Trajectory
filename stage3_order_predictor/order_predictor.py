"""
Phase 9 - learned stroke order predictor.

A small MLP that takes per-stroke features (CNN embedding + spatial features)
and predicts a scalar "order score" in [0, 1]:
    0.0 = first stroke drawn in this line
    1.0 = last stroke drawn

At inference, all detected blobs are scored and sorted by predicted score
to produce the temporal stroke order.

Training:
    Features come from GT strokes (clean trajectories rendered + CNN embedded).
    Labels are normalized GT order index (i / (n_strokes - 1)).

Why this is principled, not heuristic:
    The right-to-left heuristic fails on:
      - Diacritic dots that get added AFTER the host letter is drawn
      - Writers that draw all baselines first, then detail strokes
      - Stacked elements at same x-position
    A learned predictor captures these patterns from data.
"""

from __future__ import annotations
import config

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARENT_DIR / "phase5b"))
sys.path.insert(0, str(PARENT_DIR / "phase7"))
sys.path.insert(0, str(THIS_DIR))


EMBED_DIM = 256
SPATIAL_FEATURES = 6   # x_center, y_center, width, height, area, n_strokes_in_line


class OrderPredictor(nn.Module):
    """
    Tiny MLP: embedding + spatial features -> scalar order score [0, 1].
    ~100k params.
    """
    def __init__(self,
                 embed_dim: int = EMBED_DIM,
                 spatial_dim: int = SPATIAL_FEATURES,
                 hidden: int = 128):
        super().__init__()
        in_dim = embed_dim + spatial_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, embeddings: torch.Tensor,
                spatial: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, embed_dim)
        spatial:    (B, spatial_dim)
        returns:    (B,) scores in [0, 1]
        """
        x = torch.cat([embeddings, spatial], dim=-1)
        return self.net(x).squeeze(-1)


def extract_spatial_features(bbox: tuple,
                              n_strokes_in_line: int,
                              canvas_size: Tuple[int, int] = (64, 1600)) -> np.ndarray:
    """
    Build spatial feature vector for one stroke.

    All features are normalized to [0, 1] using canvas size.

    Returns: (SPATIAL_FEATURES,) float32
    """
    H, W = canvas_size
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) * 0.5 / max(W - 1, 1)
    cy = (y0 + y1) * 0.5 / max(H - 1, 1)
    w = (x1 - x0) / max(W - 1, 1)
    h = (y1 - y0) / max(H - 1, 1)
    area = w * h
    # n_strokes_in_line normalized by typical max ~100
    n_norm = min(n_strokes_in_line / 100.0, 1.0)
    return np.array([cx, cy, w, h, area, n_norm], dtype=np.float32)


def build_order_dataset(vocab_dir: Path,
                        knn_index_path: Path,
                        canvas_size: Tuple[int, int] = (64, 1600)
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build training dataset from the KNN index + GT trajectories.

    For each TRAIN-line stroke (those in the KNN index):
        - embedding from KNN index
        - spatial features from stroke trajectory
        - normalized order target = stroke_idx / (n_strokes_in_line - 1)

    Returns:
        embeddings: (N, embed_dim)
        spatial:    (N, spatial_dim)
        targets:    (N,)
        line_idx:   (N,)        for line-level split
    """
    from trajectory_dataset import TrajectoryDataset
    from stroke_sequence_utils import extract_pen_down_strokes

    print(f"[order-dataset] loading KNN index from {knn_index_path}")
    idx_data = np.load(knn_index_path)
    embeddings_all = idx_data["embeddings"]
    line_idx_all = idx_data["line_idx"]
    stroke_idx_all = idx_data["stroke_idx"]
    N = len(embeddings_all)
    print(f"[order-dataset] N={N} train-pool strokes")

    # Get stroke counts per line + bboxes per (line, stroke)
    # We need to iterate the dataset for the bboxes
    print(f"[order-dataset] loading trajectory dataset...")
    ds = TrajectoryDataset(
        meta_csv_path=config.MAIN_ALL_CSV,
        normalize_coords=True,
        normalization_mode="canvas",
    )

    # Per-line: list of (stroke_idx, bbox)
    by_line: dict = {}
    line_counts: dict = {}
    for global_i, (li, si) in enumerate(zip(line_idx_all, stroke_idx_all)):
        by_line.setdefault(int(li), []).append((global_i, int(si)))

    spatial = np.zeros((N, SPATIAL_FEATURES), dtype=np.float32)
    targets = np.zeros(N, dtype=np.float32)
    n_loaded = 0
    for li, items in by_line.items():
        try:
            _, traj_tensor, _, _ = ds[li]
        except Exception:
            continue
        strokes = extract_pen_down_strokes(traj_tensor.numpy())
        n_strokes = len(strokes)
        line_counts[li] = n_strokes
        for global_i, si in items:
            if si >= n_strokes:
                continue
            xy = strokes[si][:, :2]
            bbox = (
                float(xy[:, 0].min()), float(xy[:, 1].min()),
                float(xy[:, 0].max()), float(xy[:, 1].max()),
            )
            spatial[global_i] = extract_spatial_features(
                bbox, n_strokes, canvas_size=canvas_size,
            )
            # target = position in drawing order, normalized to [0, 1]
            targets[global_i] = float(si) / max(n_strokes - 1, 1)
            n_loaded += 1
        if len(line_counts) % 200 == 0:
            print(f"  [order-dataset] processed {len(line_counts)} lines, "
                  f"{n_loaded}/{N} strokes")

    print(f"[order-dataset] dataset built: {n_loaded} strokes across "
          f"{len(line_counts)} lines")
    return embeddings_all.astype(np.float32), spatial, targets, line_idx_all.astype(np.int32)


def train_order_predictor(
    vocab_dir: Path,
    knn_index_path: Path,
    out_path: Path,
    val_frac: float = 0.1,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden: int = 128,
    device: str = "cpu",
):
    """End-to-end training entrypoint. Saves the MLP state_dict to out_path."""
    embeddings, spatial, targets, line_idx = build_order_dataset(
        vocab_dir, knn_index_path,
    )

    # Line-level split
    rng = np.random.RandomState(0)
    unique_lines = np.unique(line_idx)
    perm = rng.permutation(len(unique_lines))
    n_val = int(len(unique_lines) * val_frac)
    val_lines = set(unique_lines[perm[:n_val]].tolist())
    is_val = np.array([li in val_lines for li in line_idx])
    train_mask = ~is_val
    val_mask = is_val
    print(f"[order-train] train={int(train_mask.sum())}  val={int(val_mask.sum())}")

    # To tensors
    emb_t = torch.from_numpy(embeddings).float()
    sp_t = torch.from_numpy(spatial).float()
    tg_t = torch.from_numpy(targets).float()
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    model = OrderPredictor(embed_dim=embeddings.shape[1],
                            spatial_dim=spatial.shape[1],
                            hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"[order-train] model params: {sum(p.numel() for p in model.parameters()):,}")
    best_val_mse = float("inf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        # shuffle
        np.random.shuffle(train_idx)
        running_loss = 0.0; n_batches = 0
        for i in range(0, len(train_idx), batch_size):
            batch = train_idx[i:i + batch_size]
            e = emb_t[batch].to(device)
            s = sp_t[batch].to(device)
            t = tg_t[batch].to(device)
            optimizer.zero_grad()
            pred = model(e, s)
            loss = F.mse_loss(pred, t)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach())
            n_batches += 1
        train_mse = running_loss / max(n_batches, 1)

        # Val
        model.eval()
        with torch.no_grad():
            ve = emb_t[val_idx].to(device)
            vs = sp_t[val_idx].to(device)
            vt = tg_t[val_idx].to(device)
            vp = model(ve, vs)
            val_mse = float(F.mse_loss(vp, vt).detach())
            # Spearman-style: correlation of predicted scores with true normalized order
            corr = float(torch.corrcoef(torch.stack([vp, vt]))[0, 1])

        marker = ""
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), out_path)
            marker = "  *BEST*"
        print(f"  epoch {epoch+1:2d}/{epochs}  train_mse={train_mse:.4f}  "
              f"val_mse={val_mse:.4f}  pearson_r={corr:.3f}{marker}")

    print(f"[order-train] best val_mse={best_val_mse:.4f}  saved to {out_path}")
    return out_path


def load_order_predictor(ckpt_path: Path,
                         embed_dim: int = EMBED_DIM,
                         spatial_dim: int = SPATIAL_FEATURES,
                         hidden: int = 128,
                         device: str = "cpu") -> OrderPredictor:
    model = OrderPredictor(embed_dim=embed_dim, spatial_dim=spatial_dim, hidden=hidden)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-dir", type=str,
                        default=str(config.VOCAB_DIR_K100))
    parser.add_argument("--knn-index", type=str,
                        default=str(config.KNN_INDEX))
    parser.add_argument("--out", type=str,
                        default=str(config.ORDER_PREDICTOR))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_order_predictor(
        vocab_dir=Path(args.vocab_dir),
        knn_index_path=Path(args.knn_index),
        out_path=Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
