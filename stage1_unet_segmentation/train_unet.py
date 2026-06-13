"""
Phase 10 - train U-Net stroke segmenter.

Input:  rendered line image (synthetic offline-style + augmentation)
Output: 3-class semantic mask {background, stroke-interior, stroke-boundary}

Loss: class-weighted cross-entropy. Background pixels are ~85% of total, so
weight them down so the network actually learns the rare classes.

After each epoch:
    - report mean train + val loss
    - save best checkpoint (lowest val loss)
    - save a 4-sample visualization showing image / GT / pred
"""

from __future__ import annotations
import config

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PARENT_DIR))

from stroke_seg_dataset import (StrokeSegDataset, build_canvas_traj_dataset,
                                  line_level_split_indices)
from unet import StrokeUNet, count_parameters
from visualize_sample import colorize_instance, colorize_semantic


def collate_seg(batch):
    images = torch.stack([s["image"] for s in batch])
    semantics = torch.stack([s["semantic"] for s in batch])
    instances = [s["instance_mask"] for s in batch]
    return {"image": images, "semantic": semantics, "instances": instances}


def colorize_pred(pred_argmax: np.ndarray) -> np.ndarray:
    """Same palette as semantic GT visualization."""
    h, w = pred_argmax.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)
    rgb[pred_argmax == 1] = [0.0, 0.0, 0.0]
    rgb[pred_argmax == 2] = [1.0, 0.2, 0.2]
    return rgb


def make_epoch_viz(out_path, model, val_ds, device, n=4):
    model.eval()
    rng = np.random.RandomState(0)
    picks = rng.choice(len(val_ds), size=min(n, len(val_ds)), replace=False)
    rows = 4 * n   # image / GT semantic / pred semantic / pred argmax over instances? (skip last)
    fig, axes = plt.subplots(rows, 1, figsize=(15, 1.4 * rows))
    if rows == 1:
        axes = np.array(axes).reshape(-1)

    with torch.no_grad():
        for i, idx in enumerate(picks):
            s = val_ds[int(idx)]
            img_t = s["image"].unsqueeze(0).to(device)
            logits = model(img_t)[0].cpu().numpy()     # (3, H, W)
            probs = np.exp(logits - logits.max(axis=0, keepdims=True))
            probs = probs / probs.sum(axis=0, keepdims=True)
            argmax = probs.argmax(axis=0)

            img = s["image"].squeeze(0).numpy()
            sem_gt = s["semantic"].numpy()

            ax = axes[i * 4 + 0]
            ax.imshow(img, cmap="gray_r", aspect="auto")
            ax.set_title(f"val[{int(idx)}] input  ({s['n_strokes']} GT strokes)",
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

            ax = axes[i * 4 + 1]
            ax.imshow(colorize_semantic(sem_gt), aspect="auto")
            ax.set_title("GT semantic (black=interior, red=boundary)", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

            ax = axes[i * 4 + 2]
            ax.imshow(colorize_pred(argmax), aspect="auto")
            ax.set_title("U-Net argmax prediction", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

            ax = axes[i * 4 + 3]
            ax.imshow(probs[2], cmap="hot", aspect="auto", vmin=0, vmax=1)
            ax.set_title("U-Net boundary probability (hot = high)", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default=config.MASTER_RAW_CSV)
    parser.add_argument("--out-dir", type=str,
                        default=str(THIS_DIR / "unet_out"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=None,
                        help="(optional) cap training samples for quick smoke test")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "viz").mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[train] device={device}")

    base = build_canvas_traj_dataset(args.csv)
    train_idx, val_idx = line_level_split_indices(
        len(base), val_frac=args.val_frac, seed=args.seed,
    )
    if args.max_train is not None:
        train_idx = train_idx[: args.max_train]
    print(f"[train] split: train={len(train_idx)}  val={len(val_idx)}")

    full_train = StrokeSegDataset(base, augment=True, seed=args.seed)
    full_val = StrokeSegDataset(base, augment=False, seed=args.seed)
    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_val, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_seg, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_seg, num_workers=args.num_workers,
    )

    model = StrokeUNet(n_classes=3, base_ch=args.base_ch).to(device)
    print(f"[train] model params: {count_parameters(model):,}")

    # Class weights: background is huge, boundary is tiny - bias loss toward
    # the rare-but-critical classes.
    class_weights = torch.tensor([0.1, 1.0, 5.0], device=device,
                                  dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr * 0.05,
    )

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        t0 = time.time()
        tr_loss_sum, tr_count = 0.0, 0
        for step, batch in enumerate(train_loader):
            img = batch["image"].to(device)
            sem = batch["semantic"].to(device)
            logits = model(img)
            loss = criterion(logits, sem)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tr_loss_sum += float(loss.item()) * img.size(0)
            tr_count += img.size(0)
            if step % 50 == 0:
                print(f"  [ep {epoch:02d}] step {step:04d}/{len(train_loader):04d} "
                      f"  loss={float(loss.item()):.4f}  "
                      f"(elapsed {time.time()-t0:.1f}s)", flush=True)
        sched.step()
        train_loss = tr_loss_sum / max(tr_count, 1)

        # ---- val ----
        model.eval()
        val_loss_sum, val_count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                sem = batch["semantic"].to(device)
                logits = model(img)
                loss = criterion(logits, sem)
                val_loss_sum += float(loss.item()) * img.size(0)
                val_count += img.size(0)
        val_loss = val_loss_sum / max(val_count, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        epoch_dt = time.time() - t0
        print(f"[ep {epoch:02d}] train={train_loss:.4f}  val={val_loss:.4f}  "
              f"dt={epoch_dt:.1f}s  lr={sched.get_last_lr()[0]:.5f}", flush=True)

        # ---- save ----
        torch.save({"state_dict": model.state_dict(),
                    "args": vars(args),
                    "history": history},
                    out_dir / "unet_last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"state_dict": model.state_dict(),
                        "args": vars(args),
                        "history": history,
                        "best_val": best_val,
                        "epoch": epoch},
                       out_dir / "unet_best.pt")
            print(f"  [save] new best val={val_loss:.4f}")

        make_epoch_viz(out_dir / "viz" / f"epoch_{epoch:02d}.png",
                       model, full_val, device, n=4)

    # final history plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("U-Net stroke segmentation training")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "history.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print()
    print("=" * 72)
    print(f"[train] done. best val={best_val:.4f}")
    print(f"[train] outputs in {out_dir}")


if __name__ == "__main__":
    main()
