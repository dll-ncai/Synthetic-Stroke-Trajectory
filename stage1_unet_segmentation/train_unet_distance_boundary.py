"""
Phase 10 - U-Net retrain with DISTANCE-TRANSFORM boundary supervision.

Different from dilated_focal: instead of uniformly widening the thin 3x3 band,
we re-define "boundary" as any ink pixel close (Euclidean distance < D) to ink
from a DIFFERENT stroke instance. This produces a boundary band that extends
THROUGH stroke bodies at junctions — i.e., the cuts can actually disconnect
strokes via 8-connectivity.

No edits to stroke_seg_dataset.py — we recompute the boundary class in memory
from the instance mask each batch.
"""

from __future__ import annotations
import config
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader, Subset

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PARENT_DIR))

from stroke_seg_dataset import (StrokeSegDataset, build_canvas_traj_dataset,
                                  line_level_split_indices)
from unet import StrokeUNet, count_parameters
from train_unet import collate_seg, make_epoch_viz
from train_unet_focal import FocalLoss, per_class_iou


def compute_distance_boundary(instance_mask_np: np.ndarray,
                                  max_dist: float = 4.0) -> np.ndarray:
    """Returns a (H, W) bool array: True where the pixel is an ink pixel
    of some stroke AND is within `max_dist` Euclidean distance of an ink
    pixel from a DIFFERENT stroke.

    instance_mask_np: (H, W) int array, 0=bg, 1..N=stroke IDs
    """
    H, W = instance_mask_np.shape
    boundary = np.zeros((H, W), dtype=bool)
    inst_ids = np.unique(instance_mask_np)
    inst_ids = inst_ids[inst_ids > 0]
    if len(inst_ids) < 2:
        return boundary
    for iid in inst_ids:
        other_mask = (instance_mask_np > 0) & (instance_mask_np != iid)
        if not other_mask.any():
            continue
        # distance from each pixel to nearest "other ink" pixel
        # distance_transform_edt returns distance to nearest True in the input
        # so we invert other_mask: True means NOT other ink; distance = how far to other ink
        dist_to_other = ndimage.distance_transform_edt(~other_mask)
        this_ink = (instance_mask_np == iid)
        near_other = this_ink & (dist_to_other <= max_dist)
        boundary |= near_other
    return boundary


def build_distance_semantic(semantic: torch.Tensor,
                              instance: list,
                              max_dist: float) -> torch.Tensor:
    """Build a new 3-class semantic mask using distance-transform boundaries.
    Returns: (B, H, W) long tensor with values in {0, 1, 2}.
    """
    B = semantic.size(0)
    out = semantic.clone()
    for b in range(B):
        inst_np = instance[b].cpu().numpy().astype(np.int32)
        new_bnd = compute_distance_boundary(inst_np, max_dist=max_dist)
        # Preserve background (=0); ink stays 1; boundary pixels become 2.
        sem_b = out[b]
        ink_mask = sem_b >= 1
        sem_b[ink_mask] = 1
        sem_b[torch.from_numpy(new_bnd).to(sem_b.device)] = 2
        out[b] = sem_b
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default=config.MASTER_RAW_CSV)
    parser.add_argument("--out-dir", type=str,
                        default=str(THIS_DIR / "unet_distance_out"))
    parser.add_argument("--init-from", type=str,
                        default=str(config.UNET_PHASE1_BROAD))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--alpha-bg", type=float, default=0.25)
    parser.add_argument("--alpha-interior", type=float, default=1.0)
    parser.add_argument("--alpha-boundary", type=float, default=5.0)
    parser.add_argument("--max-dist", type=float, default=4.0,
                        help="distance threshold for new boundary class")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
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
    print(f"[dist] device={device}")

    base = build_canvas_traj_dataset(args.csv)
    train_idx, val_idx = line_level_split_indices(
        len(base), val_frac=args.val_frac, seed=args.seed,
    )
    print(f"[dist] split: train={len(train_idx)}  val={len(val_idx)}")

    full_train = StrokeSegDataset(base, augment=True, seed=args.seed)
    full_val = StrokeSegDataset(base, augment=False, seed=args.seed)
    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_val, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                shuffle=True, collate_fn=collate_seg,
                                num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_seg,
                              num_workers=args.num_workers)

    model = StrokeUNet(n_classes=3, base_ch=args.base_ch).to(device)
    print(f"[dist] model params: {count_parameters(model):,}")

    if args.init_from and Path(args.init_from).exists():
        ck = torch.load(args.init_from, map_location=device,
                          weights_only=False)
        model.load_state_dict(ck["state_dict"])
        print(f"[dist] warm-started from {args.init_from}")

    alpha = torch.tensor([args.alpha_bg, args.alpha_interior,
                            args.alpha_boundary],
                           device=device, dtype=torch.float32)
    criterion = FocalLoss(alpha=alpha, gamma=args.gamma)
    print(f"[dist] focal loss: gamma={args.gamma}  "
          f"alpha=[{args.alpha_bg}, {args.alpha_interior}, "
          f"{args.alpha_boundary}]")
    print(f"[dist] distance-based boundary GT: max_dist={args.max_dist} pixels")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr * 0.05,
    )

    best_inst_iou = -1.0
    best_under_seg = 1.0
    history = {"train_loss": [], "val_loss": [],
                "iou_bg": [], "iou_interior": [], "iou_boundary": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss_sum, tr_count = 0.0, 0
        for step, batch in enumerate(train_loader):
            img = batch["image"].to(device)
            sem = batch["semantic"].to(device)
            inst = batch["instances"]  # list of (H, W) tensors
            # Rebuild semantic with distance-based boundary
            sem_new = build_distance_semantic(sem, inst,
                                                  max_dist=args.max_dist)
            logits = model(img)
            loss = criterion(logits, sem_new)
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

        # Val: compare to ORIGINAL thin-band GT so numbers are comparable
        # to baseline boundary IoU
        model.eval()
        val_loss_sum, val_count = 0.0, 0
        cls_iou_sum = np.zeros(3, dtype=np.float64)
        cls_iou_count = 0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                sem = batch["semantic"].to(device)
                logits = model(img)
                loss = criterion(logits, sem)
                val_loss_sum += float(loss.item()) * img.size(0)
                val_count += img.size(0)
                pred = logits.argmax(dim=1)
                for b in range(img.size(0)):
                    ious, _ = per_class_iou(pred[b], sem[b], n_classes=3)
                    cls_iou_sum += ious
                    cls_iou_count += 1
        val_loss = val_loss_sum / max(val_count, 1)
        ious = cls_iou_sum / max(cls_iou_count, 1)
        bg_iou, int_iou, bnd_iou = ious[0], ious[1], ious[2]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["iou_bg"].append(bg_iou)
        history["iou_interior"].append(int_iou)
        history["iou_boundary"].append(bnd_iou)

        epoch_dt = time.time() - t0
        print(f"[ep {epoch:02d}] train={train_loss:.4f}  val={val_loss:.4f}  "
              f"IoU bg={bg_iou:.3f} int={int_iou:.3f} "
              f"**bnd={bnd_iou:.3f}**  "
              f"dt={epoch_dt:.1f}s", flush=True)

        # Save last + best
        torch.save({"state_dict": model.state_dict(),
                    "args": vars(args), "history": history, "epoch": epoch},
                    out_dir / "unet_last.pt")
        # Save best by val_loss (since boundary IoU is the wrong metric here)
        if val_loss < (history["val_loss"][-2] if len(history["val_loss"]) > 1
                          else float("inf")):
            torch.save({"state_dict": model.state_dict(),
                        "args": vars(args), "history": history,
                        "epoch": epoch},
                       out_dir / "unet_best.pt")
            print(f"  [save] new best (by val loss) at ep {epoch}")

        make_epoch_viz(out_dir / "viz" / f"epoch_{epoch:02d}.png",
                       model, full_val, device, n=4)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("focal loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Loss")
    axes[1].plot(history["iou_bg"], label="background")
    axes[1].plot(history["iou_interior"], label="interior")
    axes[1].plot(history["iou_boundary"], label="boundary",
                  color="red", linewidth=2)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("IoU")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f"Per-class IoU (dist-based bnd, d={args.max_dist})")
    fig.tight_layout()
    fig.savefig(out_dir / "history.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print()
    print("=" * 72)
    print("DONE. Evaluate with:")
    print(f"  python3 phase10/eval_unet_metrics.py "
          f"--ckpt {out_dir}/unet_last.pt --n-samples 100")


if __name__ == "__main__":
    main()
