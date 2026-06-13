"""
Phase 10 - U-Net retrain with focal loss + DILATED ground-truth boundary.

The 3x3 neighborhood definition of boundary in stroke_seg_dataset.py produces
a thin band that:
  (1) is hard to predict precisely (small misalignment = big IoU hit)
  (2) is too narrow to actually disconnect strokes when cut (8-connectivity)

This script dilates the GT boundary band IN MEMORY by N pixels before
computing loss, without modifying stroke_seg_dataset.py. Larger band:
  - more positive examples for the model to learn from
  - thicker predicted bands at inference -> actually disconnect strokes

Also keeps focal loss for the class imbalance.
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


def dilate_boundary_class(semantic: torch.Tensor,
                            kernel_size: int = 3) -> torch.Tensor:
    """
    Dilate the boundary class (class 2) by `kernel_size` pixels.
    Pixels that become boundary by dilation are RE-LABELED as boundary,
    overwriting interior (class 1) but not background (class 0).

    semantic: (B, H, W) long with values in {0, 1, 2}
    returns:  (B, H, W) long, same shape, with class-2 region widened.
    """
    # Build binary mask of boundary
    bnd = (semantic == 2).float().unsqueeze(1)            # (B, 1, H, W)
    # Build binary mask of any ink (interior or boundary)
    ink = (semantic >= 1).float().unsqueeze(1)
    # Dilate boundary by max-pool with stride 1
    pad = kernel_size // 2
    bnd_dilated = F.max_pool2d(bnd, kernel_size=kernel_size,
                                  stride=1, padding=pad)            # (B, 1, H, W)
    # Where dilated boundary AND ink -> set to class 2
    new_bnd = (bnd_dilated > 0.5) & (ink > 0.5)                     # (B, 1, H, W)
    out = semantic.clone()
    out[new_bnd.squeeze(1)] = 2
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default=config.MASTER_RAW_CSV)
    parser.add_argument("--out-dir", type=str,
                        default=str(THIS_DIR / "unet_focal_dilated_out"))
    parser.add_argument("--init-from", type=str,
                        default=str(config.UNET_PHASE1_BROAD))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--alpha-bg", type=float, default=0.25)
    parser.add_argument("--alpha-interior", type=float, default=1.0)
    parser.add_argument("--alpha-boundary", type=float, default=5.0)
    parser.add_argument("--dilate-kernel", type=int, default=5,
                        help="kernel size for boundary dilation (3, 5, 7, ...)")
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
    print(f"[focal-dilated] device={device}")

    base = build_canvas_traj_dataset(args.csv)
    train_idx, val_idx = line_level_split_indices(
        len(base), val_frac=args.val_frac, seed=args.seed,
    )
    print(f"[focal-dilated] split: train={len(train_idx)}  val={len(val_idx)}")

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
    print(f"[focal-dilated] model params: {count_parameters(model):,}")

    if args.init_from and Path(args.init_from).exists():
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        print(f"[focal-dilated] warm-started from {args.init_from}")

    alpha = torch.tensor([args.alpha_bg, args.alpha_interior,
                            args.alpha_boundary],
                           device=device, dtype=torch.float32)
    criterion = FocalLoss(alpha=alpha, gamma=args.gamma)
    print(f"[focal-dilated] focal loss: gamma={args.gamma}  "
          f"alpha=[{args.alpha_bg}, {args.alpha_interior}, "
          f"{args.alpha_boundary}]")
    print(f"[focal-dilated] DILATING boundary GT with kernel={args.dilate_kernel}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr * 0.05,
    )

    best_boundary_iou = -1.0
    history = {"train_loss": [], "val_loss": [],
                "iou_bg": [], "iou_interior": [], "iou_boundary": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss_sum, tr_count = 0.0, 0
        for step, batch in enumerate(train_loader):
            img = batch["image"].to(device)
            sem = batch["semantic"].to(device)
            sem_dilated = dilate_boundary_class(sem,
                                                   kernel_size=args.dilate_kernel)
            logits = model(img)
            loss = criterion(logits, sem_dilated)
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

        # ---- val: evaluate against ORIGINAL (non-dilated) GT -----
        # We dilate for training (easier learning), but report IoU on the
        # real boundary band so numbers are comparable to baseline.
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

        torch.save({"state_dict": model.state_dict(),
                    "args": vars(args), "history": history, "epoch": epoch},
                    out_dir / "unet_last.pt")
        if bnd_iou > best_boundary_iou:
            best_boundary_iou = bnd_iou
            torch.save({"state_dict": model.state_dict(),
                        "args": vars(args), "history": history,
                        "best_boundary_iou": best_boundary_iou,
                        "epoch": epoch},
                       out_dir / "unet_best.pt")
            print(f"  [save] new best boundary IoU = {bnd_iou:.3f}")

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
    axes[1].set_title(f"Per-class IoU (dilate kernel={args.dilate_kernel})")
    fig.tight_layout()
    fig.savefig(out_dir / "history.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print()
    print("=" * 72)
    print(f"DONE. best boundary IoU: {best_boundary_iou:.3f}")


if __name__ == "__main__":
    main()
