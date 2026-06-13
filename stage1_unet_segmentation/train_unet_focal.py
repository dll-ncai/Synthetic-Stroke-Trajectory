"""
Phase 10 - U-Net retrain with focal loss + boundary IoU tracking.

Changes vs train_unet.py:
  - Focal loss (gamma=2.0) with alpha class weighting
  - Tracks boundary IoU during validation
  - Saves best checkpoint by BOUNDARY IoU (not just loss)
  - Reports per-class metrics each epoch

Addresses supervisor feedback:
  - Class imbalance (boundary = 0.5% of pixels) -> focal loss focuses on hard examples
  - Need a metric for junction quality -> boundary IoU explicitly tracked
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


class FocalLoss(nn.Module):
    """
    Multi-class focal loss with optional alpha class weighting.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy-classified examples (high p_t) so the model
    focuses on hard/rare classes during training.
    """
    def __init__(self, alpha=None, gamma: float = 2.0,
                  reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha  # tensor of shape (C,) or None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        # logits: (B, C, H, W); target: (B, H, W) long
        log_probs = F.log_softmax(logits, dim=1)             # (B, C, H, W)
        probs = log_probs.exp()
        target_onehot = F.one_hot(target,
                                    num_classes=logits.size(1)).permute(0, 3, 1, 2).float()
        pt = (probs * target_onehot).sum(dim=1).clamp(min=1e-9)   # (B, H, W)
        log_pt = (log_probs * target_onehot).sum(dim=1)            # (B, H, W)
        focal_term = (1.0 - pt) ** self.gamma
        loss = -focal_term * log_pt
        if self.alpha is not None:
            alpha_t = self.alpha[target]                            # (B, H, W)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def per_class_iou(pred_argmax: torch.Tensor, target: torch.Tensor,
                   n_classes: int = 3):
    """Returns (iou_per_class, support_per_class) as numpy arrays."""
    ious = np.zeros(n_classes, dtype=np.float64)
    supports = np.zeros(n_classes, dtype=np.int64)
    pred = pred_argmax.cpu().numpy()
    tgt = target.cpu().numpy()
    for c in range(n_classes):
        pc = (pred == c)
        tc = (tgt == c)
        inter = int((pc & tc).sum())
        union = int((pc | tc).sum())
        ious[c] = inter / union if union > 0 else 0.0
        supports[c] = int(tc.sum())
    return ious, supports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default=config.MASTER_RAW_CSV)
    parser.add_argument("--out-dir", type=str,
                        default=str(THIS_DIR / "unet_focal_out"))
    parser.add_argument("--init-from", type=str, default=None,
                        help="warm-start from existing unet_best.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="focal loss gamma")
    parser.add_argument("--alpha-bg", type=float, default=0.25)
    parser.add_argument("--alpha-interior", type=float, default=1.0)
    parser.add_argument("--alpha-boundary", type=float, default=5.0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=None)
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
    print(f"[train-focal] device={device}")

    base = build_canvas_traj_dataset(args.csv)
    train_idx, val_idx = line_level_split_indices(
        len(base), val_frac=args.val_frac, seed=args.seed,
    )
    if args.max_train is not None:
        train_idx = train_idx[: args.max_train]
    print(f"[train-focal] split: train={len(train_idx)}  val={len(val_idx)}")

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
    print(f"[train-focal] model params: {count_parameters(model):,}")

    if args.init_from is not None and Path(args.init_from).exists():
        ck = torch.load(args.init_from, map_location=device)
        model.load_state_dict(ck["state_dict"])
        print(f"[train-focal] warm-started from {args.init_from}")

    alpha = torch.tensor([args.alpha_bg, args.alpha_interior,
                           args.alpha_boundary],
                          device=device, dtype=torch.float32)
    criterion = FocalLoss(alpha=alpha, gamma=args.gamma)
    print(f"[train-focal] focal loss: gamma={args.gamma}  "
          f"alpha=[{args.alpha_bg}, {args.alpha_interior}, "
          f"{args.alpha_boundary}]")

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

        # ---- val: loss + per-class IoU ----
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
                    "args": vars(args), "history": history,
                    "epoch": epoch},
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
    axes[1].set_title("Per-class IoU")
    fig.tight_layout()
    fig.savefig(out_dir / "history.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print()
    print("=" * 72)
    print(f"DONE. best boundary IoU: {best_boundary_iou:.3f}")
    print(f"checkpoint: {out_dir / 'unet_best.pt'}")


if __name__ == "__main__":
    main()
