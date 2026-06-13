"""
Phase 10 - retrain U-Net with thinner stroke rasterization (2-4 px instead
of the original 3-7 px) to better match UHWR's actual ink thickness.

Warm-starts from the existing baseline U-Net for fast convergence.
Saves to a new directory so the current production model is untouched.
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
from torch.utils.data import DataLoader, Subset

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PARENT_DIR))

from stroke_seg_dataset import (StrokeSegDataset, build_canvas_traj_dataset,
                                  line_level_split_indices)
from unet import StrokeUNet, count_parameters
from train_unet import collate_seg, make_epoch_viz
from train_unet_focal import per_class_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default=config.MASTER_RAW_CSV)
    parser.add_argument("--out-dir", type=str,
                        default=str(THIS_DIR / "unet_thinner_out"))
    parser.add_argument("--init-from", type=str,
                        default=str(config.UNET_PHASE1_BROAD))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--thickness-min", type=int, default=2)
    parser.add_argument("--thickness-max", type=int, default=4)
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
    print(f"[thinner] device={device}")
    print(f"[thinner] thickness range: {args.thickness_min}-{args.thickness_max}px")

    base = build_canvas_traj_dataset(args.csv)
    train_idx, val_idx = line_level_split_indices(
        len(base), val_frac=args.val_frac, seed=args.seed,
    )
    print(f"[thinner] split: train={len(train_idx)}  val={len(val_idx)}")

    # Use thinner thickness range
    full_train = StrokeSegDataset(
        base, augment=True, seed=args.seed,
        thickness_range=(args.thickness_min, args.thickness_max),
    )
    full_val = StrokeSegDataset(
        base, augment=False, seed=args.seed,
        thickness_range=(args.thickness_min, args.thickness_max),
    )
    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_val, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                shuffle=True, collate_fn=collate_seg,
                                num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_seg,
                              num_workers=args.num_workers)

    model = StrokeUNet(n_classes=3, base_ch=args.base_ch).to(device)
    print(f"[thinner] model params: {count_parameters(model):,}")

    if args.init_from and Path(args.init_from).exists():
        ck = torch.load(args.init_from, map_location=device,
                          weights_only=False)
        model.load_state_dict(ck["state_dict"])
        print(f"[thinner] warm-started from {args.init_from}")

    # Keep original weighted CE (best on downstream)
    class_weights = torch.tensor([0.1, 1.0, 5.0], device=device,
                                    dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr * 0.05,
    )

    best_val = float("inf")
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

        # Val with per-class IoU
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
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"state_dict": model.state_dict(),
                        "args": vars(args), "history": history,
                        "best_val": best_val, "epoch": epoch},
                       out_dir / "unet_best.pt")
            print(f"  [save] new best val loss = {val_loss:.4f}")

        make_epoch_viz(out_dir / "viz" / f"epoch_{epoch:02d}.png",
                       model, full_val, device, n=4)

    print()
    print("=" * 72)
    print(f"DONE. Saved to {out_dir / 'unet_best.pt'}")
    print(f"Compare with original: python3 phase10/eval_unet_compare_all.py")


if __name__ == "__main__":
    main()
