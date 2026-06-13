"""
Phase 7 - local stroke shape classifier, FIXED version.

Differences from train_local_shape.py:

    Fix 1 (val split): split by LINE, not by stroke. The previous split
        had 100% line-level leakage (every val line also had strokes in
        train). Now val lines are completely held out.

    Fix 2 (honest metric): report val_top1_noDIAC = top-1 accuracy on
        NON-diacritic strokes. This strips out the easy 29% diacritic
        bag and shows whether the model actually discriminates shape.
        Previous reporting of val_top1=0.30 looked like learning but
        was only 1.4 pts above the "always predict DIAC" baseline.

    Fix 3 (clean input): use CleanStrokeDataset which renders each
        stroke as a CLEAN patch (no overlapping ink from neighbouring
        strokes). Previous pipeline cropped from the full line image
        which contained other strokes' ink at the bbox boundary.

    Fix 5 (scratch encoder): use StrokeCNNClassifier (scratch CNN from
        phase5b/cnn_classifier.py) instead of the frozen Maqsood OCR
        encoder. Maqsood was empirically shown to produce non-
        discriminative features for single-stroke crops (val_top1_noDIAC
        ~ 0.8% = random chance). Scratch CNN was 10x better at epoch 0
        in earlier experiments.

Usage:
    python3 phase7/train_local_shape_clean.py
    python3 phase7/train_local_shape_clean.py --epochs 12 --batch-size 64
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
sys.path.insert(0, str(THIS_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from cnn_classifier import StrokeCNNClassifier   # phase5b scratch CNN
from clean_stroke_dataset import (
    CleanStrokeDataset, clean_collate, line_level_split,
)


CSV_PATH = config.MAIN_ALL_CSV
DEFAULT_OUT_DIR = THIS_DIR / "local_shape_clean_out"
VOCAB_DIR_DEFAULT = config.VOCAB_DIR_K100


def topk_acc(logits, targets, k):
    topk = logits.topk(k, dim=-1).indices
    correct = (topk == targets.unsqueeze(-1)).any(dim=-1).float()
    return correct.sum().item(), correct.numel()


@torch.no_grad()
def evaluate(model, loader, device, diacritic_class_id):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    sums = {
        "top1_correct": 0, "top1_total": 0,
        "top5_correct": 0, "top5_total": 0,
        "top1_diac_correct": 0, "top1_diac_total": 0,
        "top1_nondiac_correct": 0, "top1_nondiac_total": 0,
        "top5_nondiac_correct": 0, "top5_nondiac_total": 0,
    }
    for patches, labels, _ in loader:
        patches = patches.to(device); labels = labels.to(device)
        logits = model(patches)
        loss = F.cross_entropy(logits, labels)
        total_loss += float(loss.detach())
        n_batches += 1

        t1c, t1n = topk_acc(logits, labels, 1)
        t5c, t5n = topk_acc(logits, labels, 5)
        sums["top1_correct"] += t1c; sums["top1_total"] += t1n
        sums["top5_correct"] += t5c; sums["top5_total"] += t5n

        is_diac = (labels == diacritic_class_id)
        if is_diac.any():
            c, n = topk_acc(logits[is_diac], labels[is_diac], 1)
            sums["top1_diac_correct"] += c; sums["top1_diac_total"] += n
        nond = ~is_diac
        if nond.any():
            c, n = topk_acc(logits[nond], labels[nond], 1)
            sums["top1_nondiac_correct"] += c; sums["top1_nondiac_total"] += n
            c5, n5 = topk_acc(logits[nond], labels[nond], 5)
            sums["top5_nondiac_correct"] += c5; sums["top5_nondiac_total"] += n5

    def acc(num, denom):
        return num / max(denom, 1)
    return {
        "loss": total_loss / max(n_batches, 1),
        "top1": acc(sums["top1_correct"], sums["top1_total"]),
        "top5": acc(sums["top5_correct"], sums["top5_total"]),
        "top1_DIAC": acc(sums["top1_diac_correct"], sums["top1_diac_total"]),
        "top1_noDIAC": acc(sums["top1_nondiac_correct"], sums["top1_nondiac_total"]),
        "top5_noDIAC": acc(sums["top5_nondiac_correct"], sums["top5_nondiac_total"]),
        "n_diac": sums["top1_diac_total"],
        "n_nondiac": sums["top1_nondiac_total"],
    }


def train_one_epoch(model, loader, optimizer, device, grad_clip,
                    global_step, log_every, diacritic_class_id):
    model.train()
    running_loss = 0.0
    running_correct = 0; running_total = 0
    running_nondiac_correct = 0; running_nondiac_total = 0
    n_batches = 0
    for patches, labels, _ in loader:
        patches = patches.to(device); labels = labels.to(device)
        logits = model(patches)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += float(loss.detach())
        c, n = topk_acc(logits.detach(), labels, 1)
        running_correct += c; running_total += n
        is_diac = (labels == diacritic_class_id)
        nond = ~is_diac
        if nond.any():
            c2, n2 = topk_acc(logits.detach()[nond], labels[nond], 1)
            running_nondiac_correct += c2; running_nondiac_total += n2
        n_batches += 1
        global_step += 1
        if global_step % log_every == 0:
            nd_acc = running_nondiac_correct / max(running_nondiac_total, 1)
            print(f"    [step {global_step}] loss={running_loss/n_batches:.4f}  "
                  f"top1={running_correct/max(running_total,1):.3f}  "
                  f"top1_noDIAC={nd_acc:.3f}")

    return {
        "loss": running_loss / max(n_batches, 1),
        "top1": running_correct / max(running_total, 1),
        "top1_noDIAC": running_nondiac_correct / max(running_nondiac_total, 1),
    }, global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="fraction of LINES used for validation")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--crop-w", type=int, default=256)
    parser.add_argument("--crop-h", type=int, default=64)
    parser.add_argument("--pen-width", type=int, default=2)
    parser.add_argument("--padding-frac", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--vocab-dir", type=str, default=str(VOCAB_DIR_DEFAULT))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                        help="output directory for checkpoints and plots")
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    vocab_dir = Path(args.vocab_dir)
    vocab = np.load(vocab_dir / "vocab.npz")
    K_total = int(vocab["k_total"]) if "k_total" in vocab.files else int(vocab["k"]) + 1
    diacritic_class_id = (
        int(vocab["diacritic_class_id"])
        if "diacritic_class_id" in vocab.files else K_total - 1
    )

    print("=" * 72)
    print("Phase 7 - LOCAL SHAPE (clean): scratch CNN + line-split + noDIAC metric")
    print(f"  vocab_dir={vocab_dir}")
    print(f"  K_total={K_total}  diacritic_class_id={diacritic_class_id}")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"  crop={args.crop_h}x{args.crop_w}  pen_width={args.pen_width}px")
    print(f"  out: {OUT_DIR}")
    print("=" * 72)

    full_ds = CleanStrokeDataset(
        meta_csv_path=CSV_PATH,
        vocab_dir=vocab_dir,
        crop_size=(args.crop_h, args.crop_w),
        pen_width=args.pen_width,
        padding_frac=args.padding_frac,
    )
    train_idx, val_idx = line_level_split(full_ds, val_frac=args.val_frac, seed=0)
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    print(f"[data] total_strokes={len(full_ds)}  train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=clean_collate, num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=clean_collate, num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )

    model = StrokeCNNClassifier(
        n_classes=K_total, hidden=args.hidden, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] scratch CNN params={n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    history = {
        "train_loss": [], "train_top1": [], "train_top1_noDIAC": [],
        "val_loss": [], "val_top1": [], "val_top5": [],
        "val_top1_noDIAC": [], "val_top1_DIAC": [], "val_top5_noDIAC": [],
    }
    best_noDIAC = -1.0
    overall_t0 = time.time()
    global_step = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        print(f"\n[epoch {epoch}] training... steps={len(train_loader)}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")
        train_avg, global_step = train_one_epoch(
            model, train_loader, optimizer, device,
            args.grad_clip, global_step, args.log_every, diacritic_class_id,
        )
        scheduler.step()
        val_avg = evaluate(model, val_loader, device, diacritic_class_id)

        history["train_loss"].append(train_avg["loss"])
        history["train_top1"].append(train_avg["top1"])
        history["train_top1_noDIAC"].append(train_avg["top1_noDIAC"])
        history["val_loss"].append(val_avg["loss"])
        history["val_top1"].append(val_avg["top1"])
        history["val_top5"].append(val_avg["top5"])
        history["val_top1_noDIAC"].append(val_avg["top1_noDIAC"])
        history["val_top1_DIAC"].append(val_avg["top1_DIAC"])
        history["val_top5_noDIAC"].append(val_avg["top5_noDIAC"])

        # Save by noDIAC accuracy (the honest metric)
        improved = val_avg["top1_noDIAC"] > best_noDIAC
        marker = " *BEST_noDIAC*" if improved else ""
        if improved:
            best_noDIAC = val_avg["top1_noDIAC"]
            torch.save(model.state_dict(), OUT_DIR / "model_best_noDIAC.pt")
        torch.save(model.state_dict(), OUT_DIR / "model_last.pt")

        dt = time.time() - t0; cum = (time.time() - overall_t0) / 60
        print(
            f"[epoch {epoch}]\n"
            f"  train_loss={train_avg['loss']:.4f}  "
            f"train_top1={train_avg['top1']:.3f}  "
            f"train_top1_noDIAC={train_avg['top1_noDIAC']:.3f}\n"
            f"  val_loss={val_avg['loss']:.4f}  "
            f"val_top1={val_avg['top1']:.3f}  "
            f"val_top5={val_avg['top5']:.3f}\n"
            f"  val_top1_noDIAC={val_avg['top1_noDIAC']:.3f}{marker}  "
            f"(n={val_avg['n_nondiac']})  "
            f"val_top5_noDIAC={val_avg['top5_noDIAC']:.3f}\n"
            f"  val_top1_DIAC={val_avg['top1_DIAC']:.3f}  "
            f"(n={val_avg['n_diac']})\n"
            f"  time={dt/60:.1f}min  cum={cum:.1f}min"
        )

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    xs = list(range(args.epochs))
    axes[0].plot(xs, history["train_loss"], "o-", label="train")
    axes[0].plot(xs, history["val_loss"], "o-", label="val")
    axes[0].set_title("cross-entropy loss"); axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, history["train_top1"], "o-", label="train top-1")
    axes[1].plot(xs, history["val_top1"], "o-", label="val top-1")
    axes[1].plot(xs, history["val_top5"], "o-", label="val top-5")
    # Baseline: always predict DIAC (~29% of dataset). Reference line.
    axes[1].axhline(0.29, color="gray", linestyle="--", linewidth=0.8,
                    label="always-DIAC baseline")
    axes[1].set_title("overall accuracy"); axes[1].set_xlabel("epoch")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3); axes[1].set_ylim(0, 1)

    axes[2].plot(xs, history["train_top1_noDIAC"], "o-", label="train noDIAC top-1")
    axes[2].plot(xs, history["val_top1_noDIAC"], "o-", label="val noDIAC top-1")
    axes[2].plot(xs, history["val_top5_noDIAC"], "o-", label="val noDIAC top-5")
    axes[2].axhline(0.01, color="gray", linestyle="--", linewidth=0.8,
                    label="random baseline (1/100)")
    axes[2].set_title("noDIAC accuracy (the honest metric)")
    axes[2].set_xlabel("epoch")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3); axes[2].set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_curves.png", dpi=120)
    plt.close(fig)

    print()
    print("=" * 72)
    print(f"Best val_top1_noDIAC: {best_noDIAC:.3f}  (random chance over 100 = 0.010)")
    print(f"Total wall-clock: {(time.time()-overall_t0)/60:.1f} min")
    print(f"Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
