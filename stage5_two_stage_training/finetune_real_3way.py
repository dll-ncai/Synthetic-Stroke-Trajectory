"""
Phase 13 - STAGE 2 (3-way split): fine-tune recognizer with proper train/val/test.

Reads splits from main_all_splits.json (produced by make_splits.py).
Same model architecture and training recipe as finetune_real.py, but:
  - 80/10/10 train/val/test split (instead of 90/10 train/val)
  - val used for model selection (best checkpoint)
  - TEST set held out and evaluated ONCE at the end
  - Reports train/val/test CER separately
"""

from __future__ import annotations
import config

import argparse
import json
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
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARENT_DIR / "phase10"))
sys.path.insert(0, str(PARENT_DIR / "phase11"))

from vocab_loader import load_vocab, decode_ids
from recognizer import TrajRecognizer, count_params
from recog_datasets import OnlineRecogDataset, collate_recog


def edit_distance(a, b):
    if len(a) == 0: return len(b)
    if len(b) == 0: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def greedy_ctc_decode(log_probs_TBC, blank_idx=0):
    preds = log_probs_TBC.argmax(dim=-1).transpose(0, 1)
    out = []
    for seq in preds:
        prev = None; decoded = []
        for tok in seq.tolist():
            if tok != prev and tok != blank_idx:
                decoded.append(tok)
            prev = tok
        out.append(decoded)
    return out


def eval_loader(model, loader, device):
    """Returns (loss, cer, wer)."""
    model.eval()
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    use_cpu_ctc = (device.type == "mps")
    loss_sum, count = 0.0, 0
    edit_chars, total_chars = 0, 0
    edit_words, total_words = 0, 0
    with torch.no_grad():
        for x_pad, y_cat, in_lens, tgt_lens in loader:
            x_pad = x_pad.to(device)
            in_lens_d = in_lens.to(device)
            logits = model(x_pad, in_lens_d)
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_TBC = log_probs.transpose(0, 1)
            if use_cpu_ctc:
                loss = ctc(log_probs_TBC.cpu(), y_cat.cpu(),
                            in_lens.cpu(), tgt_lens.cpu())
            else:
                loss = ctc(log_probs_TBC, y_cat.to(device),
                            in_lens.to(device), tgt_lens.to(device))
            loss_sum += float(loss.item()) * x_pad.size(0)
            count += x_pad.size(0)
            decoded = greedy_ctc_decode(log_probs_TBC.cpu(), blank_idx=0)
            tgts = torch.split(y_cat, tgt_lens.tolist())
            for pred_ids, tgt in zip(decoded, tgts):
                tgt_ids = tgt.tolist()
                edit_chars += edit_distance(pred_ids, tgt_ids)
                total_chars += len(tgt_ids)
    avg_loss = loss_sum / max(count, 1)
    cer = edit_chars / max(total_chars, 1)
    return avg_loss, cer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default=config.MAIN_ALL_CSV)
    parser.add_argument("--splits", type=str,
                        default=str(THIS_DIR / "main_all_splits.json"))
    parser.add_argument("--vocab", type=str,
                        default=str(config.RECOG_VOCAB_JSON))
    parser.add_argument("--pretrained-ckpt", type=str,
                        default=str(config.RECOG_PRETRAINED))
    parser.add_argument("--out", type=str,
                        default=str(THIS_DIR / "recognizer_finetuned_3way.pt"))
    parser.add_argument("--history-png", type=str,
                        default=str(THIS_DIR / "finetune_3way_history.png"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[finetune-3way] device={device}")
    char_to_idx = load_vocab(args.vocab)
    V = len(char_to_idx) + 1

    # Read splits
    with open(args.splits) as f:
        splits = json.load(f)
    train_idx = splits["train_indices"]
    val_idx = splits["val_indices"]
    test_idx = splits["test_indices"]
    print(f"[finetune-3way] splits: train={len(train_idx)}  "
          f"val={len(val_idx)}  test={len(test_idx)}")

    # Datasets
    train_ds = OnlineRecogDataset(args.csv, char_to_idx=char_to_idx,
                                    indices=train_idx,
                                    max_points=args.max_points)
    val_ds = OnlineRecogDataset(args.csv, char_to_idx=char_to_idx,
                                  indices=val_idx,
                                  max_points=args.max_points)
    test_ds = OnlineRecogDataset(args.csv, char_to_idx=char_to_idx,
                                   indices=test_idx,
                                   max_points=args.max_points)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                shuffle=True, collate_fn=collate_recog)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_recog)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate_recog)

    # Model
    model = TrajRecognizer(vocab_size_with_blank=V, input_dim=3).to(device)
    if args.pretrained_ckpt and Path(args.pretrained_ckpt).exists():
        ck = torch.load(args.pretrained_ckpt, map_location=device,
                         weights_only=False)
        model.load_state_dict(ck["state_dict"])
        print(f"[finetune-3way] loaded pretrained: {args.pretrained_ckpt}")
    print(f"[finetune-3way] model params: {count_params(model):,}")

    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    use_cpu_ctc = (device.type == "mps")
    if use_cpu_ctc:
        print("[finetune-3way] MPS -> CPU fallback for CTC loss")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=2)

    best_val_cer = float("inf")
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_cer": []}

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss_sum, tr_count = 0.0, 0
        for step, (x_pad, y_cat, in_lens, tgt_lens) in enumerate(train_loader):
            x_pad = x_pad.to(device)
            in_lens_d = in_lens.to(device)
            logits = model(x_pad, in_lens_d)
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_TBC = log_probs.transpose(0, 1)
            if use_cpu_ctc:
                loss = ctc(log_probs_TBC.cpu(), y_cat.cpu(),
                            in_lens.cpu(), tgt_lens.cpu())
            else:
                loss = ctc(log_probs_TBC, y_cat.to(device),
                            in_lens.to(device), tgt_lens.to(device))
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tr_loss_sum += float(loss.item()) * x_pad.size(0)
            tr_count += x_pad.size(0)
            if step % 50 == 0:
                print(f"  [finetune-3way ep {ep:02d}] "
                      f"step {step:04d}/{len(train_loader):04d}  "
                      f"loss={float(loss.item()):.4f}  "
                      f"({time.time()-t0:.1f}s)", flush=True)
        train_loss = tr_loss_sum / max(tr_count, 1)

        val_loss, val_cer = eval_loader(model, val_loader, device)
        sched.step(val_loss)
        epoch_dt = time.time() - t0
        print(f"[finetune-3way ep {ep:02d}] train={train_loss:.4f}  "
              f"val={val_loss:.4f}  val_CER={val_cer:.4f}  "
              f"lr={optim.param_groups[0]['lr']:.2e}  dt={epoch_dt:.1f}s",
              flush=True)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_cer"].append(val_cer)

        # Save last
        torch.save({"state_dict": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "scheduler_state": sched.state_dict(),
                    "epoch": ep, "history": history,
                    "char_to_idx": char_to_idx,
                    "vocab_size_with_blank": V, "args": vars(args)},
                    args.out.replace(".pt", "_last.pt"))
        # Save best by val
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            best_epoch = ep
            torch.save({"state_dict": model.state_dict(),
                        "epoch": ep, "best_val_cer": best_val_cer,
                        "history": history,
                        "char_to_idx": char_to_idx,
                        "vocab_size_with_blank": V, "args": vars(args)},
                       args.out)
            print(f"  [save] new best val CER = {val_cer:.4f}")

    # Final test evaluation on best checkpoint
    print()
    print("=" * 72)
    print("LOADING BEST CHECKPOINT FOR TEST EVAL")
    print("=" * 72)
    ck = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    print(f"[finetune-3way] loaded best (epoch {ck['epoch']}, "
          f"val_CER={ck['best_val_cer']:.4f})")
    test_loss, test_cer = eval_loader(model, test_loader, device)

    # History plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("CTC loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["val_cer"], label="val CER")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("CER")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.history_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print()
    print("=" * 72)
    print("FINAL 3-WAY SPLIT RESULTS")
    print("=" * 72)
    print(f"  Best val CER  (epoch {best_epoch}): {best_val_cer:.4f}")
    print(f"  TEST CER      (held out):          {test_cer:.4f}  "
          "<-- headline number")
    print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}  "
          f"Test: {len(test_idx)} samples")
    print()
    print(f"checkpoint: {args.out}")


if __name__ == "__main__":
    main()
