"""
Phase 11 - train BiLSTM-CTC recognizer (matches proven ICDAR-paper setup).

Training data: online stroke trajectories.
Validation: held-out online val.

Outputs:
    out/recognizer_best.pt   (lowest val CER)
    out/recognizer_last.pt
    out/train_history.png
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
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

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


def greedy_ctc_decode(log_probs_TBC: torch.Tensor, blank_idx: int = 0):
    """log_probs_TBC: (T, B, V). Returns list of label-id sequences."""
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


def compute_ctc(ctc, log_probs_TBC, y_cat, input_lengths, target_lengths,
                  use_cpu_fallback: bool):
    if use_cpu_fallback:
        return ctc(log_probs_TBC.float().cpu(), y_cat.cpu(),
                    input_lengths.cpu(), target_lengths.cpu())
    return ctc(log_probs_TBC, y_cat, input_lengths, target_lengths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default=config.MAIN_ALL_CSV)
    parser.add_argument("--vocab", type=str,
                        default=str(config.RECOG_VOCAB_JSON))
    parser.add_argument("--out-dir", type=str,
                        default=str(THIS_DIR / "recognizer_out"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-points", type=int, default=2000)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume from")
    parser.add_argument("--extra-epochs", type=int, default=0,
                        help="Train for THIS many epochs from the resumed "
                             "checkpoint (overrides --epochs)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cpu": device = torch.device("cpu")
    elif args.device == "mps": device = torch.device("mps")
    elif args.device == "cuda": device = torch.device("cuda")
    else:
        if torch.backends.mps.is_available(): device = torch.device("mps")
        elif torch.cuda.is_available(): device = torch.device("cuda")
        else: device = torch.device("cpu")
    print(f"[train] device={device}")

    char_to_idx = load_vocab(args.vocab)
    V = len(char_to_idx) + 1   # +1 for blank at idx 0
    print(f"[train] vocab size (incl blank=0): {V}")

    # ---- Sanity: confirm blank=0 and chars start at 1 ----
    min_id = min(char_to_idx.values())
    max_id = max(char_to_idx.values())
    print(f"[train] label id range: [{min_id}, {max_id}]  (blank reserved at 0)")
    assert min_id == 1, f"chars must start at 1, got {min_id}"
    assert max_id == V - 1, f"max char id should be V-1={V-1}, got {max_id}"

    # Build line-level split
    base_for_split = OnlineRecogDataset(args.csv, char_to_idx=char_to_idx,
                                          max_points=args.max_points)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(base_for_split))
    n_val = int(len(base_for_split) * args.val_frac)
    val_idx_pos = sorted(perm[:n_val].tolist())
    train_idx_pos = sorted(perm[n_val:].tolist())
    # Convert "positions in base_for_split.indices" back to base csv indices
    base_indices = base_for_split.indices
    train_idx = [base_indices[i] for i in train_idx_pos]
    val_idx = [base_indices[i] for i in val_idx_pos]
    print(f"[train] split: train={len(train_idx)}  val={len(val_idx)}")

    train_ds = OnlineRecogDataset(args.csv, char_to_idx=char_to_idx,
                                    indices=train_idx, max_points=args.max_points)
    val_ds = OnlineRecogDataset(args.csv, char_to_idx=char_to_idx,
                                  indices=val_idx, max_points=args.max_points)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                shuffle=True, collate_fn=collate_recog,
                                num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_recog,
                              num_workers=args.num_workers)

    # ---- Diagnostic: sequence length stats + label sanity ----
    print("\n[diag] checking first 3 train batches...")
    input_lens_all = []; target_lens_all = []
    label_min_seen = float("inf"); label_max_seen = -1
    for ib, (x_pad, y_cat, in_lens, tgt_lens) in enumerate(train_loader):
        input_lens_all.extend(in_lens.tolist())
        target_lens_all.extend(tgt_lens.tolist())
        if y_cat.numel() > 0:
            label_min_seen = min(label_min_seen, int(y_cat.min().item()))
            label_max_seen = max(label_max_seen, int(y_cat.max().item()))
        if ib >= 2:
            break
    il = np.array(input_lens_all); tl = np.array(target_lens_all)
    print(f"[diag] input  lengths: mean={il.mean():.1f} median={np.median(il):.0f} "
          f"min={il.min()} max={il.max()}")
    print(f"[diag] target lengths: mean={tl.mean():.1f} median={np.median(tl):.0f} "
          f"min={tl.min()} max={tl.max()}")
    print(f"[diag] T/L ratio:     mean={(il/np.maximum(tl,1)).mean():.2f}  "
          f"(CTC needs T >= L; healthy > 3)")
    print(f"[diag] label id range in data: [{label_min_seen}, {label_max_seen}]  "
          f"(blank=0 should NOT appear; min must be 1)")
    if label_min_seen == 0:
        print("[diag] WARNING: label id 0 appears in targets - conflicts with blank!")
    short_ratio = float((il < tl * 2).mean())
    if short_ratio > 0:
        print(f"[diag] WARNING: {short_ratio*100:.1f}% of samples have T < 2*L  "
              f"(CTC may struggle)")
    print()

    model = TrajRecognizer(vocab_size_with_blank=V, input_dim=3).to(device)
    print(f"[train] model params: {count_params(model):,}")

    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    use_cpu_ctc = (device.type == "mps")
    if use_cpu_ctc:
        print("[train] MPS detected -> CPU fallback for CTC loss op")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=2,
    )

    best_cer = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_cer": [], "val_wer": []}
    start_epoch = 1
    end_epoch = args.epochs

    # ---- Resume ----
    if args.resume is not None:
        print(f"[resume] loading checkpoint: {args.resume}")
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        if "optim_state" in ck:
            optim.load_state_dict(ck["optim_state"])
            print("[resume] restored optimizer state")
        if "scheduler_state" in ck:
            scheduler.load_state_dict(ck["scheduler_state"])
            print("[resume] restored scheduler state")
        if "history" in ck:
            history = ck["history"]
        prev_ep = int(ck.get("epoch", len(history["val_cer"])))
        prev_best = float(ck.get("best_val_cer", min(history["val_cer"]) if history["val_cer"] else float("inf")))
        best_cer = prev_best
        start_epoch = prev_ep + 1
        if args.extra_epochs > 0:
            end_epoch = prev_ep + args.extra_epochs
        else:
            end_epoch = args.epochs
        print(f"[resume] previous epoch = {prev_ep}  best CER = {prev_best:.3f}")
        print(f"[resume] will train epochs {start_epoch}..{end_epoch}")

    for ep in range(start_epoch, end_epoch + 1):
        model.train()
        t0 = time.time()
        tr_loss_sum, tr_count = 0.0, 0
        for step, (x_pad, y_cat, in_lens, tgt_lens) in enumerate(train_loader):
            x_pad = x_pad.to(device); y_cat = y_cat.to(device)
            in_lens = in_lens.to(device); tgt_lens = tgt_lens.to(device)
            logits = model(x_pad, in_lens)            # (B, T, V)
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_TBC = log_probs.transpose(0, 1)
            loss = compute_ctc(ctc, log_probs_TBC, y_cat,
                                in_lens, tgt_lens, use_cpu_ctc)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tr_loss_sum += float(loss.item()) * x_pad.size(0)
            tr_count += x_pad.size(0)
            if step % 50 == 0:
                print(f"  [ep {ep:02d}] step {step:04d}/{len(train_loader):04d} "
                      f"loss={float(loss.item()):.4f}  "
                      f"({time.time()-t0:.1f}s)", flush=True)
        train_loss = tr_loss_sum / max(tr_count, 1)

        # ---- val ----
        model.eval()
        v_loss_sum, v_count = 0.0, 0
        edit_chars = 0; total_chars = 0
        edit_words = 0; total_words = 0
        decoded_samples = []      # save 3 (ref, pred) pairs per epoch
        with torch.no_grad():
            for x_pad, y_cat, in_lens, tgt_lens in val_loader:
                x_pad = x_pad.to(device); y_cat = y_cat.to(device)
                in_lens_d = in_lens.to(device); tgt_lens_d = tgt_lens.to(device)
                logits = model(x_pad, in_lens_d)
                log_probs = F.log_softmax(logits, dim=-1)
                log_probs_TBC = log_probs.transpose(0, 1)
                loss = compute_ctc(ctc, log_probs_TBC, y_cat,
                                    in_lens_d, tgt_lens_d, use_cpu_ctc)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    v_loss_sum += float(loss.item()) * x_pad.size(0)
                    v_count += x_pad.size(0)
                decoded = greedy_ctc_decode(log_probs_TBC.cpu(), blank_idx=0)
                targets_split = torch.split(y_cat, tgt_lens.tolist())
                for pred_ids, tgt in zip(decoded, targets_split):
                    tgt_ids = tgt.tolist()
                    edit_chars += edit_distance(pred_ids, tgt_ids)
                    total_chars += len(tgt_ids)
                    pred_text = decode_ids(pred_ids, char_to_idx)
                    tgt_text = decode_ids(tgt_ids, char_to_idx)
                    if len(decoded_samples) < 3:
                        decoded_samples.append((tgt_text, pred_text))
                    pw = pred_text.split(); tw = tgt_text.split()
                    edit_words += edit_distance(pw, tw)
                    total_words += len(tw)
        val_loss = v_loss_sum / max(v_count, 1)
        val_cer = edit_chars / max(total_chars, 1)
        val_wer = edit_words / max(total_words, 1)
        scheduler.step(val_loss)

        epoch_dt = time.time() - t0
        print(f"[ep {ep:02d}] train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  CER={val_cer:.3f}  WER={val_wer:.3f}  "
              f"lr={optim.param_groups[0]['lr']:.2e}  "
              f"dt={epoch_dt:.1f}s", flush=True)
        # Print decoded samples each epoch for visibility
        for ref, pred in decoded_samples:
            print(f"  REF : {ref}")
            print(f"  PRED: {pred}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_cer"].append(val_cer)
        history["val_wer"].append(val_wer)

        ckpt_state = {
            "state_dict": model.state_dict(),
            "optim_state": optim.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "char_to_idx": char_to_idx,
            "vocab_size_with_blank": V,
            "args": vars(args), "history": history,
            "epoch": ep, "best_val_cer": best_cer,
        }
        torch.save(ckpt_state, out_dir / "recognizer_last.pt")
        if val_cer < best_cer:
            best_cer = val_cer
            ckpt_state["best_val_cer"] = best_cer
            torch.save(ckpt_state, out_dir / "recognizer_best.pt")
            print(f"  [save] new best val CER = {val_cer:.3f}")

    # Plot history
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("CTC loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["val_cer"], "g-", label="CER")
    axes[1].plot(history["val_wer"], "b-", label="WER", alpha=0.6)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("rate")
    axes[1].set_title("Val CER / WER"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "train_history.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print()
    print("=" * 72)
    print(f"[train] done.  best val CER = {best_cer:.3f}")


if __name__ == "__main__":
    main()
