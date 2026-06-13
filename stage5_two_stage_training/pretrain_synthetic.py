"""
Phase 13 - STAGE 1: pretrain recognizer from scratch on synthetic
Path B trajectories (cached by phase13/generate_synthetic.py).

Outputs:
    phase13/recognizer_pretrained.pt   (best by synthetic val CER)
    phase13/pretrain_history.png
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
from torch.utils.data import Dataset, DataLoader

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARENT_DIR / "phase10"))
sys.path.insert(0, str(PARENT_DIR / "phase11"))

from vocab_loader import load_vocab, decode_ids
from recognizer import TrajRecognizer, count_params
from recog_datasets import collate_recog


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


class CachedSyntheticDataset(Dataset):
    def __init__(self, npz_path: str, max_samples: int = 0,
                 max_points_filter: int = 0):
        data = np.load(npz_path, allow_pickle=True)
        tokens = data["tokens"]
        token_lens_all = data["token_lens"]
        text_ids_flat = data["text_ids"]
        text_lens = data["text_lens"]
        sources = data["sources"]
        N_total = tokens.shape[0]

        # Slice text_ids_flat into per-sample lists FIRST (needed for indexing).
        all_text_ids = []
        cursor = 0
        for L in text_lens:
            all_text_ids.append(text_ids_flat[cursor:cursor + int(L)])
            cursor += int(L)

        # Build keep_mask: drop samples whose trajectory length > filter.
        if max_points_filter > 0:
            keep = token_lens_all <= max_points_filter
        else:
            keep = np.ones(N_total, dtype=bool)
        kept_idx = np.where(keep)[0]
        n_filtered = N_total - len(kept_idx)

        # Optional cap on number of kept samples.
        if max_samples > 0 and max_samples < len(kept_idx):
            kept_idx = kept_idx[:max_samples]

        self.tokens = tokens[kept_idx]
        self.token_lens = token_lens_all[kept_idx].copy()
        self.text_ids_list = [all_text_ids[i] for i in kept_idx]
        self.sources = sources[kept_idx]

        N = len(kept_idx)
        actual_max_T = int(self.token_lens.max()) if N > 0 else 0
        print(f"[CachedSyntheticDataset] kept {N}/{N_total} samples  "
              f"(filtered_out_len>{max_points_filter}={n_filtered}, "
              f"actual_max_T={actual_max_T})")

    def __len__(self):
        return int(self.tokens.shape[0])

    def __getitem__(self, k):
        T = int(self.token_lens[k])
        tokens = self.tokens[k, :T]
        ids = self.text_ids_list[k]
        return (
            torch.from_numpy(tokens).float(),
            torch.tensor(ids, dtype=torch.long),
            int(T), int(len(ids)),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-npz", type=str,
                        default=str(THIS_DIR / "synthetic_cache.npz"))
    parser.add_argument("--vocab", type=str,
                        default=str(config.RECOG_VOCAB_JSON))
    parser.add_argument("--out", type=str,
                        default=str(THIS_DIR / "recognizer_pretrained.pt"))
    parser.add_argument("--history-png", type=str,
                        default=str(THIS_DIR / "pretrain_history.png"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", type=str, default=None,
                        help="path to a _last.pt checkpoint to resume from")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="if >0, cap kept samples to N (after length filter)")
    parser.add_argument("--max-points-filter", type=int, default=0,
                        help="if >0, DROP samples whose trajectory > N points "
                             "(does NOT truncate; cleaner alignment for CTC)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[pretrain] device={device}")

    char_to_idx = load_vocab(args.vocab)
    V = len(char_to_idx) + 1
    print(f"[pretrain] vocab size (with blank): {V}")

    full = CachedSyntheticDataset(args.synthetic_npz,
                                    max_samples=args.max_samples,
                                    max_points_filter=args.max_points_filter)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(full))
    n_val = int(len(full) * args.val_frac)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    train_ds = torch.utils.data.Subset(full, train_idx)
    val_ds = torch.utils.data.Subset(full, val_idx)
    print(f"[pretrain] split: train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                shuffle=True, collate_fn=collate_recog,
                                num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_recog,
                              num_workers=0)

    model = TrajRecognizer(vocab_size_with_blank=V, input_dim=3).to(device)
    print(f"[pretrain] model params: {count_params(model):,}")
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    use_cpu_ctc = (device.type == "mps")
    if use_cpu_ctc:
        print("[pretrain] MPS detected -> CPU fallback for CTC loss")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=2,
    )

    best_cer = float("inf")
    history = {"train": [], "val": [], "cer": []}
    start_epoch = 1

    # ---- Resume from checkpoint if requested ----
    if args.resume is not None:
        print(f"[pretrain] resuming from {args.resume}")
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        if "optim_state" in ck:
            optim.load_state_dict(ck["optim_state"])
            print("[pretrain] restored optimizer state")
        if "scheduler_state" in ck:
            sched.load_state_dict(ck["scheduler_state"])
            print("[pretrain] restored scheduler state")
        if "history" in ck:
            history = ck["history"]
        prev_ep = int(ck.get("epoch", len(history.get("cer", []))))
        # best CER tracking: take the lowest from history
        if history.get("cer"):
            best_cer = float(min(history["cer"]))
        if "best_val_cer" in ck:
            best_cer = min(best_cer, float(ck["best_val_cer"]))
        start_epoch = prev_ep + 1
        print(f"[pretrain] previous epoch = {prev_ep}  best CER = {best_cer:.3f}")
        print(f"[pretrain] will train epochs {start_epoch}..{args.epochs}")
        if start_epoch > args.epochs:
            print(f"[pretrain] already at or past target epoch {args.epochs}; "
                  f"nothing to do")
            return

    for ep in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss_sum, tr_count = 0.0, 0
        for step, (x_pad, y_cat, in_lens, tgt_lens) in enumerate(train_loader):
            x_pad = x_pad.to(device); y_cat_d = y_cat.to(device)
            in_lens_d = in_lens.to(device); tgt_lens_d = tgt_lens.to(device)
            logits = model(x_pad, in_lens_d)
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_TBC = log_probs.transpose(0, 1)
            if use_cpu_ctc:
                loss = ctc(log_probs_TBC.float().cpu(), y_cat.cpu(),
                            in_lens.cpu(), tgt_lens.cpu())
            else:
                loss = ctc(log_probs_TBC, y_cat_d, in_lens_d, tgt_lens_d)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tr_loss_sum += float(loss.item()) * x_pad.size(0)
            tr_count += x_pad.size(0)
            if step % 50 == 0:
                print(f"  [pretrain ep {ep:02d}] step {step:04d}/{len(train_loader):04d} "
                      f"loss={float(loss.item()):.4f}  ({time.time()-t0:.1f}s)",
                      flush=True)
        train_loss = tr_loss_sum / max(tr_count, 1)

        model.eval()
        v_loss_sum, v_count, edit_chars, total_chars = 0.0, 0, 0, 0
        samples = []
        with torch.no_grad():
            for x_pad, y_cat, in_lens, tgt_lens in val_loader:
                x_pad = x_pad.to(device); y_cat_d = y_cat.to(device)
                in_lens_d = in_lens.to(device); tgt_lens_d = tgt_lens.to(device)
                logits = model(x_pad, in_lens_d)
                log_probs = F.log_softmax(logits, dim=-1)
                log_probs_TBC = log_probs.transpose(0, 1)
                if use_cpu_ctc:
                    loss = ctc(log_probs_TBC.float().cpu(), y_cat.cpu(),
                                in_lens.cpu(), tgt_lens.cpu())
                else:
                    loss = ctc(log_probs_TBC, y_cat_d, in_lens_d, tgt_lens_d)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    v_loss_sum += float(loss.item()) * x_pad.size(0)
                    v_count += x_pad.size(0)
                decoded = greedy_ctc_decode(log_probs_TBC.cpu(), blank_idx=0)
                tgts = torch.split(y_cat, tgt_lens.tolist())
                for pred_ids, tgt in zip(decoded, tgts):
                    tgt_ids = tgt.tolist()
                    edit_chars += edit_distance(pred_ids, tgt_ids)
                    total_chars += len(tgt_ids)
                    if len(samples) < 2:
                        samples.append((decode_ids(tgt_ids, char_to_idx),
                                          decode_ids(pred_ids, char_to_idx)))
        val_loss = v_loss_sum / max(v_count, 1)
        val_cer = edit_chars / max(total_chars, 1)
        sched.step(val_loss)

        dt = time.time() - t0
        print(f"[pretrain ep {ep:02d}] train={train_loss:.4f}  val={val_loss:.4f}  "
              f"CER={val_cer:.3f}  lr={optim.param_groups[0]['lr']:.2e}  "
              f"dt={dt:.1f}s", flush=True)
        for ref, pred in samples:
            print(f"  REF : {ref}")
            print(f"  PRED: {pred}")
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["cer"].append(val_cer)

        # Always save "last" so finetune script can resume from either
        torch.save({"state_dict": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "scheduler_state": sched.state_dict(),
                    "char_to_idx": char_to_idx,
                    "vocab_size_with_blank": V,
                    "history": history,
                    "epoch": ep,
                    "best_val_cer": best_cer,
                    "args": vars(args)},
                    args.out.replace(".pt", "_last.pt"))
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save({"state_dict": model.state_dict(),
                        "char_to_idx": char_to_idx,
                        "vocab_size_with_blank": V,
                        "history": history,
                        "best_val_cer": best_cer,
                        "epoch": ep,
                        "args": vars(args)},
                       args.out)
            print(f"  [save] new best synthetic CER = {val_cer:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train"], label="train")
    axes[0].plot(history["val"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("PRETRAIN CTC loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["cer"], "g-")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val CER (synthetic)")
    axes[1].set_title("PRETRAIN val CER")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.history_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print()
    print("=" * 72)
    print(f"[pretrain] DONE.  best val CER on synthetic = {best_cer:.3f}")
    print(f"[pretrain] best checkpoint: {args.out}")
    print(f"[pretrain] last checkpoint: {args.out.replace('.pt', '_last.pt')}")
    print()
    print("Next (fine-tune on real online):")
    print(f"  python3 phase13/finetune_real.py --pretrained {args.out}")


if __name__ == "__main__":
    main()
