"""
Phase 13 - replicate ICDAR-paper recipe with Path B trajectories:

    Stage 1: PRETRAIN recognizer from scratch on synthetic Path B trajectories
             (UHWR + UPTI cache from phase13/generate_synthetic.py)
    Stage 2: FINE-TUNE on real online trajectories (the same 2,163 we used
             for the original training)
    Stage 3: Save final checkpoint

Evaluate after with phase11/eval_recognizer.py.

Outputs:
    phase13/recognizer_pretrained.pt   (after stage 1)
    phase13/recognizer_finetuned.pt    (after stage 2 — THE FINAL MODEL)
    phase13/train_history.png
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


# ---------------------------------------------------------------------------
# Cached synthetic dataset (loaded from npz, no Path B at training time)
# ---------------------------------------------------------------------------

class CachedSyntheticDataset(Dataset):
    def __init__(self, npz_path: str, max_points: int = 500):
        data = np.load(npz_path, allow_pickle=True)
        self.tokens = data["tokens"]            # (N, max_T, 3)
        self.token_lens = data["token_lens"]    # (N,)
        text_ids_flat = data["text_ids"]        # concatenated
        text_lens = data["text_lens"]
        # Split concatenated text_ids back into per-sample arrays
        self.text_ids_list = []
        cursor = 0
        for L in text_lens:
            self.text_ids_list.append(text_ids_flat[cursor:cursor + int(L)])
            cursor += int(L)
        self.sources = data["sources"]
        self.max_points = int(data["max_points"]) if "max_points" in data.files else max_points
        print(f"[CachedSyntheticDataset] loaded {self.tokens.shape[0]} samples "
              f"({self.tokens.shape[1]} max T)")

    def __len__(self):
        return int(self.tokens.shape[0])

    def __getitem__(self, k):
        T = int(self.token_lens[k])
        tokens = self.tokens[k, :T]
        ids = self.text_ids_list[k]
        return (
            torch.from_numpy(tokens).float(),
            torch.tensor(ids, dtype=torch.long),
            int(T),
            int(len(ids)),
        )


# ---------------------------------------------------------------------------
# Training loop (shared between pretrain and fine-tune)
# ---------------------------------------------------------------------------

def train_one_stage(model, train_loader, val_loader, ctc, optim, scheduler,
                     epochs: int, device, blank_idx: int, char_to_idx,
                     stage_name: str, use_cpu_ctc_fallback: bool,
                     history: dict):
    best_val_cer = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss_sum, tr_count = 0.0, 0
        for step, (x_pad, y_cat, in_lens, tgt_lens) in enumerate(train_loader):
            x_pad = x_pad.to(device); y_cat_d = y_cat.to(device)
            in_lens_d = in_lens.to(device); tgt_lens_d = tgt_lens.to(device)
            logits = model(x_pad, in_lens_d)
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_TBC = log_probs.transpose(0, 1)
            if use_cpu_ctc_fallback:
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
                print(f"  [{stage_name} ep {ep:02d}] step {step:04d}/{len(train_loader):04d} "
                      f"loss={float(loss.item()):.4f}  "
                      f"({time.time()-t0:.1f}s)", flush=True)
        train_loss = tr_loss_sum / max(tr_count, 1)

        # validation
        model.eval()
        v_loss_sum, v_count = 0.0, 0
        edit_chars = total_chars = 0
        samples = []
        with torch.no_grad():
            for x_pad, y_cat, in_lens, tgt_lens in val_loader:
                x_pad = x_pad.to(device); y_cat_d = y_cat.to(device)
                in_lens_d = in_lens.to(device); tgt_lens_d = tgt_lens.to(device)
                logits = model(x_pad, in_lens_d)
                log_probs = F.log_softmax(logits, dim=-1)
                log_probs_TBC = log_probs.transpose(0, 1)
                if use_cpu_ctc_fallback:
                    loss = ctc(log_probs_TBC.float().cpu(), y_cat.cpu(),
                                in_lens.cpu(), tgt_lens.cpu())
                else:
                    loss = ctc(log_probs_TBC, y_cat_d, in_lens_d, tgt_lens_d)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    v_loss_sum += float(loss.item()) * x_pad.size(0)
                    v_count += x_pad.size(0)
                decoded = greedy_ctc_decode(log_probs_TBC.cpu(),
                                              blank_idx=blank_idx)
                targets_split = torch.split(y_cat, tgt_lens.tolist())
                for pred_ids, tgt in zip(decoded, targets_split):
                    tgt_ids = tgt.tolist()
                    edit_chars += edit_distance(pred_ids, tgt_ids)
                    total_chars += len(tgt_ids)
                    if len(samples) < 2:
                        samples.append((decode_ids(tgt_ids, char_to_idx),
                                          decode_ids(pred_ids, char_to_idx)))
        val_loss = v_loss_sum / max(v_count, 1)
        val_cer = edit_chars / max(total_chars, 1)
        if scheduler is not None:
            scheduler.step(val_loss)

        dt = time.time() - t0
        lr = optim.param_groups[0]["lr"]
        print(f"[{stage_name} ep {ep:02d}] train={train_loss:.4f}  val={val_loss:.4f}  "
              f"CER={val_cer:.3f}  lr={lr:.2e}  dt={dt:.1f}s", flush=True)
        for ref, pred in samples:
            print(f"  REF : {ref}")
            print(f"  PRED: {pred}")
        history[f"{stage_name}_train"].append(train_loss)
        history[f"{stage_name}_val"].append(val_loss)
        history[f"{stage_name}_cer"].append(val_cer)
        if val_cer < best_val_cer:
            best_val_cer = val_cer
    return best_val_cer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-npz", type=str,
                        default=str(THIS_DIR / "synthetic_cache.npz"))
    parser.add_argument("--csv", type=str,
                        default=config.MAIN_ALL_CSV)
    parser.add_argument("--vocab", type=str,
                        default=str(config.RECOG_VOCAB_JSON))
    parser.add_argument("--out-dir", type=str, default=str(THIS_DIR))
    parser.add_argument("--pretrain-epochs", type=int, default=15)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-points", type=int, default=500)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[train] device={device}")

    char_to_idx = load_vocab(args.vocab)
    V = len(char_to_idx) + 1
    print(f"[train] vocab size (with blank): {V}")

    # --------------------------------------------------------------
    # Stage 1: PRETRAIN on synthetic
    # --------------------------------------------------------------
    print()
    print("=" * 72)
    print("STAGE 1: PRETRAIN on synthetic Path B trajectories")
    print("=" * 72)

    syn_full = CachedSyntheticDataset(args.synthetic_npz,
                                        max_points=args.max_points)
    # Split synthetic into train/val
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(syn_full))
    n_val = int(len(syn_full) * args.val_frac)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    syn_train = torch.utils.data.Subset(syn_full, train_idx)
    syn_val = torch.utils.data.Subset(syn_full, val_idx)
    print(f"[train] synthetic split: train={len(syn_train)}  val={len(syn_val)}")
    syn_train_loader = DataLoader(syn_train, batch_size=args.batch_size,
                                    shuffle=True, collate_fn=collate_recog,
                                    num_workers=0)
    syn_val_loader = DataLoader(syn_val, batch_size=args.batch_size,
                                  shuffle=False, collate_fn=collate_recog,
                                  num_workers=0)

    model = TrajRecognizer(vocab_size_with_blank=V, input_dim=3).to(device)
    print(f"[train] model params: {count_params(model):,}")
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    use_cpu_ctc = (device.type == "mps")
    optim_pre = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)
    sched_pre = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim_pre, mode="min", factor=0.5, patience=2,
    )
    history = {
        "pretrain_train": [], "pretrain_val": [], "pretrain_cer": [],
        "finetune_train": [], "finetune_val": [], "finetune_cer": [],
    }
    best_pre_cer = train_one_stage(
        model, syn_train_loader, syn_val_loader,
        ctc, optim_pre, sched_pre, args.pretrain_epochs,
        device, blank_idx=0, char_to_idx=char_to_idx,
        stage_name="pretrain", use_cpu_ctc_fallback=use_cpu_ctc,
        history=history,
    )
    torch.save({"state_dict": model.state_dict(),
                "char_to_idx": char_to_idx,
                "vocab_size_with_blank": V,
                "history": history,
                "best_pretrain_cer": best_pre_cer,
                "args": vars(args)},
                out_dir / "recognizer_pretrained.pt")
    print(f"[train] pretrain done.  best CER on synthetic val = {best_pre_cer:.3f}")
    print(f"[train] checkpoint saved: {out_dir / 'recognizer_pretrained.pt'}")

    # --------------------------------------------------------------
    # Stage 2: FINE-TUNE on real online (using existing online split)
    # --------------------------------------------------------------
    print()
    print("=" * 72)
    print("STAGE 2: FINE-TUNE on real online trajectories")
    print("=" * 72)

    base_ds = OnlineRecogDataset(args.csv, char_to_idx=char_to_idx,
                                   max_points=args.max_points)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(base_ds))
    n_val_real = int(len(base_ds) * args.val_frac)
    val_idx_pos = sorted(perm[:n_val_real].tolist())
    train_idx_pos = sorted(perm[n_val_real:].tolist())
    real_train_idx = [base_ds.indices[i] for i in train_idx_pos]
    real_val_idx = [base_ds.indices[i] for i in val_idx_pos]
    real_train = OnlineRecogDataset(args.csv, char_to_idx=char_to_idx,
                                      indices=real_train_idx,
                                      max_points=args.max_points)
    real_val = OnlineRecogDataset(args.csv, char_to_idx=char_to_idx,
                                    indices=real_val_idx,
                                    max_points=args.max_points)
    print(f"[train] real split: train={len(real_train)}  val={len(real_val)}")
    real_train_loader = DataLoader(real_train, batch_size=args.batch_size,
                                     shuffle=True, collate_fn=collate_recog,
                                     num_workers=0)
    real_val_loader = DataLoader(real_val, batch_size=args.batch_size,
                                   shuffle=False, collate_fn=collate_recog,
                                   num_workers=0)

    # Fresh optimizer at smaller LR for fine-tune
    optim_ft = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
    sched_ft = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim_ft, mode="min", factor=0.5, patience=2,
    )
    best_ft_cer = train_one_stage(
        model, real_train_loader, real_val_loader,
        ctc, optim_ft, sched_ft, args.finetune_epochs,
        device, blank_idx=0, char_to_idx=char_to_idx,
        stage_name="finetune", use_cpu_ctc_fallback=use_cpu_ctc,
        history=history,
    )
    torch.save({"state_dict": model.state_dict(),
                "char_to_idx": char_to_idx,
                "vocab_size_with_blank": V,
                "history": history,
                "best_val_cer": best_ft_cer,
                "args": vars(args)},
                out_dir / "recognizer_finetuned.pt")
    print(f"[train] fine-tune done.  best CER on real val = {best_ft_cer:.3f}")
    print(f"[train] FINAL checkpoint: {out_dir / 'recognizer_finetuned.pt'}")

    # History plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(history["pretrain_train"], label="train")
    axes[0, 0].plot(history["pretrain_val"], label="val")
    axes[0, 0].set_title("PRETRAIN loss"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(history["pretrain_cer"], "g-")
    axes[0, 1].set_title("PRETRAIN val CER"); axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(history["finetune_train"], label="train")
    axes[1, 0].plot(history["finetune_val"], label="val")
    axes[1, 0].set_title("FINE-TUNE loss"); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(history["finetune_cer"], "g-")
    axes[1, 1].set_title("FINE-TUNE val CER"); axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "train_history.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print()
    print("=" * 72)
    print("ALL STAGES DONE")
    print(f"  Stage 1 (synthetic val CER):  {best_pre_cer:.3f}")
    print(f"  Stage 2 (real val CER):       {best_ft_cer:.3f}")
    print("=" * 72)
    print()
    print("Next: evaluate on UHWR + UPTI test sets:")
    print(f"  python3 phase11/eval_recognizer.py "
          f"--ckpt {out_dir / 'recognizer_finetuned.pt'} "
          f"--n-uhwr 200 --n-upti 200")


if __name__ == "__main__":
    main()
