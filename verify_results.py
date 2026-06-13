"""
One-command verification of the headline result.

Loads the shipped fine-tuned recognizer checkpoint and evaluates it on the
held-out OUHD-L test split (240 lines, deterministic seed=0 split), reporting
character error rate (CER). Expected: ~0.213, matching the thesis headline.

Usage:
    source setup_env.sh          # sets PYTHONPATH + STROKE_DATA_ROOT
    python3 verify_results.py    # add --device mps to use Apple GPU

This uses the ground-truth online trajectories for the test lines (the same
evaluation finetune_real_3way.py performs at the end of training). It does NOT
require the U-Net / trajectory-recovery stages, so it runs in seconds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from vocab_loader import load_vocab
from recog_datasets import OnlineRecogDataset, collate_recog
from recognizer import TrajRecognizer
from finetune_real_3way import eval_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=config.MAIN_ALL_CSV)
    ap.add_argument("--splits",
                    default=str(Path("stage5_two_stage_training") / "main_all_splits.json"))
    ap.add_argument("--vocab", default=config.RECOG_VOCAB_JSON)
    ap.add_argument("--ckpt", default=config.RECOG_FINETUNED)
    ap.add_argument("--max-points", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    splits_path = Path(args.splits)
    if not splits_path.exists():
        raise SystemExit(
            f"Splits file not found: {splits_path}\n"
            "Generate it first:\n"
            "    python3 stage5_two_stage_training/make_splits.py")

    device = torch.device(args.device)
    char_to_idx = load_vocab(args.vocab)
    V = len(char_to_idx) + 1
    print(f"[verify] vocab size (with blank) = {V}")

    test_idx = json.load(open(splits_path))["test_indices"]
    test_ds = OnlineRecogDataset(args.csv, char_to_idx=char_to_idx,
                                 indices=test_idx, max_points=args.max_points)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_recog)

    model = TrajRecognizer(vocab_size_with_blank=V, input_dim=3).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    print(f"[verify] loaded checkpoint: {args.ckpt}")

    _, test_cer = eval_loader(model, test_loader, device)
    print(f"\n>>> TEST CER = {test_cer:.4f}   (thesis headline: 0.213)")


if __name__ == "__main__":
    main()
