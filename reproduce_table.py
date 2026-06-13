"""
Reproduce the main results table on the held-out OUHD-L test split (240 lines).

Each row is evaluated with the trajectory recipe its checkpoint was trained on:
  - real-only baseline  -> RAW trajectories      (simplification OFF)
  - synthetic / two-stage -> SIMPLIFIED trajectories (simplification ON)

The "simplification" step (uniform arc-length resampling + light smoothing)
matches the geometry-only domain of the synthetic pretraining data; the
real-only baseline was trained without it, on raw tablet trajectories.

Usage:
    source setup_env.sh
    python3 stage5_two_stage_training/make_splits.py   # if not already done
    python3 reproduce_table.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
import recog_datasets
from vocab_loader import load_vocab
from recognizer import TrajRecognizer
from finetune_real_3way import eval_loader

# (label, checkpoint, simplify_recipe, reported_test_cer)
ROWS = [
    ("Synthetic pretrain only (zero-shot)", config.RECOG_PRETRAINED, True, 0.783),
    ("Real-only baseline",                  config.RECOG_REALONLY,   False, 0.281),
    ("Synthetic pretrain + fine-tune (ours)", config.RECOG_FINETUNED, True, 0.213),
]


def eval_ckpt(ckpt: str, simplify: bool, csv: str, test_idx, char_to_idx,
              device: torch.device, max_points: int) -> float:
    recog_datasets.APPLY_SIMPLIFICATION = simplify   # per-row recipe
    ds = recog_datasets.OnlineRecogDataset(csv, char_to_idx=char_to_idx,
                                           indices=test_idx, max_points=max_points)
    loader = DataLoader(ds, batch_size=16, shuffle=False,
                        collate_fn=recog_datasets.collate_recog)
    V = len(char_to_idx) + 1
    model = TrajRecognizer(vocab_size_with_blank=V, input_dim=3).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=False)["state_dict"])
    return eval_loader(model, loader, device)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=config.MAIN_ALL_CSV)
    ap.add_argument("--splits",
                    default=str(Path("stage5_two_stage_training") / "main_all_splits.json"))
    ap.add_argument("--vocab", default=config.RECOG_VOCAB_JSON)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-points", type=int, default=2000)
    args = ap.parse_args()

    splits_path = Path(args.splits)
    if not splits_path.exists():
        raise SystemExit("Splits not found. Run: "
                         "python3 stage5_two_stage_training/make_splits.py")

    device = torch.device(args.device)
    char_to_idx = load_vocab(args.vocab)
    test_idx = json.load(open(splits_path))["test_indices"]

    print(f"{'Training Setup':40s} {'recipe':9s} {'reported':9s} {'reproduced':10s}")
    print("-" * 72)
    for label, ckpt, simplify, reported in ROWS:
        cer = eval_ckpt(ckpt, simplify, args.csv, test_idx, char_to_idx,
                        device, args.max_points)
        recipe = "cleaned" if simplify else "raw"
        print(f"{label:40s} {recipe:9s} {reported:<9.3f} {cer:<10.4f}")


if __name__ == "__main__":
    main()
