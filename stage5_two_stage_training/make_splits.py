"""
Phase 13 - create deterministic 80/10/10 train/val/test splits for
main_all.csv (OUHD-L online dataset).

Writes splits to a JSON file so train + eval scripts can read the
same indices reproducibly.

Usage:
    python3 phase13/make_splits.py
"""

from __future__ import annotations
import config
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default=config.MAIN_ALL_CSV)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str,
                        default=str(THIS_DIR / "main_all_splits.json"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    n = len(df)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    n_train = n - n_val - n_test

    test_idx = sorted(perm[:n_test].tolist())
    val_idx = sorted(perm[n_test:n_test + n_val].tolist())
    train_idx = sorted(perm[n_test + n_val:].tolist())

    out = {
        "csv": args.csv, "seed": args.seed, "total_samples": int(n),
        "val_frac": args.val_frac, "test_frac": args.test_frac,
        "n_train": len(train_idx), "n_val": len(val_idx),
        "n_test": len(test_idx),
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[splits] wrote {args.out}")
    print(f"[splits] train={len(train_idx)}  val={len(val_idx)}  "
          f"test={len(test_idx)}  total={n}")


if __name__ == "__main__":
    main()
