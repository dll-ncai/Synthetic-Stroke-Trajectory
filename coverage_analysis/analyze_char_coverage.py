"""
Phase 13 - character coverage analysis: train vs test mismatch.

Addresses supervisor feedback: "which characters are in train but not test or
vice versa?"

Splits the OUHD-L (UCOM main_all.csv) using the same seed=0 as fine-tune,
then reports:
  - chars in train missing from val (over-represented in train)
  - chars in val missing from train (recognizer can't possibly learn these)
  - per-char frequency comparison
  - overall coverage stats
"""

from __future__ import annotations
import config
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARENT_DIR / "phase11"))

from vocab_loader import load_vocab, normalize_text


VOCAB_PATH = config.RECOG_VOCAB_JSON


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default=config.MAIN_ALL_CSV)
    parser.add_argument("--vocab", type=str, default=str(VOCAB_PATH))
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str,
                        default=str(THIS_DIR / "char_coverage_report.txt"))
    args = parser.parse_args()

    char_to_idx = load_vocab(args.vocab)
    valid = set(char_to_idx.keys())
    print(f"[cov] vocab: {len(valid)} chars (excluding blank/specials)")

    import pandas as pd
    df = pd.read_csv(args.csv)
    if "line" not in df.columns:
        # try common alternatives
        for cand in ("text", "transcription", "label"):
            if cand in df.columns:
                df["line"] = df[cand]; break
    print(f"[cov] loaded {len(df)} samples from {args.csv}")

    # Same split as fine-tune (seed=0, val_frac=0.1)
    n = len(df)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    n_val = int(n * args.val_frac)
    val_idx = sorted(perm[:n_val].tolist())
    train_idx = sorted(perm[n_val:].tolist())

    train_lines = [str(df.iloc[i].get("line", "")) for i in train_idx]
    val_lines = [str(df.iloc[i].get("line", "")) for i in val_idx]

    def collect_chars(lines):
        counter = Counter()
        total = 0
        for ln in lines:
            ln_norm = normalize_text(ln, valid)
            for ch in ln_norm:
                if ch in valid:
                    counter[ch] += 1
                    total += 1
        return counter, total

    train_cnt, train_total = collect_chars(train_lines)
    val_cnt, val_total = collect_chars(val_lines)
    train_chars = set(train_cnt.keys())
    val_chars = set(val_cnt.keys())

    print(f"[cov] train: {len(train_idx)} lines, {train_total} chars, "
          f"{len(train_chars)} unique chars")
    print(f"[cov]   val: {len(val_idx)} lines, {val_total} chars, "
          f"{len(val_chars)} unique chars")

    in_train_not_val = train_chars - val_chars
    in_val_not_train = val_chars - train_chars
    shared = train_chars & val_chars

    lines = []
    lines.append("=" * 78)
    lines.append("CHARACTER COVERAGE: train vs val (seed=0, val_frac=0.1)")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Vocab size: {len(valid)} chars (+ blank)")
    lines.append(f"Train: {len(train_idx)} lines, {train_total} total chars, "
                  f"{len(train_chars)} unique chars seen")
    lines.append(f"Val:   {len(val_idx)} lines, {val_total} total chars, "
                  f"{len(val_chars)} unique chars seen")
    lines.append("")
    lines.append(f"Shared chars (train AND val): {len(shared)}")
    lines.append(f"In train, NOT in val: {len(in_train_not_val)}")
    lines.append(f"In val, NOT in train: {len(in_val_not_train)}  "
                  "<-- recognizer can't learn these")
    lines.append("")

    if in_val_not_train:
        lines.append("CHARS IN VAL BUT NOT TRAIN  (these cause unavoidable errors)")
        lines.append("-" * 78)
        for ch in sorted(in_val_not_train, key=lambda c: -val_cnt[c]):
            lines.append(f"  '{ch}'  val_count={val_cnt[ch]}")
        lines.append("")
    else:
        lines.append("[OK] no chars unique to val")
        lines.append("")

    if in_train_not_val:
        lines.append("CHARS IN TRAIN BUT NOT VAL  (over-represented in train)")
        lines.append("-" * 78)
        top10 = sorted(in_train_not_val, key=lambda c: -train_cnt[c])[:10]
        for ch in top10:
            lines.append(f"  '{ch}'  train_count={train_cnt[ch]}")
        if len(in_train_not_val) > 10:
            lines.append(f"  ... and {len(in_train_not_val) - 10} more")
        lines.append("")

    # Frequency comparison for shared chars
    lines.append("PER-CHAR FREQUENCY COMPARISON  (top 30 most common in train)")
    lines.append("-" * 78)
    lines.append(f"  {'char':5s}  {'train%':>8s}  {'val%':>8s}  "
                  f"{'ratio':>8s}  {'train_n':>9s}  {'val_n':>8s}")
    top_train = sorted(train_cnt.keys(), key=lambda c: -train_cnt[c])[:30]
    for ch in top_train:
        tp = train_cnt[ch] / max(train_total, 1) * 100
        vp = val_cnt[ch] / max(val_total, 1) * 100
        ratio = (tp / vp) if vp > 0 else float("inf")
        rstr = "inf" if ratio == float("inf") else f"{ratio:.2f}"
        lines.append(f"  {repr(ch):5s}  {tp:7.3f}  {vp:7.3f}  {rstr:>8s}  "
                      f"{train_cnt[ch]:>9d}  {val_cnt[ch]:>8d}")
    lines.append("")

    # Headline numbers
    coverage = len(shared) / max(len(val_chars), 1) * 100
    lines.append("HEADLINE")
    lines.append("-" * 78)
    lines.append(f"  Val character coverage by train: {coverage:.1f}%")
    lines.append(f"  (= {len(shared)} val-chars also seen in train / "
                  f"{len(val_chars)} val-chars total)")
    if val_total > 0:
        unseen_n = sum(val_cnt[c] for c in in_val_not_train)
        unseen_pct = unseen_n / val_total * 100
        lines.append(f"  Val char-occurrences from unseen chars: "
                      f"{unseen_n} ({unseen_pct:.2f}%)")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines))
    print(f"[cov] report written to {out_path}")
    print()
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    main()
