"""
Phase 13 - ligature coverage analysis.

Implements the ligature-counting methodology from:
    Naeem, Zia, Awan, Shafait, ul Hasan.
    "Impact of Ligature Coverage on Training Practical Urdu OCR Systems."
    ICDAR 2017.

A ligature = a continuous letter sequence within a word, separated by
non-connector characters. Urdu non-connectors break a word into multiple
ligatures because the pen is implicitly lifted after them.

Non-connectors: ا د ذ ر ز ژ و ڈ ڑ (and isolated forms like ء)

Reports:
  - unique ligatures in train
  - unique ligatures in val
  - ligatures in val NOT in train (= unavoidable recognition errors)
  - val ligature coverage by train
  - frequency distribution
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


# Urdu non-connectors: chars after which the pen lifts within a word.
# These BREAK a word into ligature units (Naeem et al. 2017, Lehal 2012/13).
NON_CONNECTORS = set("ادذرزژوڈڑء")

# Treat whitespace and punctuation as ligature separators too.
WORD_SEPARATORS = set(" \t\n،۔؛؟,.!?:-/()[]")


def extract_ligatures(text: str) -> list:
    """Split a text line into ligatures.
    A ligature ends after a non-connector character OR before a word separator.
    """
    ligs = []
    current = []
    for ch in text:
        if ch in WORD_SEPARATORS:
            if current:
                ligs.append("".join(current)); current = []
            continue
        current.append(ch)
        if ch in NON_CONNECTORS:
            # ligature ends WITH the non-connector
            ligs.append("".join(current)); current = []
    if current:
        ligs.append("".join(current))
    # filter empties
    return [l for l in ligs if l]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default=config.MAIN_ALL_CSV)
    parser.add_argument("--vocab", type=str,
                        default=str(config.RECOG_VOCAB_JSON))
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str,
                        default=str(THIS_DIR / "ligature_coverage_report.txt"))
    args = parser.parse_args()

    char_to_idx = load_vocab(args.vocab)
    valid = set(char_to_idx.keys())

    import pandas as pd
    df = pd.read_csv(args.csv)
    if "line" not in df.columns:
        for cand in ("text", "transcription", "label"):
            if cand in df.columns:
                df["line"] = df[cand]; break
    print(f"[lig] loaded {len(df)} samples from {args.csv}")

    n = len(df)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    n_val = int(n * args.val_frac)
    val_idx = sorted(perm[:n_val].tolist())
    train_idx = sorted(perm[n_val:].tolist())

    def collect(idx_list):
        lig_cnt = Counter()
        n_lig_total = 0
        n_lines = 0
        for i in idx_list:
            text = str(df.iloc[i].get("line", ""))
            text = normalize_text(text, valid)
            ligs = extract_ligatures(text)
            for lig in ligs:
                lig_cnt[lig] += 1
                n_lig_total += 1
            n_lines += 1
        return lig_cnt, n_lig_total, n_lines

    train_cnt, train_total, train_n = collect(train_idx)
    val_cnt, val_total, val_n = collect(val_idx)
    train_set = set(train_cnt.keys())
    val_set = set(val_cnt.keys())
    shared = train_set & val_set
    in_val_not_train = val_set - train_set
    in_train_not_val = train_set - val_set

    coverage_unique = len(shared) / max(len(val_set), 1) * 100
    unseen_occ = sum(val_cnt[l] for l in in_val_not_train)
    occ_coverage = (val_total - unseen_occ) / max(val_total, 1) * 100

    lines = []
    lines.append("=" * 78)
    lines.append("LIGATURE COVERAGE  (Naeem et al. 2017 methodology)")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Definition: a ligature is a continuous letter sequence within a")
    lines.append("word, separated by Urdu non-connectors (ا د ذ ر ز ژ و ڈ ڑ ء)")
    lines.append("or word boundaries.")
    lines.append("")
    lines.append(f"Train:  {train_n} lines, {train_total} total ligatures, "
                  f"{len(train_set)} UNIQUE ligatures")
    lines.append(f"Val:    {val_n} lines, {val_total} total ligatures, "
                  f"{len(val_set)} UNIQUE ligatures")
    lines.append("")
    lines.append("OVERLAP")
    lines.append("-" * 78)
    lines.append(f"  shared (train AND val): {len(shared)}")
    lines.append(f"  in train, NOT in val:   {len(in_train_not_val)}")
    lines.append(f"  in val, NOT in train:   {len(in_val_not_train)}  "
                  "<-- unavoidable errors")
    lines.append("")
    lines.append("COVERAGE METRICS")
    lines.append("-" * 78)
    lines.append(f"  unique-ligature coverage of val by train: {coverage_unique:.1f}%")
    lines.append(f"  occurrence-weighted coverage:             {occ_coverage:.1f}%")
    lines.append(f"  (= {val_total - unseen_occ} / {val_total} val "
                  f"ligature occurrences are seen in train)")
    lines.append("")

    if in_val_not_train:
        lines.append(f"TOP 30 LIGATURES IN VAL NOT IN TRAIN  "
                      f"(total {len(in_val_not_train)})")
        lines.append("-" * 78)
        top = sorted(in_val_not_train, key=lambda l: -val_cnt[l])[:30]
        for lig in top:
            lines.append(f"  '{lig}'  val_count={val_cnt[lig]}")
        if len(in_val_not_train) > 30:
            lines.append(f"  ... and {len(in_val_not_train) - 30} more")
        lines.append("")

    lines.append("TOP 20 MOST COMMON LIGATURES IN TRAIN")
    lines.append("-" * 78)
    top_train = sorted(train_set, key=lambda l: -train_cnt[l])[:20]
    for lig in top_train:
        tp = train_cnt[lig] / max(train_total, 1) * 100
        vp = val_cnt.get(lig, 0) / max(val_total, 1) * 100
        lines.append(f"  '{lig:>8s}'  train_n={train_cnt[lig]:>6d} ({tp:5.2f}%)  "
                      f"val_n={val_cnt.get(lig, 0):>5d} ({vp:5.2f}%)")
    lines.append("")

    # Headline
    lines.append("HEADLINE NUMBERS")
    lines.append("-" * 78)
    lines.append(f"  Unique ligature coverage:         {coverage_unique:.1f}%")
    lines.append(f"  Occurrence-weighted coverage:     {occ_coverage:.1f}%")
    lines.append(f"  Unseen ligatures in val:          {len(in_val_not_train)} "
                  f"({len(in_val_not_train)/max(len(val_set),1)*100:.1f}% of val unique)")
    lines.append(f"  Ligature occurrences from unseen: {unseen_occ} "
                  f"({unseen_occ/max(val_total,1)*100:.2f}% of val total)")
    lines.append("")
    lines.append("CONTEXT (Naeem et al. 2017)")
    lines.append("-" * 78)
    lines.append("  Total estimated ligatures in Urdu Nastaliq: ~26,000")
    lines.append("  UPTI v1 covered:  6,000  (~23%)")
    lines.append("  UPTI 2.0 covered: 18,000 (~70%)")
    lines.append(f"  Our OUHD-L train: {len(train_set)} unique ligatures "
                  f"({len(train_set)/26000*100:.1f}% of estimated Urdu total)")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines))
    print(f"[lig] report written to {out_path}")
    print()
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    main()
