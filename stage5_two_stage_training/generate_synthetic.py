"""
Phase 13 - generate synthetic trajectory training data via Path B.

For each sample from UHWR + UPTI:
    1. Read image
    2. Run Path B (with finalized settings: thresh_boundary=0.7, min_area=2,
       min_noise_area=2, learned order predictor enabled)
    3. Tokenize -> (T, 3) [x, y, pen_state]
    4. Apply same simplification used during training (arc-length resample +
       Gaussian smooth)
    5. Encode text -> token ids (blank=0, chars 1..V-1)
    6. Cache to npz

Output: phase13/synthetic_cache.npz with:
    - tokens (N, max_T, 3) padded
    - token_lens (N,)
    - text_ids (concatenated)
    - text_lens (N,)
    - sources (list of strings: "uhwr/<id>" or "upti/<id>")

Usage (smoke test sizes by default):
    python3 phase13/generate_synthetic.py
Or larger:
    python3 phase13/generate_synthetic.py --uhwr-n 8500 --upti-n 20000
"""

from __future__ import annotations
import config

import argparse
import sys
import time
import zipfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARENT_DIR / "phase5b"))
sys.path.insert(0, str(PARENT_DIR / "phase7"))
sys.path.insert(0, str(PARENT_DIR / "phase9"))
sys.path.insert(0, str(PARENT_DIR / "phase10"))
sys.path.insert(0, str(PARENT_DIR / "phase11"))

from vocab_loader import load_vocab, encode_text
from traj_tokenizer import (
    drop_near_duplicates, cap_sequence_length, simplify_trajectory,
)
from recog_datasets import (
    PathBExtractor, ONLINE_TARGET_STEP, APPLY_SIMPLIFICATION,
)


UNET_CKPT = config.UNET_DEPLOY
CLASSIFIER_CKPT = config.SHAPE_CNN
VOCAB_DIR = config.VOCAB_DIR_K100
KNN_INDEX = config.KNN_INDEX
ORDER_CKPT = config.ORDER_PREDICTOR
VOCAB_PATH = config.RECOG_VOCAB_JSON

UHWR_ROOT = Path(config.UHWR_ROOT)
UPTI_IMG_ZIP = config.UPTI_IMG_ZIP
UPTI_LBL_ZIP = config.UPTI_LBL_ZIP


def process_image_to_tokens(raw_gray: np.ndarray, extractor: PathBExtractor,
                              max_points: int = 500) -> np.ndarray:
    """Run Path B + same simplification used in training. Mirrors what
    UHWRRecogDataset.__getitem__ does so cached synthetic data has the
    SAME distribution as inference data."""
    tokens = extractor(raw_gray)
    if APPLY_SIMPLIFICATION:
        tokens = simplify_trajectory(tokens,
                                       target_step=ONLINE_TARGET_STEP,
                                       smooth_sigma=0.6)
    tokens = drop_near_duplicates(tokens)
    tokens = cap_sequence_length(tokens, max_T=max_points)
    return tokens


def load_uhwr_samples(label_file: Path, max_n: int,
                       char_to_idx) -> List[tuple]:
    """Returns list of (image_path, text_ids, source_tag) for UHWR train."""
    out = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2: continue
            rel, text = parts
            ids = encode_text(text.strip(), char_to_idx)
            if len(ids) == 0: continue
            out.append((rel.strip(), ids, f"uhwr/{Path(rel).stem}"))
            if len(out) >= max_n:
                break
    return out


def load_upti_samples(img_zip, lbl_zip, split: str, resolution: str,
                      max_n: int, char_to_idx) -> List[tuple]:
    """Returns list of (img_path_in_zip, text_ids, source_tag) for UPTI."""
    label_set = set(lbl_zip.namelist())
    img_files = [
        f for f in img_zip.namelist()
        if f.startswith(f"images/{split}/")
        and f.endswith(".png")
        and f"/{resolution}/" in f
    ]
    # Shuffle deterministically and take first max_n
    rng = np.random.RandomState(0)
    rng.shuffle(img_files)
    out = []
    for ip in img_files:
        sid = Path(ip).stem
        lp = f"groundtruth/{split}/{sid}.txt"
        if lp not in label_set: continue
        text = lbl_zip.read(lp).decode("utf-8", errors="ignore").strip()
        ids = encode_text(text, char_to_idx)
        if len(ids) == 0: continue
        out.append((ip, ids, f"upti/{sid}"))
        if len(out) >= max_n:
            break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uhwr-n", type=int, default=0,
                        help="number of UHWR train samples to use (0=skip)")
    parser.add_argument("--upti-n", type=int, default=50000,
                        help="number of UPTI samples to use (ICDAR scale)")
    parser.add_argument("--upti-resolution", type=str, default="low")
    parser.add_argument("--upti-split", type=str, default="train")
    parser.add_argument("--max-points", type=int, default=1024,
                        help="max trajectory length (ICDAR uses 1024)")
    parser.add_argument("--out", type=str,
                        default=str(THIS_DIR / "synthetic_cache_upti50k_mp1024.npz"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print(f"[gen] target sizes: UHWR={args.uhwr_n}  UPTI={args.upti_n}  "
          f"total={args.uhwr_n + args.upti_n}")

    char_to_idx = load_vocab(str(VOCAB_PATH))
    print(f"[gen] vocab loaded ({len(char_to_idx)} chars + blank=0)")

    device = torch.device(args.device)
    extractor = PathBExtractor(
        unet_ckpt=str(UNET_CKPT), device=device,
        classifier_ckpt=str(CLASSIFIER_CKPT),
        vocab_dir=str(VOCAB_DIR),
        order_ckpt=str(ORDER_CKPT),
        knn_index=str(KNN_INDEX),
    )
    print("[gen] Path B extractor ready (learned order enabled)")

    # Collect (img_source, text_ids, source_tag) pairs
    print("[gen] discovering samples...")
    samples = []
    if args.uhwr_n > 0:
        uhwr_label = UHWR_ROOT / "train.txt"
        uhwr_samples = load_uhwr_samples(uhwr_label, args.uhwr_n, char_to_idx)
        print(f"[gen]   UHWR: {len(uhwr_samples)} samples")
        samples.extend([("uhwr", rel, ids, tag)
                        for rel, ids, tag in uhwr_samples])

    upti_img_zip = upti_lbl_zip = None
    if args.upti_n > 0:
        upti_img_zip = zipfile.ZipFile(UPTI_IMG_ZIP, "r")
        upti_lbl_zip = zipfile.ZipFile(UPTI_LBL_ZIP, "r")
        upti_samples = load_upti_samples(
            upti_img_zip, upti_lbl_zip,
            split=args.upti_split, resolution=args.upti_resolution,
            max_n=args.upti_n, char_to_idx=char_to_idx,
        )
        print(f"[gen]   UPTI: {len(upti_samples)} samples")
        samples.extend([("upti", ip, ids, tag)
                        for ip, ids, tag in upti_samples])

    print(f"[gen] total samples to process: {len(samples)}")
    print(f"[gen] settings: thresh_boundary=0.7, min_area=2, min_noise_area=2, "
          f"simplification={APPLY_SIMPLIFICATION}, target_step={ONLINE_TARGET_STEP}")

    # Process each one
    tokens_list = []
    text_ids_list = []
    sources_list = []
    skipped = 0
    t0 = time.time()
    for i, (src, ref, ids, tag) in enumerate(samples):
        try:
            if src == "uhwr":
                img_path = UHWR_ROOT / ref
                raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            else:  # upti
                data = upti_img_zip.read(ref)
                raw = cv2.imdecode(np.frombuffer(data, np.uint8),
                                    cv2.IMREAD_GRAYSCALE)
            if raw is None:
                skipped += 1; continue
            tokens = process_image_to_tokens(raw, extractor,
                                               max_points=args.max_points)
            if tokens.shape[0] < 5:
                skipped += 1; continue
            tokens_list.append(tokens)
            text_ids_list.append(np.asarray(ids, dtype=np.int64))
            sources_list.append(tag)
        except Exception as e:
            print(f"  [skip] {tag}: {e}")
            skipped += 1; continue

        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (len(samples) - i - 1) / max(rate, 1e-6)
            print(f"  [gen] {i+1}/{len(samples)}  kept={len(tokens_list)} "
                  f"skipped={skipped}  ({rate:.2f}/s, ETA {eta/60:.1f}min)",
                  flush=True)

    print(f"[gen] DONE  kept={len(tokens_list)} skipped={skipped}")

    # Save to padded npz
    N = len(tokens_list)
    max_T = max(t.shape[0] for t in tokens_list) if N > 0 else 1
    tokens_pad = np.zeros((N, max_T, 3), dtype=np.float32)
    token_lens = np.zeros(N, dtype=np.int32)
    for i, t in enumerate(tokens_list):
        tokens_pad[i, :t.shape[0]] = t
        token_lens[i] = t.shape[0]
    text_ids_concat = np.concatenate(text_ids_list).astype(np.int64) \
        if text_ids_list else np.zeros(0, dtype=np.int64)
    text_lens = np.array([len(t) for t in text_ids_list], dtype=np.int32)

    np.savez_compressed(
        args.out,
        tokens=tokens_pad,
        token_lens=token_lens,
        text_ids=text_ids_concat,
        text_lens=text_lens,
        sources=np.asarray(sources_list, dtype=object),
        max_points=np.int32(args.max_points),
    )
    print(f"[gen] saved {N} samples to {args.out}")
    print(f"[gen] padded shape: {tokens_pad.shape}  "
          f"text_ids: {text_ids_concat.shape}")


if __name__ == "__main__":
    main()
