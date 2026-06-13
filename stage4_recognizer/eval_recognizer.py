"""
Phase 11 - evaluate a trained recognizer on:
    - Online val (within-domain, with GT trajectory)
    - UHWR via Path B (cross-dataset, scanned handwriting)
    - UPTI via Path B (cross-dataset, printed multi-font)

Reports per-dataset CER and WER plus a few decoded samples.
"""

from __future__ import annotations
import config

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARENT_DIR / "phase10"))
sys.path.insert(0, str(THIS_DIR))

from vocab_loader import load_vocab, decode_ids
from recognizer import TrajRecognizer
from recog_datasets import (
    OnlineRecogDataset, UHWRRecogDataset, UPTIRecogDataset,
    PathBExtractor, collate_recog,
)


UNET_CKPT_DEFAULT = config.UNET_DEPLOY
CLASSIFIER_CKPT_DEFAULT = config.SHAPE_CNN
VOCAB_DIR_DEFAULT = config.VOCAB_DIR_K100
KNN_INDEX_DEFAULT = config.KNN_INDEX
ORDER_CKPT_DEFAULT = config.ORDER_PREDICTOR
UHWR_ROOT = config.UHWR_ROOT
UPTI_IMG_ZIP = config.UPTI_IMG_ZIP
UPTI_LBL_ZIP = config.UPTI_LBL_ZIP


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


def eval_loader(model, loader, char_to_idx, device, name: str, n_show: int = 5):
    model.eval()
    edit_chars = 0; total_chars = 0
    edit_words = 0; total_words = 0
    samples = []
    with torch.no_grad():
        for x_pad, y_cat, in_lens, tgt_lens in loader:
            x_pad = x_pad.to(device)
            in_lens_d = in_lens.to(device)
            logits = model(x_pad, in_lens_d)
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_TBC = log_probs.transpose(0, 1)
            decoded = greedy_ctc_decode(log_probs_TBC.cpu(), blank_idx=0)
            targets_split = torch.split(y_cat, tgt_lens.tolist())
            for pred_ids, tgt in zip(decoded, targets_split):
                tgt_ids = tgt.tolist()
                edit_chars += edit_distance(pred_ids, tgt_ids)
                total_chars += len(tgt_ids)
                pred_text = decode_ids(pred_ids, char_to_idx)
                tgt_text = decode_ids(tgt_ids, char_to_idx)
                if len(samples) < n_show:
                    samples.append((tgt_text, pred_text))
                pw = pred_text.split(); tw = tgt_text.split()
                edit_words += edit_distance(pw, tw)
                total_words += len(tw)
    cer = edit_chars / max(total_chars, 1)
    wer = edit_words / max(total_words, 1)
    print(f"\n--- {name} ---")
    print(f"  CER: {cer:.3f}   WER: {wer:.3f}   n_chars_eval: {total_chars}")
    print(f"  sample predictions:")
    for ref, pred in samples:
        print(f"    REF : {ref}")
        print(f"    PRED: {pred}")
        print()
    return cer, wer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=config.RECOG_FINETUNED)
    parser.add_argument("--vocab", type=str,
                        default=str(config.RECOG_VOCAB_JSON))
    parser.add_argument("--csv", type=str,
                        default=config.MAIN_ALL_CSV)
    parser.add_argument("--unet-ckpt", type=str, default=str(UNET_CKPT_DEFAULT))
    parser.add_argument("--classifier-ckpt", type=str,
                        default=str(CLASSIFIER_CKPT_DEFAULT))
    parser.add_argument("--vocab-dir", type=str, default=str(VOCAB_DIR_DEFAULT))
    parser.add_argument("--knn-index", type=str, default=str(KNN_INDEX_DEFAULT))
    parser.add_argument("--order-ckpt", type=str, default=str(ORDER_CKPT_DEFAULT))
    parser.add_argument("--no-learned-order", action="store_true",
                        help="disable learned order; use R-to-L sort only")
    parser.add_argument("--uhwr-root", type=str, default=UHWR_ROOT)
    parser.add_argument("--upti-img-zip", type=str, default=UPTI_IMG_ZIP)
    parser.add_argument("--upti-lbl-zip", type=str, default=UPTI_LBL_ZIP)
    parser.add_argument("--n-uhwr", type=int, default=200)
    parser.add_argument("--n-upti", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "cpu": device = torch.device("cpu")
    elif args.device == "mps": device = torch.device("mps")
    elif args.device == "cuda": device = torch.device("cuda")
    else:
        if torch.backends.mps.is_available(): device = torch.device("mps")
        elif torch.cuda.is_available(): device = torch.device("cuda")
        else: device = torch.device("cpu")
    print(f"[eval] device={device}")

    char_to_idx = load_vocab(args.vocab)
    V = len(char_to_idx) + 1
    model = TrajRecognizer(vocab_size_with_blank=V, input_dim=3).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    if "best_val_cer" in ck:
        print(f"[eval] loaded ckpt epoch={ck.get('epoch', '?')}  "
              f"best_val_cer={ck['best_val_cer']:.3f}")

    # ---- Online val ----
    base_for_split = OnlineRecogDataset(args.csv, char_to_idx=char_to_idx)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(base_for_split))
    n_val = int(len(base_for_split) * args.val_frac)
    val_idx_pos = sorted(perm[:n_val].tolist())
    val_idx = [base_for_split.indices[i] for i in val_idx_pos]
    online_val_ds = OnlineRecogDataset(args.csv, char_to_idx=char_to_idx,
                                         indices=val_idx)
    online_val_loader = DataLoader(online_val_ds, batch_size=args.batch_size,
                                     shuffle=False, collate_fn=collate_recog)
    online_cer, online_wer = eval_loader(model, online_val_loader, char_to_idx,
                                           device, "ONLINE val (within-domain)")

    # ---- Path B extractor (shared) ----
    extractor_kwargs = {}
    if not args.no_learned_order:
        extractor_kwargs.update(dict(
            classifier_ckpt=args.classifier_ckpt,
            vocab_dir=args.vocab_dir,
            order_ckpt=args.order_ckpt,
            knn_index=args.knn_index,
        ))
    extractor = PathBExtractor(args.unet_ckpt, device=device, **extractor_kwargs)

    # ---- UHWR test ----
    uhwr_label = str(Path(args.uhwr_root) / "test.txt")
    uhwr_ds = UHWRRecogDataset(
        label_file=uhwr_label, uhwr_root=args.uhwr_root,
        char_to_idx=char_to_idx, extractor=extractor,
        max_samples=args.n_uhwr,
    )
    uhwr_loader = DataLoader(uhwr_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_recog)
    uhwr_cer, uhwr_wer = eval_loader(model, uhwr_loader, char_to_idx,
                                       device, "UHWR test (Path B cross-dataset)")

    # ---- UPTI test ----
    upti_ds = UPTIRecogDataset(
        img_zip_path=args.upti_img_zip, label_zip_path=args.upti_lbl_zip,
        char_to_idx=char_to_idx, extractor=extractor,
        split="test", resolution="low", max_samples=args.n_upti,
    )
    upti_loader = DataLoader(upti_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_recog)
    upti_cer, upti_wer = eval_loader(model, upti_loader, char_to_idx,
                                       device, "UPTI test (Path B cross-dataset)")

    # Summary
    print()
    print("=" * 72)
    print("RECOGNITION RESULTS SUMMARY")
    print("=" * 72)
    print(f"  Online val (within-domain):  CER {online_cer:.3f}   WER {online_wer:.3f}")
    print(f"  UHWR  test (Path B):         CER {uhwr_cer:.3f}    WER {uhwr_wer:.3f}")
    print(f"  UPTI  test (Path B):         CER {upti_cer:.3f}    WER {upti_wer:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
