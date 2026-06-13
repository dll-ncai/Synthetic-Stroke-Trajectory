"""
Phase 10 - proper U-Net evaluation metrics on val set.

Reports per-class quality metrics that tell us whether the boundary class
is actually being learned well:

  Per-class semantic metrics (vs GT 3-class mask):
    - IoU (Intersection over Union)
    - Precision
    - Recall
    - F1

  Stroke-instance metrics (vs GT instance mask):
    - Mean per-instance IoU (best-match)
    - Predicted vs GT stroke count

Run:
    python3 phase10/eval_unet_metrics.py
"""

from __future__ import annotations
import config
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PARENT_DIR))

from stroke_seg_dataset import (StrokeSegDataset, build_canvas_traj_dataset,
                                  line_level_split_indices)
from unet import StrokeUNet
from infer_unet import predict_semantic, extract_instances


def per_class_metrics(pred_argmax: np.ndarray, gt_semantic: np.ndarray,
                       n_classes: int = 3):
    """Returns dict per class: {iou, precision, recall, f1, support}."""
    out = {}
    for c in range(n_classes):
        pred_c = (pred_argmax == c)
        gt_c = (gt_semantic == c)
        tp = int((pred_c & gt_c).sum())
        fp = int((pred_c & ~gt_c).sum())
        fn = int((~pred_c & gt_c).sum())
        union = tp + fp + fn
        iou = tp / union if union > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(gt_c.sum())
        out[c] = {"iou": iou, "precision": precision, "recall": recall,
                   "f1": f1, "support": support}
    return out


def best_match_instance_iou(pred_inst: np.ndarray, gt_inst: np.ndarray):
    """For each GT stroke, find the best-overlapping predicted stroke.
    Returns mean IoU across GT strokes.
    """
    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids > 0]
    if len(gt_ids) == 0:
        return 0.0, 0, int(pred_inst.max())
    pred_ids = np.unique(pred_inst)
    pred_ids = pred_ids[pred_ids > 0]
    if len(pred_ids) == 0:
        return 0.0, len(gt_ids), 0

    ious = []
    for g in gt_ids:
        gt_m = (gt_inst == g)
        best = 0.0
        for p in pred_ids:
            pred_m = (pred_inst == p)
            inter = int((gt_m & pred_m).sum())
            if inter == 0:
                continue
            union = int((gt_m | pred_m).sum())
            iou = inter / union if union > 0 else 0.0
            if iou > best:
                best = iou
        ious.append(best)
    return float(np.mean(ious)), len(gt_ids), len(pred_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default=config.MASTER_RAW_CSV)
    parser.add_argument("--ckpt", type=str,
                        default=str(config.UNET_DEPLOY))
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=100,
                        help="number of val samples to evaluate (0 = all)")
    parser.add_argument("--thresh-interior", type=float, default=0.4)
    parser.add_argument("--thresh-boundary", type=float, default=0.7)
    parser.add_argument("--min-area", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = StrokeUNet(n_classes=3, base_ch=16).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"[eval] loaded U-Net from {args.ckpt}")

    base = build_canvas_traj_dataset(args.csv)
    _, val_idx = line_level_split_indices(len(base), val_frac=args.val_frac,
                                             seed=args.seed)
    if args.n_samples > 0:
        val_idx = val_idx[:args.n_samples]
    val_ds = StrokeSegDataset(base, augment=False, seed=args.seed)
    print(f"[eval] evaluating on {len(val_idx)} val samples")

    cls_iou = [[] for _ in range(3)]
    cls_prec = [[] for _ in range(3)]
    cls_rec = [[] for _ in range(3)]
    cls_f1 = [[] for _ in range(3)]
    cls_support = [0 for _ in range(3)]
    inst_ious = []
    gt_counts = []
    pred_counts = []

    for k, idx in enumerate(val_idx):
        s = val_ds[int(idx)]
        img = s["image"].squeeze(0).numpy()
        gt_sem = s["semantic"].numpy()
        gt_inst = s["instance_mask"].numpy()

        probs = predict_semantic(model, img, device)
        pred_arg = probs.argmax(axis=0)
        pred_inst = extract_instances(probs, args.thresh_interior,
                                        args.thresh_boundary,
                                        min_area=args.min_area)

        cm = per_class_metrics(pred_arg, gt_sem, n_classes=3)
        for c in range(3):
            cls_iou[c].append(cm[c]["iou"])
            cls_prec[c].append(cm[c]["precision"])
            cls_rec[c].append(cm[c]["recall"])
            cls_f1[c].append(cm[c]["f1"])
            cls_support[c] += cm[c]["support"]

        inst_iou, n_gt, n_pred = best_match_instance_iou(pred_inst, gt_inst)
        inst_ious.append(inst_iou)
        gt_counts.append(n_gt)
        pred_counts.append(n_pred)

        if (k + 1) % 20 == 0 or (k + 1) == len(val_idx):
            print(f"  [{k+1}/{len(val_idx)}]")

    class_names = ["background", "interior", "boundary"]
    print()
    print("=" * 78)
    print("PER-CLASS SEMANTIC METRICS  (mean over val samples)")
    print("=" * 78)
    print(f"  {'class':12s}  {'IoU':>7s}  {'Prec':>7s}  {'Recall':>7s}  "
          f"{'F1':>7s}  {'support':>10s}")
    for c in range(3):
        print(f"  {class_names[c]:12s}  "
              f"{np.mean(cls_iou[c]):7.4f}  "
              f"{np.mean(cls_prec[c]):7.4f}  "
              f"{np.mean(cls_rec[c]):7.4f}  "
              f"{np.mean(cls_f1[c]):7.4f}  "
              f"{cls_support[c]:>10d}")
    print()
    total = sum(cls_support)
    if total > 0:
        print(f"  pixel distribution: bg={cls_support[0]/total*100:.1f}%  "
              f"interior={cls_support[1]/total*100:.1f}%  "
              f"boundary={cls_support[2]/total*100:.1f}%")
    print()
    print("=" * 78)
    print("STROKE-INSTANCE METRICS")
    print("=" * 78)
    print(f"  mean per-instance IoU (best match):  {np.mean(inst_ious):.4f}")
    print(f"  mean GT strokes / line:    {np.mean(gt_counts):.1f}")
    print(f"  mean pred strokes / line:  {np.mean(pred_counts):.1f}")
    over_seg = (np.array(pred_counts) > np.array(gt_counts)).mean() * 100
    under_seg = (np.array(pred_counts) < np.array(gt_counts)).mean() * 100
    print(f"  over-segmented:  {over_seg:.1f}%  of lines")
    print(f"  under-segmented: {under_seg:.1f}%  of lines")


if __name__ == "__main__":
    main()
