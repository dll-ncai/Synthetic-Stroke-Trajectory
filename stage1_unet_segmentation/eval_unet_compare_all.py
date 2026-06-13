"""
Phase 10 - combined comparison table across all U-Net variants.

Evaluates each saved checkpoint on the SAME val samples and prints a clean
side-by-side comparison table with:
  - Per-class boundary metrics (IoU, Precision, Recall, F1)
  - Per-instance segmentation metrics (mean IoU, under/over-seg %, stroke counts)

Helps select the best U-Net for the production pipeline.
"""

from __future__ import annotations
import config
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PARENT_DIR))

from stroke_seg_dataset import (StrokeSegDataset, build_canvas_traj_dataset,
                                  line_level_split_indices)
from unet import StrokeUNet
from infer_unet import predict_semantic, extract_instances
from eval_unet_metrics import per_class_metrics, best_match_instance_iou


# All variants to compare
MODELS = [
    ("Baseline (weighted CE)", "unet_out/unet_best.pt"),
    ("Focal alpha=5",          "unet_focal_out/unet_best.pt"),
    ("Focal + dilated GT",     "unet_focal_dilated_out/unet_best.pt"),
    ("Distance-transform",     "unet_distance_out/unet_best.pt"),
    ("Thinner (2-4px)",        "unet_thinner_out/unet_best.pt"),
]


def evaluate_model(ckpt_path: str, val_ds, val_idx, device,
                     thresh_interior=0.4, thresh_boundary=0.7, min_area=2):
    """Returns dict with all metrics for one model."""
    model = StrokeUNet(n_classes=3, base_ch=16).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    bnd_iou_l, bnd_prec_l, bnd_rec_l, bnd_f1_l = [], [], [], []
    int_iou_l = []
    inst_ious = []
    gt_counts, pred_counts = [], []

    for idx in val_idx:
        s = val_ds[int(idx)]
        img = s["image"].squeeze(0).numpy()
        gt_sem = s["semantic"].numpy()
        gt_inst = s["instance_mask"].numpy()

        probs = predict_semantic(model, img, device)
        pred_arg = probs.argmax(axis=0)
        pred_inst = extract_instances(probs, thresh_interior,
                                        thresh_boundary, min_area=min_area)

        cm = per_class_metrics(pred_arg, gt_sem, n_classes=3)
        bnd_iou_l.append(cm[2]["iou"])
        bnd_prec_l.append(cm[2]["precision"])
        bnd_rec_l.append(cm[2]["recall"])
        bnd_f1_l.append(cm[2]["f1"])
        int_iou_l.append(cm[1]["iou"])

        inst_iou, n_gt, n_pred = best_match_instance_iou(pred_inst, gt_inst)
        inst_ious.append(inst_iou)
        gt_counts.append(n_gt)
        pred_counts.append(n_pred)

    return {
        "bnd_iou": float(np.mean(bnd_iou_l)),
        "bnd_prec": float(np.mean(bnd_prec_l)),
        "bnd_rec": float(np.mean(bnd_rec_l)),
        "bnd_f1": float(np.mean(bnd_f1_l)),
        "int_iou": float(np.mean(int_iou_l)),
        "inst_iou": float(np.mean(inst_ious)),
        "gt_strokes": float(np.mean(gt_counts)),
        "pred_strokes": float(np.mean(pred_counts)),
        "over_seg": float((np.array(pred_counts) > np.array(gt_counts)).mean()),
        "under_seg": float((np.array(pred_counts) < np.array(gt_counts)).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default=config.MASTER_RAW_CSV)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--thresh-interior", type=float, default=0.4)
    parser.add_argument("--thresh-boundary", type=float, default=0.7)
    parser.add_argument("--min-area", type=int, default=2)
    parser.add_argument("--thickness-min", type=int, default=None,
                        help="if set, rasterize val at this thickness (overrides default)")
    parser.add_argument("--thickness-max", type=int, default=None,
                        help="if set, rasterize val at this thickness (overrides default)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    base = build_canvas_traj_dataset(args.csv)
    _, val_idx = line_level_split_indices(len(base),
                                              val_frac=args.val_frac,
                                              seed=args.seed)
    if args.n_samples > 0:
        val_idx = val_idx[:args.n_samples]
    if args.thickness_min is not None and args.thickness_max is not None:
        val_ds = StrokeSegDataset(base, augment=False, seed=args.seed,
                                    thickness_range=(args.thickness_min,
                                                       args.thickness_max))
        print(f"Val rasterized at thickness {args.thickness_min}-{args.thickness_max}px")
    else:
        val_ds = StrokeSegDataset(base, augment=False, seed=args.seed)
    print(f"Evaluating {len(MODELS)} models on {len(val_idx)} val samples...\n")

    results = []
    for name, rel_ckpt in MODELS:
        ckpt = THIS_DIR / rel_ckpt
        if not ckpt.exists():
            print(f"  [skip] {name}: {ckpt} not found")
            continue
        print(f"  evaluating: {name}")
        m = evaluate_model(str(ckpt), val_ds, val_idx, device,
                              args.thresh_interior, args.thresh_boundary,
                              args.min_area)
        results.append((name, m))

    print()
    print("=" * 116)
    print("PER-CLASS BOUNDARY METRICS")
    print("=" * 116)
    print(f"  {'Model':28s}  {'IoU':>8s}  {'Prec':>8s}  {'Recall':>8s}  "
          f"{'F1':>8s}  {'IntIoU':>8s}")
    print("  " + "-" * 80)
    for name, m in results:
        print(f"  {name:28s}  "
              f"{m['bnd_iou']:8.4f}  {m['bnd_prec']:8.4f}  "
              f"{m['bnd_rec']:8.4f}  {m['bnd_f1']:8.4f}  "
              f"{m['int_iou']:8.4f}")

    print()
    print("=" * 116)
    print("PER-INSTANCE STROKE SEGMENTATION METRICS")
    print("=" * 116)
    print(f"  {'Model':28s}  {'InstIoU':>9s}  {'GT/line':>9s}  "
          f"{'Pred/line':>11s}  {'OverSeg%':>10s}  {'UnderSeg%':>11s}")
    print("  " + "-" * 100)
    for name, m in results:
        print(f"  {name:28s}  "
              f"{m['inst_iou']:9.4f}  "
              f"{m['gt_strokes']:9.1f}  {m['pred_strokes']:11.1f}  "
              f"{m['over_seg']*100:10.1f}  {m['under_seg']*100:11.1f}")

    # Selection summary
    print()
    print("=" * 116)
    print("BEST BY EACH METRIC")
    print("=" * 116)
    if results:
        for metric, label, lower_is_better in [
            ("bnd_iou", "Boundary IoU", False),
            ("bnd_f1", "Boundary F1", False),
            ("inst_iou", "Per-instance IoU", False),
            ("under_seg", "Under-segmentation %", True),
            ("over_seg", "Over-segmentation %", True),
        ]:
            best = min(results, key=lambda x: x[1][metric]) if lower_is_better \
                   else max(results, key=lambda x: x[1][metric])
            print(f"  Best {label:30s}: {best[0]:28s}  "
                  f"({best[1][metric]:.4f})")

    print()
    print("=" * 116)
    print("LaTeX TABLE (paste into thesis)")
    print("=" * 116)
    print("\\begin{table}[!th]")
    print("\\centering")
    print("\\begin{tabular}{|l|c|c|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Method} & \\textbf{Bnd IoU} & \\textbf{Bnd F1} & "
          "\\textbf{Inst IoU} & \\textbf{Under-seg \\%} & "
          "\\textbf{Over-seg \\%} \\\\ \\hline")
    for name, m in results:
        print(f"{name} & {m['bnd_iou']:.3f} & {m['bnd_f1']:.3f} & "
              f"{m['inst_iou']:.3f} & {m['under_seg']*100:.1f} & "
              f"{m['over_seg']*100:.1f} \\\\ \\hline")
    print("\\end{tabular}")
    print("\\caption{Comparison of U-Net training variants on 100 validation lines. "
          "Best per-instance IoU indicates the cleanest stroke separation; "
          "lower under-segmentation indicates better recovery of touching strokes.}")
    print("\\label{tab:unet-comparison}")
    print("\\end{table}")


if __name__ == "__main__":
    main()
