"""
Phase 10 - test U-Net stroke segmenter on completely external UHWR scans.

This is the TRUE deployment test. No ground truth available; we only
report:
    - number of predicted stroke instances per line
    - visualization: original scan / U-Net argmax / predicted instances

The UHWR images may have:
    - Different ink/background polarity than our training rendering
    - Variable aspect ratios (lines can be much wider than 1600)
    - Different stroke thickness / writing style
    - Paper noise, scan artifacts

We detect ink polarity automatically (majority pixel = background) and
resize-to-fit 64x1600 with letterboxing.

Usage:
    python3 phase10/test_uhwr.py
    python3 phase10/test_uhwr.py --image-dir /path/to/images --n-samples 8
    python3 phase10/test_uhwr.py --image-files 00100.jpg,00200.jpg
"""

from __future__ import annotations
import config

import argparse
import sys
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from unet import StrokeUNet
from infer_unet import predict_semantic, extract_instances, colorize_instance


DEFAULT_UHWR_DIR = config.UHWR_IMAGES_DIR
TARGET_H, TARGET_W = 64, 1600


def preprocess_uhwr_image(img: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Convert a raw UHWR scan into the U-Net input format.

    Steps:
        1. Grayscale -> float [0, 1]
        2. Auto-detect polarity: if mean > 0.5 the image is white-on-black
           "ink=dark", so we invert to "ink=1, bg=0" (our training convention).
        3. Resize so height fits TARGET_H, preserving aspect ratio.
        4. Center-pad width to TARGET_W (or center-crop if too wide).

    Returns (image, meta) - meta has the placement info for inverse-mapping.
    """
    h, w = img.shape
    img_f = img.astype(np.float32) / 255.0
    # Mean of a clean white-on-black scan is high (paper > 0.5); invert so ink=1
    if img_f.mean() > 0.5:
        img_f = 1.0 - img_f

    # Resize so the longest dim (typically width) lands at TARGET_W and
    # height fits within TARGET_H (preserving aspect).
    scale_h = TARGET_H / h
    scale_w = TARGET_W / w
    scale = min(scale_h, scale_w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    if new_h > 0 and new_w > 0:
        img_resized = cv2.resize(img_f, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img_f

    padded = np.zeros((TARGET_H, TARGET_W), dtype=np.float32)
    y_off = (TARGET_H - new_h) // 2
    x_off = (TARGET_W - new_w) // 2
    if y_off >= 0 and x_off >= 0:
        padded[y_off:y_off + new_h, x_off:x_off + new_w] = img_resized
    else:
        # Line too wide; crop centrally
        crop_y = max(-y_off, 0)
        crop_x = max(-x_off, 0)
        cut_h = min(new_h - crop_y, TARGET_H)
        cut_w = min(new_w - crop_x, TARGET_W)
        padded[:cut_h, :cut_w] = img_resized[crop_y:crop_y + cut_h,
                                              crop_x:crop_x + cut_w]

    meta = {
        "orig_h": int(h), "orig_w": int(w),
        "new_h": int(new_h), "new_w": int(new_w),
        "y_off": int(y_off), "x_off": int(x_off),
        "scale": float(scale),
    }
    return padded, meta


def run_on_image(model, img_path, device, thresh_interior, thresh_boundary,
                  min_area):
    raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return None
    img, meta = preprocess_uhwr_image(raw)
    probs = predict_semantic(model, img, device)
    inst = extract_instances(probs, thresh_interior, thresh_boundary, min_area)
    return {
        "raw": raw, "image": img, "probs": probs,
        "instance_mask": inst, "n_components": int(inst.max()),
        "meta": meta,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=str(config.UNET_DEPLOY))
    parser.add_argument("--image-dir", type=str, default=DEFAULT_UHWR_DIR)
    parser.add_argument("--image-files", type=str, default=None,
                        help="comma-separated specific filenames in --image-dir")
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thresh-interior", type=float, default=0.4)
    parser.add_argument("--thresh-boundary", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=15)
    parser.add_argument("--out-dir", type=str,
                        default=str(THIS_DIR / "uhwr_test"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[uhwr] device={device}")

    model = StrokeUNet(n_classes=3, base_ch=16).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[uhwr] loaded checkpoint: {args.ckpt}")
    if "best_val" in ckpt:
        print(f"        best_val={ckpt['best_val']:.4f}  epoch={ckpt.get('epoch', '?')}")

    img_dir = Path(args.image_dir)
    all_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not all_files:
        print(f"[uhwr] no images in {img_dir}")
        return
    print(f"[uhwr] found {len(all_files)} images")

    if args.image_files is not None:
        names = [s.strip() for s in args.image_files.split(",")]
        picks = [img_dir / n for n in names if (img_dir / n).exists()]
    else:
        rng = np.random.RandomState(args.seed)
        picks = rng.choice(all_files,
                            size=min(args.n_samples, len(all_files)),
                            replace=False).tolist()

    print(f"[uhwr] running on {len(picks)} images")
    print(f"[uhwr] thresh_interior={args.thresh_interior}  "
          f"thresh_boundary={args.thresh_boundary}  min_area={args.min_area}")

    n = len(picks)
    fig, axes = plt.subplots(3 * n, 1, figsize=(15, 1.4 * 3 * n))
    if 3 * n == 1:
        axes = np.array(axes).reshape(-1)

    counts = []
    for i, p in enumerate(picks):
        result = run_on_image(model, p, device,
                               args.thresh_interior, args.thresh_boundary,
                               args.min_area)
        if result is None:
            print(f"  [skip] could not read {p.name}")
            continue
        n_comp = result["n_components"]
        counts.append(n_comp)
        print(f"  {p.name}: {n_comp} predicted stroke instances")

        ax = axes[i * 3 + 0]
        ax.imshow(result["image"], cmap="gray_r", aspect="auto")
        ax.set_title(f"{p.name} - preprocessed (orig {result['meta']['orig_h']}"
                     f"x{result['meta']['orig_w']}, "
                     f"-> {result['meta']['new_h']}x{result['meta']['new_w']})",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[i * 3 + 1]
        # boundary probability heatmap to inspect what U-Net is doing
        ax.imshow(result["probs"][2], cmap="hot", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"U-Net boundary probability (hot = predicted boundary)",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[i * 3 + 2]
        ax.imshow(colorize_instance(result["instance_mask"]), aspect="auto")
        ax.set_title(f"Predicted stroke instances: {n_comp} components",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    out_viz = out_dir / "uhwr_predictions.png"
    plt.savefig(out_viz, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print()
    print("=" * 72)
    print(f"[uhwr] component count summary over {len(counts)} images:")
    if counts:
        arr = np.asarray(counts)
        print(f"        mean={arr.mean():.1f}  median={np.median(arr):.1f}  "
              f"min={arr.min()}  max={arr.max()}")
        print(f"        all counts: {counts}")
    print(f"[uhwr] visualizations saved to: {out_viz}")
    print()
    print("Reference: for context, online lines typically have 25-50 strokes")
    print("per line. If UHWR predictions land in that ballpark, the U-Net")
    print("is generalizing across domains. If predictions are <10 or >100,")
    print("there's a domain-gap issue we need to address.")


if __name__ == "__main__":
    main()
