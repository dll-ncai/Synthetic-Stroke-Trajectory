"""
Phase 11 - dataset wrappers for recognition (matches ICDAR-paper conventions).

All datasets yield: (tokens, text_ids, n_tokens, n_text)
    - tokens:    torch.float32 (T, 3)   [x_zscore, y_zscore, pen_state]
    - text_ids:  torch.long    (L,)     1..(V-1), blank=0 reserved
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path
from typing import Optional, List, Dict  # noqa: F401

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARENT_DIR / "phase5b"))
sys.path.insert(0, str(PARENT_DIR / "phase7"))
sys.path.insert(0, str(PARENT_DIR / "phase9"))
sys.path.insert(0, str(PARENT_DIR / "phase10"))

from trajectory_dataset import TrajectoryDataset

from unet import StrokeUNet
from infer_unet import predict_semantic, extract_instances
from clean_preprocess import preprocess_uhwr_image_clean
from test_uhwr import preprocess_uhwr_image
from full_pipeline_unet_uhwr import instance_mask_to_components
from template_stitcher import extract_skeleton
from mask_to_training_patch import (
    order_skeleton_multi_pass, _smooth_polyline,
    mask_to_training_style_patch,
)

# For learned order prediction over Path B detected strokes
from cnn_classifier import StrokeCNNClassifier
from knn_retrieval import make_embedding_extractor
from order_predictor import load_order_predictor, extract_spatial_features

from vocab_loader import encode_text
from traj_tokenizer import (
    online_traj_to_tokens, path_b_polylines_to_tokens,
    drop_near_duplicates, cap_sequence_length,
    arc_length_resample_per_stroke, simplify_trajectory,
)


# Calibrated against ONLINE training data median within-stroke step size
# after z-score normalization (see phase11/calibrate_step_size.py).
ONLINE_TARGET_STEP = 0.117

# Whether to apply trajectory simplification (uniform spacing + light
# smoothing). Enables BOTH online (during training) and Path B (during
# inference) cleanup so the styles converge.
#
# Selectable via the STROKE_SIMPLIFY env var (default on). The real-only
# baseline checkpoint was trained on RAW trajectories, so reproduce its
# reported CER with STROKE_SIMPLIFY=0; the two-stage model uses the default.
APPLY_SIMPLIFICATION = os.environ.get("STROKE_SIMPLIFY", "1").lower() not in ("0", "false", "no")


# ----------------------------------------------------------------------
# Online dataset
# ----------------------------------------------------------------------

class OnlineRecogDataset(Dataset):
    def __init__(self, meta_csv_path: str, char_to_idx: Dict[str, int],
                 indices: Optional[List[int]] = None,
                 max_points: int = 2000):
        self.base = TrajectoryDataset(
            meta_csv_path=meta_csv_path,
            normalize_coords=True,
            normalization_mode="canvas",
        )
        self.indices = indices if indices is not None else list(range(len(self.base)))
        self.char_to_idx = char_to_idx
        self.max_points = max_points
        # Pre-filter: drop samples whose text encodes to empty after
        # normalization (would cause CTC trouble)
        good = []
        for k, di in enumerate(self.indices):
            text = str(self.base.meta_df.iloc[di].get("line", "")).strip()
            ids = encode_text(text, char_to_idx)
            if 0 < len(ids):
                good.append(di)
        if len(good) != len(self.indices):
            print(f"[OnlineRecogDataset] filtered {len(self.indices) - len(good)} "
                  f"samples with empty/invalid text")
            self.indices = good

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        di = self.indices[k]
        _, traj_t, traj_len, _ = self.base[di]
        traj = traj_t.numpy()[:traj_len]
        tokens = online_traj_to_tokens(traj)
        # SIMPLIFY: uniform arc-length + light Gaussian smoothing.
        # Removes tablet jitter and velocity variation while preserving
        # writer's stroke order and within-stroke direction. Makes online
        # trajectories structurally closer to Path B at inference time.
        if APPLY_SIMPLIFICATION:
            tokens = simplify_trajectory(tokens,
                                          target_step=ONLINE_TARGET_STEP,
                                          smooth_sigma=0.6)
        tokens = drop_near_duplicates(tokens)
        tokens = cap_sequence_length(tokens, max_T=self.max_points)
        text = str(self.base.meta_df.iloc[di].get("line", "")).strip()
        ids = encode_text(text, self.char_to_idx)
        return (
            torch.from_numpy(tokens).float(),
            torch.tensor(ids, dtype=torch.long),
            int(tokens.shape[0]),
            int(len(ids)),
        )


# ----------------------------------------------------------------------
# Path B trajectory extractor
# ----------------------------------------------------------------------

class PathBExtractor:
    def __init__(self, unet_ckpt: str, device: torch.device,
                 thresh_interior: float = 0.4,
                 thresh_boundary: float = 0.7,
                 min_area: int = 2,
                 min_noise_area: int = 2,
                 diacritic_keep_distance: float = 15.0,
                 smooth_sigma: float = 1.2,
                 # Optional learned stroke ordering (better than R-to-L sort
                 # because it learned the diacritic-after-main pattern)
                 classifier_ckpt: Optional[str] = None,
                 vocab_dir: Optional[str] = None,
                 order_ckpt: Optional[str] = None,
                 knn_index: Optional[str] = None):
        self.device = device
        self.unet = StrokeUNet(n_classes=3, base_ch=16).to(device)
        ck = torch.load(unet_ckpt, map_location=device)
        self.unet.load_state_dict(ck["state_dict"])
        self.unet.eval()
        self.thresh_interior = thresh_interior
        self.thresh_boundary = thresh_boundary
        self.min_area = min_area
        self.min_noise_area = min_noise_area
        self.diacritic_keep_distance = diacritic_keep_distance
        self.smooth_sigma = smooth_sigma

        # Set up learned ordering if all needed paths provided
        self.use_learned_order = False
        self.classifier = None
        self.extractor_fn = None
        self.order_model = None
        if (classifier_ckpt and vocab_dir and order_ckpt and knn_index):
            try:
                vocab = np.load(Path(vocab_dir) / "vocab.npz")
                K_total = (int(vocab["k_total"]) if "k_total" in vocab.files
                           else int(vocab["k"]) + 1)
                self.classifier = StrokeCNNClassifier(
                    n_classes=K_total, hidden=256, dropout=0.0,
                ).to(device)
                self.classifier.load_state_dict(
                    torch.load(classifier_ckpt, map_location=device),
                )
                self.classifier.eval()
                self.extractor_fn = make_embedding_extractor(self.classifier)
                idx = np.load(knn_index)
                embed_dim = int(idx["embeddings"].shape[1])
                self.order_model = load_order_predictor(
                    order_ckpt, embed_dim=embed_dim,
                )
                self.order_model = self.order_model.to(device).eval()
                self.use_learned_order = True
                print("[PathBExtractor] learned order predictor enabled")
            except Exception as e:
                print(f"[PathBExtractor] could not enable learned order: {e}")
                self.use_learned_order = False

    def __call__(self, raw_gray: np.ndarray) -> np.ndarray:
        image, _ = preprocess_uhwr_image(raw_gray)
        pp = preprocess_uhwr_image_clean(
            (np.clip(image, 0, 1) * 255).astype(np.uint8),
            min_noise_area=self.min_noise_area,
            diacritic_keep_distance=self.diacritic_keep_distance,
            compute_skeleton=False,
        )
        clean = pp["clean_binary"]
        H, W = clean.shape
        probs = predict_semantic(self.unet, clean, self.device)
        inst = extract_instances(probs, self.thresh_interior,
                                  self.thresh_boundary, self.min_area)
        comps = instance_mask_to_components(inst, self.min_area)
        n = len(comps)
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # ---- Decide stroke order ----
        if self.use_learned_order and n > 0:
            # Use the learned order predictor (trained with Pearson r=0.97
            # on online data). It knows the diacritic-after-main pattern.
            patches = np.stack([mask_to_training_style_patch(c) for c in comps])
            batch = torch.from_numpy(patches).unsqueeze(1).float().to(self.device)
            with torch.no_grad():
                embeddings = self.extractor_fn(batch).cpu().numpy()
            spatial = np.zeros((n, 6), dtype=np.float32)
            for i, c in enumerate(comps):
                spatial[i] = extract_spatial_features(
                    c["bbox"], n, canvas_size=(H, W),
                )
            with torch.no_grad():
                emb_t = torch.from_numpy(embeddings).float().to(self.device)
                sp_t = torch.from_numpy(spatial).float().to(self.device)
                order_scores = self.order_model(emb_t, sp_t).cpu().numpy()
            sort_idx = np.argsort(order_scores)
        else:
            # Fallback: right-to-left by x_center
            sort_idx = np.argsort([-c["x_center"] for c in comps])

        polylines_per_stroke = []
        for ci in sort_idx:
            c = comps[int(ci)]
            bx0, by0, bx1, by1 = c["bbox"]
            blob = c["mask"][by0:by1, bx0:bx1]
            if blob.sum() < 3:
                pt = np.array([[c["x_center"], c["y_center"]]], dtype=np.float32)
                polylines_per_stroke.append([pt]); continue
            skel = extract_skeleton(blob)
            if skel.sum() < 2:
                pt = np.array([[c["x_center"], c["y_center"]]], dtype=np.float32)
                polylines_per_stroke.append([pt]); continue
            subs = []
            for sp in order_skeleton_multi_pass(skel):
                sp = _smooth_polyline(sp, sigma=self.smooth_sigma)
                canvas = sp.copy()
                canvas[:, 0] += bx0; canvas[:, 1] += by0
                subs.append(canvas)
            polylines_per_stroke.append(subs)
        return path_b_polylines_to_tokens(polylines_per_stroke)


# ----------------------------------------------------------------------
# UHWR dataset
# ----------------------------------------------------------------------

class UHWRRecogDataset(Dataset):
    def __init__(self, label_file: str, uhwr_root: str,
                 char_to_idx: Dict[str, int], extractor: PathBExtractor,
                 max_samples: Optional[int] = None,
                 max_points: int = 2000):
        self.uhwr_root = Path(uhwr_root)
        self.char_to_idx = char_to_idx
        self.extractor = extractor
        self.max_points = max_points
        with open(label_file, "r", encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]
        self.entries = []
        for line in lines:
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            img_rel, text = parts
            ids = encode_text(text.strip(), char_to_idx)
            if len(ids) == 0:
                continue
            self.entries.append((img_rel.strip(), text.strip(), ids))
        if max_samples is not None:
            self.entries = self.entries[:max_samples]
        print(f"[UHWRRecogDataset] {len(self.entries)} usable entries "
              f"from {label_file}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, k):
        img_rel, _text, ids = self.entries[k]
        img_path = self.uhwr_root / img_rel
        raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(img_path)
        tokens = self.extractor(raw)
        # Apply same simplification used during online training (uniform
        # arc-length + light Gaussian smooth). Makes the inference
        # distribution match the training distribution structurally.
        if APPLY_SIMPLIFICATION:
            tokens = simplify_trajectory(tokens,
                                          target_step=ONLINE_TARGET_STEP,
                                          smooth_sigma=0.6)
        else:
            tokens = arc_length_resample_per_stroke(tokens,
                                                      target_step=ONLINE_TARGET_STEP)
        tokens = drop_near_duplicates(tokens)
        tokens = cap_sequence_length(tokens, max_T=self.max_points)
        return (
            torch.from_numpy(tokens).float(),
            torch.tensor(ids, dtype=torch.long),
            int(tokens.shape[0]),
            int(len(ids)),
        )


# ----------------------------------------------------------------------
# UPTI dataset
# ----------------------------------------------------------------------

class UPTIRecogDataset(Dataset):
    def __init__(self, img_zip_path: str, label_zip_path: str,
                 char_to_idx: Dict[str, int], extractor: PathBExtractor,
                 split: str = "test", resolution: str = "low",
                 max_samples: Optional[int] = None,
                 max_points: int = 2000):
        self.char_to_idx = char_to_idx
        self.extractor = extractor
        self.max_points = max_points
        self.img_zip = zipfile.ZipFile(img_zip_path, "r")
        self.lbl_zip = zipfile.ZipFile(label_zip_path, "r")
        label_set = set(self.lbl_zip.namelist())
        img_files = [
            f for f in self.img_zip.namelist()
            if f.startswith(f"images/{split}/")
            and f.endswith(".png")
            and f"/{resolution}/" in f
        ]
        self.entries = []
        for ip in img_files:
            sid = Path(ip).stem
            lp = f"groundtruth/{split}/{sid}.txt"
            if lp not in label_set:
                continue
            text = self.lbl_zip.read(lp).decode("utf-8", errors="ignore").strip()
            ids = encode_text(text, char_to_idx)
            if len(ids) == 0:
                continue
            self.entries.append((ip, ids))
        if max_samples is not None:
            self.entries = self.entries[:max_samples]
        print(f"[UPTIRecogDataset] {len(self.entries)} usable entries "
              f"({split}/{resolution})")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, k):
        ip, ids = self.entries[k]
        data = self.img_zip.read(ip)
        raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        tokens = self.extractor(raw)
        # Apply same simplification used during online training (uniform
        # arc-length + light Gaussian smooth). Makes the inference
        # distribution match the training distribution structurally.
        if APPLY_SIMPLIFICATION:
            tokens = simplify_trajectory(tokens,
                                          target_step=ONLINE_TARGET_STEP,
                                          smooth_sigma=0.6)
        else:
            tokens = arc_length_resample_per_stroke(tokens,
                                                      target_step=ONLINE_TARGET_STEP)
        tokens = drop_near_duplicates(tokens)
        tokens = cap_sequence_length(tokens, max_T=self.max_points)
        return (
            torch.from_numpy(tokens).float(),
            torch.tensor(ids, dtype=torch.long),
            int(tokens.shape[0]),
            int(len(ids)),
        )


# ----------------------------------------------------------------------
# Collate (ICDAR-style: concatenated targets for CTC)
# ----------------------------------------------------------------------

def collate_recog(batch):
    """Returns:
        x_pad: (B, T_max, 3)
        y_cat: (sum_L,)  — concatenated targets
        input_lengths: (B,)
        target_lengths: (B,)
    """
    from torch.nn.utils.rnn import pad_sequence
    xs, ys, n_tok, n_text = zip(*batch)
    input_lengths = torch.tensor(n_tok, dtype=torch.long)
    target_lengths = torch.tensor(n_text, dtype=torch.long)
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)
    y_cat = torch.cat(ys, dim=0)
    return x_pad, y_cat, input_lengths, target_lengths
