"""
Online trajectory dataset loader.

Supported coordinate spaces:
  - normalize_coords=False: raw tablet coordinates
  - normalize_coords=True, normalization_mode="canvas": image pixel space
  - normalize_coords=True, normalization_mode="minmax": shared-span normalized space

The "minmax" mode uses the same span for both axes:
    span = max(x_max - x_min, y_max - y_min)
    x_norm = (x - x_min) / span
    y_norm = (y - y_min) / span
This preserves aspect ratio and smoothness instead of independently squashing Y.
"""

import os
import config
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        meta_csv_path,
        img_size=(64, 1600),
        max_traj_len=None,
        base_path=config.DATA_ROOT_STR,
        distance_thin=True,
        pen_down_min_dist_px=45.0,
        pen_up_min_dist_px=100.0,
        max_raw_points=None,
        normalize_coords=True,
        normalization_mode="minmax",
        filter_outliers=False,
    ):
        self.meta_csv_path = meta_csv_path
        self.meta_root = os.path.dirname(os.path.abspath(meta_csv_path))
        self.img_height, self.img_width = img_size
        self.max_traj_len = max_traj_len
        self.base_path = base_path
        self.distance_thin = distance_thin
        self.pen_down_min_dist_px = float(pen_down_min_dist_px)
        self.pen_up_min_dist_px = float(pen_up_min_dist_px)
        self.max_raw_points = max_raw_points
        self.normalize_coords = bool(normalize_coords)
        self.normalization_mode = normalization_mode or "minmax"
        self.filter_outliers = filter_outliers

        if self.normalization_mode not in {"canvas", "minmax"}:
            raise ValueError(
                f"Unsupported normalization_mode={self.normalization_mode!r}. "
                "Expected 'canvas' or 'minmax'."
            )

        meta_df = pd.read_csv(meta_csv_path)
        self.dropped_long_samples = 0

        if self.max_raw_points is not None:
            keep_rows = []
            for _, row in meta_df.iterrows():
                csv_path = self.resolve_data_path(row["csv"])
                if self.get_raw_trajectory_length(csv_path) <= int(self.max_raw_points):
                    keep_rows.append(row.to_dict())
                else:
                    self.dropped_long_samples += 1
            self.meta_df = pd.DataFrame(keep_rows)
        else:
            self.meta_df = meta_df

        coord_space = self._coord_space_name()
        print(
            f"[TrajectoryDataset] Loaded {len(self.meta_df)} samples"
            f" | dropped_long={self.dropped_long_samples}"
            f" | coord_space={coord_space}"
        )

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]

        img_path = self.resolve_data_path(row["img"])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        processed_img, image_meta = self.preprocess_image(image)

        csv_path = self.resolve_data_path(row["csv"])
        raw_traj = self.load_trajectory_raw(csv_path)

        if self.distance_thin:
            raw_traj = self.filter_trajectory(raw_traj)

        processed_traj, traj_meta = self.process_trajectory(raw_traj, image_meta)

        if self.max_traj_len is not None:
            processed_traj = processed_traj[: self.max_traj_len]

        traj_length = len(processed_traj)

        img_tensor = torch.from_numpy(processed_img).unsqueeze(0).float()
        traj_tensor = torch.from_numpy(processed_traj).float()

        sample_meta = {
            "id": row.get("id", idx),
            "img_path": img_path,
            "csv_path": csv_path,
            "image_meta": image_meta,
            "traj_meta": traj_meta,
        }

        return img_tensor, traj_tensor, traj_length, sample_meta

    def _coord_space_name(self):
        if not self.normalize_coords:
            return "raw"
        if self.normalization_mode == "canvas":
            return "canvas"
        return "minmax_shared_span"

    def process_trajectory(self, raw_traj: np.ndarray, image_meta: dict):
        traj_meta = {
            "coord_space": self._coord_space_name(),
            "raw_x_range": [float(raw_traj[:, 0].min()), float(raw_traj[:, 0].max())],
            "raw_y_range": [float(raw_traj[:, 1].min()), float(raw_traj[:, 1].max())],
        }

        if not self.normalize_coords:
            processed = raw_traj.astype(np.float32, copy=True)
            traj_meta["raw_passthrough"] = True
        elif self.normalization_mode == "canvas":
            processed = self._project_to_image(raw_traj, image_meta)
            traj_meta["image_px_x_range"] = [
                float(processed[:, 0].min()),
                float(processed[:, 0].max()),
            ]
            traj_meta["image_px_y_range"] = [
                float(processed[:, 1].min()),
                float(processed[:, 1].max()),
            ]
        else:
            processed, norm_params = self._normalize_minmax_shared_span(raw_traj)
            traj_meta["norm_params"] = norm_params
            traj_meta["normalized_x_range"] = [
                float(processed[:, 0].min()),
                float(processed[:, 0].max()),
            ]
            traj_meta["normalized_y_range"] = [
                float(processed[:, 1].min()),
                float(processed[:, 1].max()),
            ]

        return processed.astype(np.float32), traj_meta

    def preprocess_image(self, img: np.ndarray):
        img = img.astype(np.float32) / 255.0
        img = 1.0 - img

        h, w = img.shape
        scale_h = self.img_height / h
        scale_w = self.img_width / w
        scale = min(1.0, scale_h, scale_w)

        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        if new_h != h or new_w != w:
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img

        padded = np.zeros((self.img_height, self.img_width), dtype=np.float32)
        y_offset = (self.img_height - new_h) // 2
        x_offset = (self.img_width - new_w) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

        ink_mask = padded < 0.95
        if np.any(ink_mask):
            ys, xs = np.where(ink_mask)
            ink_bbox = {
                "x_min": int(xs.min()),
                "x_max": int(xs.max()),
                "y_min": int(ys.min()),
                "y_max": int(ys.max()),
            }
        else:
            ink_bbox = {
                "x_min": 0,
                "x_max": self.img_width - 1,
                "y_min": 0,
                "y_max": self.img_height - 1,
            }

        meta = {
            "orig_h": int(h),
            "orig_w": int(w),
            "new_h": int(new_h),
            "new_w": int(new_w),
            "scale": float(scale),
            "x_offset": int(x_offset),
            "y_offset": int(y_offset),
            "canvas_h": int(self.img_height),
            "canvas_w": int(self.img_width),
            "ink_bbox": ink_bbox,
        }
        return padded, meta

    def _normalize_minmax_shared_span(self, raw_traj: np.ndarray):
        traj = raw_traj.copy().astype(np.float32)
        x = traj[:, 0]
        y = traj[:, 1]

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        span = max(float(x_max - x_min), float(y_max - y_min), 1e-6)

        traj[:, 0] = (x - x_min) / span
        traj[:, 1] = (y - y_min) / span

        norm_params = {
            "x_min": float(x_min),
            "y_min": float(y_min),
            "span": float(span),
        }
        return traj, norm_params

    def denormalize_minmax(self, traj: np.ndarray, norm_params: dict) -> np.ndarray:
        out = traj.copy().astype(np.float32)
        span = float(norm_params["span"])
        out[:, 0] = out[:, 0] * span + float(norm_params["x_min"])
        out[:, 1] = out[:, 1] * span + float(norm_params["y_min"])
        return out

    def _project_to_image(self, raw_traj: np.ndarray, image_meta: dict) -> np.ndarray:
        traj = raw_traj.copy().astype(np.float32)
        x_raw = traj[:, 0]
        y_raw = traj[:, 1]

        x_min, x_max = x_raw.min(), x_raw.max()
        y_min, y_max = y_raw.min(), y_raw.max()

        x_span = max(x_max - x_min, 1e-6)
        y_span = max(y_max - y_min, 1e-6)

        orig_w = image_meta["orig_w"]
        orig_h = image_meta["orig_h"]

        x_norm = (x_raw - x_min) / x_span * orig_w
        y_norm = (y_raw - y_min) / y_span * orig_h

        scale = image_meta["scale"]
        x_offset = image_meta["x_offset"]
        y_offset = image_meta["y_offset"]

        x_img = x_norm * scale + x_offset
        y_img = y_norm * scale + y_offset

        x_img = np.clip(x_img, 0.0, image_meta["canvas_w"] - 1)
        y_img = np.clip(y_img, 0.0, image_meta["canvas_h"] - 1)

        out = traj.copy()
        out[:, 0] = x_img
        out[:, 1] = y_img
        return out

    def load_trajectory_raw(self, csv_path: str) -> np.ndarray:
        try:
            df = pd.read_csv(csv_path, delimiter="\t")
        except Exception:
            df = pd.read_csv(csv_path)

        if len(df.columns) == 1 and "," in str(df.columns[0]):
            df = pd.read_csv(csv_path)

        if "Time" in df.columns:
            df = df.sort_values("Time").reset_index(drop=True)

        x_col = self._pick_column(df, ["X cood.", "x", "X", "x_norm"])
        y_col = self._pick_column(df, ["Y cood.", "y", "Y", "y_norm"])
        x = df[x_col].values.astype(np.float32)
        y = df[y_col].values.astype(np.float32)

        if "Pressure" in df.columns:
            pen_state = (df["Pressure"].values <= 0).astype(np.float32)
        elif "Thickness" in df.columns:
            pen_state = (df["Thickness"].values <= 0).astype(np.float32)
        elif "pen_state" in df.columns:
            pen_state = (df["pen_state"].values <= 0).astype(np.float32)
        elif "pen_down" in df.columns:
            pen_state = (df["pen_down"].values <= 0).astype(np.float32)
        elif "Height" in df.columns:
            height = df["Height"].values
            uq = np.unique(height)
            pen_state = (
                (height == uq.min()).astype(np.float32)
                if len(uq) == 2
                else (height < 0).astype(np.float32)
            )
        else:
            pen_state = np.zeros_like(x, dtype=np.float32)

        return np.stack([x, y, pen_state], axis=1)

    def _pick_column(self, df: pd.DataFrame, candidates):
        for name in candidates:
            if name in df.columns:
                return name
        raise KeyError(f"None of {candidates} found. Available: {list(df.columns)}")

    def get_raw_trajectory_length(self, csv_path: str) -> int:
        try:
            df = pd.read_csv(csv_path, delimiter="\t")
        except Exception:
            df = pd.read_csv(csv_path)
        if len(df.columns) == 1 and "," in str(df.columns[0]):
            df = pd.read_csv(csv_path)
        return int(len(df))

    def filter_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        if trajectory.shape[0] <= 2:
            return trajectory
        segments = self._split_segments(trajectory)
        filtered = []
        for seg in segments:
            min_dist = (
                self.pen_down_min_dist_px if seg[0, 2] < 0.5 else self.pen_up_min_dist_px
            )
            seg = self._resample_segment(seg, min_dist)
            if seg.shape[0] > 0:
                filtered.append(seg)
        return self._combine_segments(filtered)

    def _split_segments(self, trajectory: np.ndarray):
        if trajectory.shape[0] == 0:
            return []
        segments, current = [], [trajectory[0].tolist()]
        cur_state = float(trajectory[0, 2])
        for row in trajectory[1:]:
            state = float(row[2])
            same = ((state < 0.5 and cur_state < 0.5) or (state >= 0.5 and cur_state >= 0.5))
            if same:
                current.append(row.tolist())
            else:
                segments.append(np.asarray(current, dtype=np.float32))
                current, cur_state = [row.tolist()], state
        if current:
            segments.append(np.asarray(current, dtype=np.float32))
        return segments

    def _combine_segments(self, segments):
        if not segments:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(segments, axis=0)

    def _resample_segment(self, segment: np.ndarray, step_px: float) -> np.ndarray:
        if segment.shape[0] < 3 or step_px <= 0:
            return segment
        pts = segment[:, :2].astype(np.float32)
        diffs = pts[1:] - pts[:-1]
        seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
        cumdist = np.concatenate(([0.0], np.cumsum(seg_lens, dtype=np.float32)))
        total_len = float(cumdist[-1])

        if total_len <= 1e-6:
            return np.asarray([segment[0], segment[-1]], dtype=np.float32)

        n_steps = max(2, int(np.floor(total_len / step_px)) + 1)
        if n_steps >= segment.shape[0]:
            return segment

        targets = np.linspace(0.0, total_len, n_steps, dtype=np.float32)
        x_new = np.interp(targets, cumdist, pts[:, 0])
        y_new = np.interp(targets, cumdist, pts[:, 1])
        pen = np.full((n_steps, 1), segment[0, 2], dtype=np.float32)
        out = np.concatenate([x_new[:, None], y_new[:, None], pen], axis=1)
        out[0] = segment[0]
        out[-1] = segment[-1]
        return out.astype(np.float32)

    def resolve_data_path(self, raw_path: str) -> str:
        raw_path = str(raw_path).strip()
        if os.path.isabs(raw_path):
            return raw_path
        for cand in [
            os.path.join(self.base_path, raw_path),
            os.path.join(self.meta_root, raw_path),
            os.path.join(os.path.dirname(self.meta_root), raw_path),
        ]:
            if os.path.exists(cand):
                return cand
        return os.path.join(self.base_path, raw_path)

    def inspect_sample(self, idx: int):
        row = self.meta_df.iloc[idx]
        csv_path = self.resolve_data_path(row["csv"])
        img_path = self.resolve_data_path(row["img"])
        raw = self.load_trajectory_raw(csv_path)
        if self.distance_thin:
            raw = self.filter_trajectory(raw)
        _, image_meta = self.preprocess_image(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
        processed, traj_meta = self.process_trajectory(raw, image_meta)
        return row, raw, processed, {"image_meta": image_meta, "traj_meta": traj_meta}


def collate_trajectory_batch(batch):
    images, trajectories, lengths, metas = zip(*batch)
    images = torch.stack(images, dim=0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    max_len = int(lengths.max().item()) if len(lengths) > 0 else 0
    padded = torch.zeros(len(trajectories), max_len, 3, dtype=torch.float32)
    for i, traj in enumerate(trajectories):
        padded[i, :traj.size(0)] = traj
    return images, padded, lengths, list(metas)


if __name__ == "__main__":
    ds = TrajectoryDataset(
        meta_csv_path=config.MASTER_RAW_CSV,
        img_size=(64, 1600),
        normalize_coords=True,
        normalization_mode="minmax",
        distance_thin=True,
        pen_down_min_dist_px=45.0,
        pen_up_min_dist_px=100.0,
    )
    img, traj, length, meta = ds[0]
    print(f"Image shape : {img.shape}")
    print(f"Traj shape  : {traj.shape}  length={length}")
    print(f"Coord space : {meta['traj_meta']['coord_space']}")
    print(f"Traj X range: [{traj[:length,0].min():.4f}, {traj[:length,0].max():.4f}]")
    print(f"Traj Y range: [{traj[:length,1].min():.4f}, {traj[:length,1].max():.4f}]")
