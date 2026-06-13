"""
Central path configuration for the Synthetic Stroke Trajectory pipeline.

All scripts read their dataset locations from here instead of hard-coding an
absolute path, so the repo is portable across machines.

Set the data root once via an environment variable before running anything:

    export STROKE_DATA_ROOT=/path/to/your/data        # bash / zsh

If STROKE_DATA_ROOT is unset, it falls back to ~/Downloads (the layout the
thesis results were produced with).

Expected directory layout under STROKE_DATA_ROOT:

    <STROKE_DATA_ROOT>/
    ├── Dataset/
    │   ├── main_all.csv          OUHD-L online corpus (UCOM main_all)
    │   └── master_raw.csv        OUHD-L raw trajectories
    ├── PaperReplication/Dataset/UHWR/
    │   └── Dataset/images/        UHWR scanned line images
    ├── images_upti2_2.zip         UPTI-2.0 line images
    └── groundtruth_upti2.zip      UPTI-2.0 ground-truth transcriptions
"""

from __future__ import annotations

import os
from pathlib import Path

# Root of all external corpora. Override with STROKE_DATA_ROOT.
DATA_ROOT = Path(os.environ.get("STROKE_DATA_ROOT", str(Path.home() / "Downloads")))

# OUHD-L online dataset CSVs
DATASET_DIR = DATA_ROOT / "Dataset"
MAIN_ALL_CSV = str(DATASET_DIR / "main_all.csv")
MASTER_RAW_CSV = str(DATASET_DIR / "master_raw.csv")

# UHWR scanned handwriting
UHWR_ROOT = str(DATA_ROOT / "PaperReplication" / "Dataset" / "UHWR")
UHWR_IMAGES_DIR = str(Path(UHWR_ROOT) / "Dataset" / "images")

# UPTI-2.0 printed corpus
UPTI_IMG_ZIP = str(DATA_ROOT / "images_upti2_2.zip")
UPTI_LBL_ZIP = str(DATA_ROOT / "groundtruth_upti2.zip")

# Sample line images used by the figure scripts in figures/
TEST_IMAGE_DIR = str(DATA_ROOT / "TestProject" / "Dataset" / "images")

# Convenience string form of the root (used where a base_path string is expected)
DATA_ROOT_STR = str(DATA_ROOT)


# ---------------------------------------------------------------------------
# Trained weights / vocab. These ship with the repo under checkpoints/, so they
# resolve relative to this file (not to STROKE_DATA_ROOT).
# ---------------------------------------------------------------------------
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"

# Sample line images bundled with the repo so the figure scripts run out of the
# box (no external dataset needed). For the full corpus, use TEST_IMAGE_DIR.
SAMPLE_IMAGE_DIR = str(Path(__file__).resolve().parent / "figures" / "sample_images")

# Stage 1 U-Net
UNET_PHASE1_BROAD = str(CKPT_DIR / "unet_phase1_broad.pt")          # warm-start / broad
UNET_DEPLOY = str(CKPT_DIR / "unet_phase2_thinner.pt")             # deployment (inference/eval)

# Stage 3 ordering
SHAPE_CNN = str(CKPT_DIR / "shape_cnn_best_noDIAC.pt")
ORDER_PREDICTOR = str(CKPT_DIR / "order_predictor.pt")
KNN_INDEX = str(CKPT_DIR / "knn_index.npz")
VOCAB_DIR_K100 = str(CKPT_DIR / "vocab_out_k100")

# Stage 4/5 recognizer
RECOG_VOCAB_JSON = str(CKPT_DIR / "vocab_minimal.json")           # char vocab (V=74)
RECOG_PRETRAINED = str(CKPT_DIR / "recognizer_pretrained_synthetic.pt")
RECOG_FINETUNED = str(CKPT_DIR / "recognizer_finetuned_3way.pt")  # headline 21.3% CER model
RECOG_REALONLY = str(CKPT_DIR / "recognizer_realonly_baseline.pt")
