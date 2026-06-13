# Synthetic Stroke Trajectory Reconstruction for Online Urdu Handwriting

Self-contained copy of the source files and trained checkpoints that make up the
thesis pipeline: *Synthetic Stroke Trajectory Reconstruction for Online Urdu
Handwriting under Low-Resource Constraints* (Momna Waheed, MS DS, SEECS-NUST).

The system converts an **offline image** of an Urdu Nastaleeq line into a
**character transcription** by first reconstructing an online-style pen
trajectory, then decoding it with a sequence recognizer.

```
Offline / rendered image
        │
        ▼
[Stage 1]  U-Net stroke segmentation  ──►  3-class mask (bg / interior / boundary)
        │
        ▼
[Stage 2]  Trajectory recovery
           boundary cut → connected components → skeletonize → walk → resample
        │            ▲
        │            │ stroke order from
        │            │
[Stage 3]  Stroke order predictor (shape embedding + spatial features → MLP)
        │
        ▼            ordered (T × [x, y, pen_state]) sequence
[Stage 4]  BiLSTM-CTC recognizer  ──►  Urdu text
        ▲
        │ trained via
[Stage 5]  Two-stage recipe: synthetic UPTI pretrain → real OUHD-L fine-tune
```

**Headline result:** 21.3% test CER on the OUHD-L held-out test split (240 lines),
a 24% relative improvement over the 27.9% real-only training baseline.

---

## Folder layout

```
pipeline_export/
├── stage1_unet_segmentation/    U-Net architecture, training, inference, eval
├── stage2_trajectory_recovery/  mask → ordered point sequence
├── stage3_order_predictor/      shape embedding + learned stroke ordering
├── stage4_recognizer/           BiLSTM-CTC model + the PathBExtractor orchestrator
├── stage5_two_stage_training/   synthetic generation, pretrain, fine-tune
├── coverage_analysis/           character + ligature coverage (Naeem et al.)
├── checkpoints/                 all trained weights + k-means vocab
└── common/                      shared dependencies imported across stages
```

> **Running the code.** These files were exported from a larger research repo
> (`phaseN/` directories). They have been made runnable: dataset and checkpoint
> paths are centralized in `config.py`, and `setup_env.sh` puts every stage
> directory on `PYTHONPATH`. **See [REPRODUCE.md](REPRODUCE.md) for the exact
> commands** — the headline 21.3% CER reproduces in seconds via
> `verify_results.py`. The orchestrator that wires every stage together is
> `stage4_recognizer/recog_datasets.py` (class `PathBExtractor`).

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.9
pip install -r requirements.txt

export STROKE_DATA_ROOT=/path/to/your/data           # where the corpora live
source setup_env.sh                                  # PYTHONPATH + data root
```

Dependencies are pinned in `requirements.txt` to the versions used to produce
the thesis results. No CUDA is required — the pipeline runs on CPU or Apple
MPS. The trained weights needed to reproduce every reported number are in
`checkpoints/`; the external corpora (OUHD-L, UPTI-2.0, UHWR) are not included
(see "External data" below).

---

## Stage 1 — U-Net Stroke Segmentation (thesis §3.3)

Small encoder–decoder (~1.9M params) producing a per-pixel 3-class mask
(background / stroke interior / inter-stroke boundary) on a 1×64×1600 line image.
Trained on rasterized OUHD-L trajectories with class-weighted cross-entropy
`[0.1, 1.0, 5.0]`, in two phases (broad 3–7px, then thinner 2–4px warm-start).

| File | Role |
|------|------|
| `unet.py` | StrokeUNet architecture (DoubleConv, channels 16→32→64→128→256) |
| `stroke_seg_dataset.py` | Rasterize trajectories → image + 3-class GT; augmentations |
| `train_unet.py` | Phase-1 training (broad thickness) |
| `train_unet_thinner.py` | Phase-2 fine-tune (2–4px, warm-started) |
| `train_unet_focal.py`, `train_unet_focal_dilated.py`, `train_unet_distance_boundary.py` | Loss-variant ablations (Tables 4.2 / 4.3) |
| `infer_unet.py` | softmax → boundary/ink masks → instance extraction |
| `clean_preprocess.py`, `test_uhwr.py` | Preprocessing for real scanned input |
| `eval_unet_metrics.py`, `eval_unet_compare_all.py` | Per-pixel + per-instance metrics, 4-variant comparison |

Checkpoints (two phases — both included):
- `checkpoints/unet_phase1_broad.pt` — phase-1 broad 3–7px (lr 1e-3, 10 epochs, val loss 0.113)
- `checkpoints/unet_phase2_thinner.pt` — **phase-2 thickness-matched 2–4px** (warm-started from phase-1, lr 2e-4, val loss 0.069). This is the **deployment U-Net** credited in Table 4.3 (Inst IoU 0.879 at 2–3px). Use this one for inference / reproducing the thickness-matched numbers.

## Stage 2 — Trajectory Recovery (§3.4)

Turns the static 3-class mask into a single ordered point sequence.

| File | Role |
|------|------|
| `full_pipeline_unet_uhwr.py` | Boundary cut + connected-component labeling → stroke components |
| `full_pipeline_unet_uhwr_skeleton.py` | Single-pass skeleton-point ordering (`order_skeleton_points`) used by the multi-pass walk |
| `hybrid_inference.py` | Component → patch rendering (`render_component_to_patch`) |
| `template_stitcher.py` | Skeletonization (`extract_skeleton`) |
| `mask_to_training_patch.py` | Multi-pass skeleton walk + smoothing (`order_skeleton_multi_pass`, `_smooth_polyline`) |
| `calibrate_step_size.py` | Arc-length resampling / step-size calibration |

## Stage 3 — Stroke Order Predictor (§3.5)

Self-supervised shape embedding + spatial features feed an MLP (262→128→64→1)
that scores each stroke's normalized writer position; strokes are sorted by score.

| File | Role |
|------|------|
| `build_vocab.py` | k-means over resampled trajectory shapes → 100 clusters (pseudo-labels) |
| `cnn_classifier.py` | CNN trained on rasterized strokes to predict cluster ID; 256-d hidden = shape embedding |
| `train_local_shape_clean.py` | Trains the shape CNN |
| `knn_retrieval.py` | `make_embedding_extractor` (strips final FC → 256-d embedding) |
| `order_predictor.py` | The ordering MLP + 6-d spatial feature extraction |

Checkpoints: `checkpoints/shape_cnn_best_noDIAC.pt`, `checkpoints/order_predictor.pt`,
`checkpoints/knn_index.npz`, `checkpoints/vocab_out_k100/`

## Stage 4 — Recognizer (§3.6)

BiLSTM-CTC (~613k params, vocab V=74). Input projection 3→64 (Linear+LayerNorm+ReLU)
→ 2-layer BiLSTM (hidden 128/dir, dropout 0.2) → Linear 256→74. Greedy CTC decode,
no language model.

| File | Role |
|------|------|
| `recog_datasets.py` | **Orchestrator** — `PathBExtractor` wires Stages 1–3 to produce trajectories; dataset classes |
| `recognizer.py` | BiLSTM-CTC model |
| `vocab_loader.py`, `traj_tokenizer.py` | Vocabulary + trajectory tokenization |
| `eval_recognizer.py` | CER evaluation |
| `train_recognizer.py` | Real-only baseline training |

## Stage 5 — Two-Stage Training (§3.7)

Pretrain on 30,000 synthetic UPTI-derived trajectories, then fine-tune on the
1,923-line OUHD-L real training split (deterministic 80/10/10 split, seed 0).

| File | Role |
|------|------|
| `generate_synthetic.py` | Run 30k UPTI line images through Stages 1–3 → synthetic trajectories |
| `pretrain_synthetic.py` | Stage 1 training (Adam lr 1e-3, batch 24, 25 epochs, cap 750) |
| `finetune_real_3way.py` | Stage 2 fine-tune (Adam lr 3e-4, batch 16, 35 epochs, cap 1024, best @ epoch 34) |
| `pretrain_then_finetune.py` | Convenience driver for both stages |
| `make_splits.py` | Deterministic 80/10/10 split |

Checkpoints:
- `checkpoints/recognizer_pretrained_synthetic.pt` — after Stage 1
- `checkpoints/recognizer_finetuned_3way.pt` — final model (**test CER 0.213 / val 0.210**)
- `checkpoints/recognizer_realonly_baseline.pt` — real-only baseline (val CER 0.279)

## Coverage Analysis (§4.3)

| File | Role |
|------|------|
| `analyze_char_coverage.py` | Per-character occurrence coverage across splits |
| `analyze_ligature_coverage.py` | Ligature coverage (Naeem et al. non-connector splitting) |

---

## Common (shared across stages)

Modules imported by name from more than one stage.

| File | Role |
|------|------|
| `trajectory_dataset.py` | Loads OUHD-L `(image, trajectory)` samples |
| `stroke_sequence_utils.py` | `extract_pen_down_strokes` and trajectory helpers |
| `visualize_sample.py` | `colorize_instance` / `colorize_semantic` mask visualizers |
| `clean_stroke_dataset.py` | `render_clean_stroke`, `CleanStrokeDataset` — per-stroke rendering used by Stage 2 recovery and Stage 3 shape training |
| `stroke_detector.py` | `binarize_image`, `detect_strokes` — binarization used by Stage 2/3 |

---

## Checkpoint quick reference

| Checkpoint | Stage | Key metric |
|------------|-------|-----------|
| `unet_phase1_broad.pt` | 1 | per-instance IoU 0.750 (broad 3–7px) |
| `unet_phase2_thinner.pt` | 1 | **per-instance IoU 0.879 (thickness-matched 2–4px, deployment)** |
| `shape_cnn_best_noDIAC.pt` | 3 | shape embedding source |
| `order_predictor.pt` | 3 | Pearson r ≈ 0.97 (writer order) |
| `recognizer_pretrained_synthetic.pt` | 5 | zero-shot real CER 78.2/78.3% |
| `recognizer_finetuned_3way.pt` | 5 | **test CER 21.3%** |
| `recognizer_realonly_baseline.pt` | 5 | val CER 27.9% |

## External data (not included)

- **OUHD-L** — real online Urdu corpus (Wacom IoT paper device), 2,403 lines;
  `master_raw.csv` referenced by training scripts.
- **UPTI-2.0** — printed Urdu corpus used for synthetic pretraining.
- **UHWR** — real scanned handwriting, used only for qualitative generalization figures.
