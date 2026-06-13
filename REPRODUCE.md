# Reproducing the results

This guide is for whoever verifies the code. It assumes the external corpora
are available (they are not committed — see "Data" below). The headline result
(**21.3% test CER**) can be reproduced in seconds with the shipped checkpoint.

## 1. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate    # Python 3.9
pip install -r requirements.txt
```

No CUDA required — runs on CPU or Apple MPS.

## 2. Configure paths (once per shell)

Every script reads its paths from `config.py`, which are driven by two things:

- **`STROKE_DATA_ROOT`** — where the external corpora live (defaults to `~/Downloads`).
- **`PYTHONPATH`** — all stage directories, so the flat cross-module imports resolve.

Both are set by sourcing the helper:

```bash
export STROKE_DATA_ROOT=/path/to/your/data   # only if not ~/Downloads
source setup_env.sh
```

### Expected data layout under `STROKE_DATA_ROOT`

```
<STROKE_DATA_ROOT>/
├── Dataset/
│   ├── main_all.csv          OUHD-L online corpus (2,403 lines)
│   └── master_raw.csv        OUHD-L raw trajectories
├── PaperReplication/Dataset/UHWR/Dataset/images/   UHWR scans
├── images_upti2_2.zip        UPTI-2.0 images
└── groundtruth_upti2.zip     UPTI-2.0 transcriptions
```

Only `Dataset/main_all.csv` is needed for the headline check below.

## 3. Reproduce the headline result (fast)

```bash
python3 stage5_two_stage_training/make_splits.py   # deterministic 80/10/10, seed 0
python3 verify_results.py                           # loads shipped checkpoint, evals test split
```

Expected output:

```
[verify] vocab size (with blank) = 74
>>> TEST CER = 0.2130   (thesis headline: 0.213)
```

`make_splits.py` writes `stage5_two_stage_training/main_all_splits.json`
(train=1923 / val=240 / test=240). `verify_results.py` loads
`checkpoints/recognizer_finetuned_3way.pt` and evaluates the held-out test
split using the ground-truth online trajectories — the same final evaluation
`finetune_real_3way.py` performs after training.

## 4. Reproduce the full training pipeline (optional, slow)

Requires the full corpora and regenerating the synthetic cache.

```bash
# Stage 5 — synthetic pretrain then real fine-tune
python3 stage5_two_stage_training/generate_synthetic.py     # UPTI/UHWR -> synthetic_cache.npz
python3 stage5_two_stage_training/pretrain_synthetic.py     # -> recognizer_pretrained
python3 stage5_two_stage_training/finetune_real_3way.py     # -> recognizer_finetuned_3way.pt
```

Stage 1–3 checkpoints used by the trajectory pipeline (`PathBExtractor` in
`stage4_recognizer/recog_datasets.py`) are all shipped under `checkpoints/`.

## Notes for the verifier (what changed when the repo was cleaned)

The files were exported from a larger research repo. To make them portable and
runnable, the following non-algorithmic changes were made — no model or training
logic was touched:

1. **Absolute paths → `config.py`.** Every hard-coded `/Users/mac/Downloads/...`
   dataset path now reads from `config.py` (driven by `STROKE_DATA_ROOT`).
2. **Checkpoint paths → `checkpoints/`.** Scripts referenced the old `phaseN/`
   layout (e.g. `phase7/local_shape_clean_out/model_best_noDIAC.pt`). These now
   point at the shipped `checkpoints/` via `config.py`. Mapping:
   - `phase10/unet_out/unet_best.pt` (inference) → `unet_phase2_thinner.pt` (deployment U-Net)
   - U-Net warm-start in ablation trainers → `unet_phase1_broad.pt`
   - `phase7/.../model_best_noDIAC.pt` → `shape_cnn_best_noDIAC.pt`
   - `phase9/knn_index.npz` → `knn_index.npz`, `phase9/order_predictor.pt` → `order_predictor.pt`
   - `phase5b/vocab_out_k100` → `vocab_out_k100/`
   - `recognizer_pretrained_upti20k_f750.pt` → `recognizer_pretrained_synthetic.pt`
3. **`vocab_minimal.json` vendored in.** The recognizer's 74-token char vocab
   was not in the original export; it is now at `checkpoints/vocab_minimal.json`.
4. **Imports.** Modules import each other by flat name; `setup_env.sh` puts all
   stage dirs on `PYTHONPATH` so they resolve. (Dead `sys.path.insert(...phaseN)`
   lines from the old layout are harmless and were left in place.)
