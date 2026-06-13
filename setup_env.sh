# Source this file before running any pipeline script:
#
#     source setup_env.sh
#
# It puts every stage directory on PYTHONPATH (the modules import each other by
# flat name, e.g. `from unet import StrokeUNet`) and sets the data root used by
# config.py. Adjust STROKE_DATA_ROOT to wherever the corpora live on your machine.

# Directory this script lives in (the repo root), regardless of where it's sourced from.
if [ -n "$BASH_SOURCE" ]; then
    REPO_ROOT="$(cd "$(dirname "$BASH_SOURCE")" && pwd)"
elif [ -n "$ZSH_VERSION" ]; then
    REPO_ROOT="$(cd "$(dirname "${(%):-%x}")" && pwd)"
else
    REPO_ROOT="$(pwd)"
fi

# Root of the external corpora (OUHD-L, UPTI, UHWR). Override before sourcing if needed.
export STROKE_DATA_ROOT="${STROKE_DATA_ROOT:-$HOME/Downloads}"

# All importable source directories.
export PYTHONPATH="\
$REPO_ROOT:\
$REPO_ROOT/common:\
$REPO_ROOT/stage1_unet_segmentation:\
$REPO_ROOT/stage2_trajectory_recovery:\
$REPO_ROOT/stage3_order_predictor:\
$REPO_ROOT/stage4_recognizer:\
$REPO_ROOT/stage5_two_stage_training:\
$REPO_ROOT/coverage_analysis:\
${PYTHONPATH:-}"

echo "[setup_env] REPO_ROOT=$REPO_ROOT"
echo "[setup_env] STROKE_DATA_ROOT=$STROKE_DATA_ROOT"
