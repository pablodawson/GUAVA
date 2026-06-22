#!/bin/bash
set -e
# =================================================================
# GUAVA optimization-only personalization (no feed-forward inferer).
#   bash scripts/optimize.sh <id> <video.mp4> [gpu] [--from <step>]
#
#   Steps:
#     1  track     EHM-Tracker full-body tracking of the whole video
#     2  optimize  explicit mesh-bound gaussians, optimized over all frames
#
#   --from <step>  skip steps before <step> (number or name)
#   Env: TRAIN_REFINER=1 also finetunes the neural render refiner.
# =================================================================
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <id> <video.mp4> [gpu] [--from <step>]"
    exit 1
fi
ID=$1
VIDEO=$2
GPU=${3:-0}

FROM_STEP=1
args=("$@")
for i in "${!args[@]}"; do
    if [ "${args[$i]}" = "--from" ]; then
        FROM_STEP="${args[$((i+1))]}"
    fi
done
case "$FROM_STEP" in
    track)    FROM_STEP=1 ;;
    optimize) FROM_STEP=2 ;;
esac
run_step() { [ "$1" -ge "$FROM_STEP" ]; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=$GPU

BASE_MODEL="assets/GUAVA"
EXP="outputs/personalize/$ID"
VIDEO_STEM="$(basename "$VIDEO" | sed 's/\.[^.]*$//')"
TRACKED="outputs/personalize/tracked/$VIDEO_STEM"
OPT="$EXP/opt"

# --- 1. track the full-body video ---
if run_step 1; then
    echo "[1/2] tracking $VIDEO"
    if [ -f "$TRACKED/optim_tracking_ehm.pkl" ]; then
        echo "    already tracked, skipping"
    else
        ( cd EHM-Tracker && python tracking_video.py -i "$VIDEO" \
            -o "$ROOT/outputs/personalize/tracked" --check_hand_score 0.0 -p 0,1 -n 1 -v 0 )
    fi
fi

# --- 2. optimize explicit gaussians over all frames ---
if run_step 2; then
    echo "[2/2] optimizing gaussians"
    REFINER_FLAG=""
    [ "${TRAIN_REFINER:-0}" = "1" ] && REFINER_FLAG="--train_refiner"
    python personalizer/optimize.py \
        --data_path "$TRACKED" --base_model "$BASE_MODEL" --exp_dir "$OPT" --devices 0 $REFINER_FLAG
fi

echo "Done. Avatar at $OPT (checkpoints/, canonical.ply, vis_results/)"
