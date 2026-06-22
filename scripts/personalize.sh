#!/bin/bash
set -e
# =================================================================
# GUAVA personalization pipeline (full-body analog of ELITE).
#   bash scripts/personalize.sh <id> <video.mp4> [gpu] [--from <step>]
#
#   Steps:
#     1  track            EHM-Tracker full-body tracking
#     2  stage1           stage-1 finetune on real frames
#     3  synth            synthetic view generation
#     4  difix            DIFIX 2D refinement of synthetic views
#     5  stage2           stage-2 joint finetune (real + refined synth)
#     6  export           export browser bundle
#
#   --from <step>  skip all steps before <step> (number or name)
# =================================================================
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <id> <video.mp4> [gpu] [--from <step>]"
    exit 1
fi
ID=$1
VIDEO=$2
GPU=${3:-0}

# Parse --from anywhere in the remaining args
FROM_STEP=1
args=("$@")
for i in "${!args[@]}"; do
    if [ "${args[$i]}" = "--from" ]; then
        FROM_STEP="${args[$((i+1))]}"
    fi
done

# Accept step names as well as numbers
case "$FROM_STEP" in
    track)  FROM_STEP=1 ;;
    stage1) FROM_STEP=2 ;;
    synth)  FROM_STEP=3 ;;
    difix)  FROM_STEP=4 ;;
    stage2) FROM_STEP=5 ;;
    export) FROM_STEP=6 ;;
esac

run_step() { [ "$1" -ge "$FROM_STEP" ]; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=$GPU

BASE_MODEL="assets/GUAVA"
ELITE_ROOT="/workspace1/pdawson/ELITE"
ARKIT_BS="assets/FLAME/flame_arkit_bs.npy"     # optional
SYNTH_MODE="${SYNTH_MODE:-both}"                # expr | views | both

EXP="outputs/personalize/$ID"
# EHM-Tracker names the output subdir after the video filename stem
VIDEO_STEM="$(basename "$VIDEO" | sed 's/\.[^.]*$//')"
TRACKED="outputs/personalize/tracked/$VIDEO_STEM"
ST1="$EXP/st1"; SYNTH="$EXP/synth"; ST2="$EXP/st2"; BUNDLE="$EXP/bundle"

# --- 1. track the full-body video ---
if run_step 1; then
    echo "[1/6] tracking $VIDEO"
    if [ -f "$TRACKED/optim_tracking_ehm.pkl" ]; then
        echo "    already tracked, skipping"
    else
        ( cd EHM-Tracker && python tracking_video.py -i "$VIDEO" \
            -o "$ROOT/outputs/personalize/tracked" --check_hand_score 0.0 -p 0,1 -n 1 -v 0 )
    fi
fi

# --- 2. stage-1 personalization on real frames ---
if run_step 2; then
    echo "[2/6] stage-1 finetune"
    python personalizer/personalize.py --stage 1 \
        --data_path "$TRACKED" --base_model "$BASE_MODEL" --exp_dir "$ST1" --devices 0
fi

# --- 3. synthetic views (random expressions + novel cameras) ---
if run_step 3; then
    echo "[3/6] synthetic view generation"
    python personalizer/infer_synth.py \
        --model_dir "$ST1" --out_dir "$SYNTH" --data_path "$TRACKED" \
        --mode "$SYNTH_MODE" --num_expr 64 --num_views 64 --devices 0
fi

# --- 4. DIFIX refine the synthetic renders ---
if run_step 4; then
    echo "[4/6] DIFIX refine"
    python personalizer/refine_synth.py --synth_dir "$SYNTH" --elite_root "$ELITE_ROOT"
fi

# --- 5. stage-2 joint finetune (real + refined synthetic), from base prior ---
if run_step 5; then
    echo "[5/6] stage-2 finetune"
    python personalizer/personalize.py --stage 2 \
        --data_path "$TRACKED" --base_model "$BASE_MODEL" --exp_dir "$ST2" \
        --synth_dir "$SYNTH" --devices 0
fi

# --- 6. export browser bundle from the stage-2 avatar ---
if run_step 6; then
    echo "[6/6] export bundle"
    ARKIT_FLAG=""
    [ -f "$ARKIT_BS" ] && ARKIT_FLAG="--arkit_bs $ARKIT_BS"
    python export_browser_bundle.py \
        --checkpoint "$ST2" --source "$TRACKED" --out "$BUNDLE" $ARKIT_FLAG
fi

echo "Done. Bundle at $BUNDLE"
