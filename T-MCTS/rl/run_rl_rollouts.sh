#!/usr/bin/env bash
# =============================================================================
# T-MCTS RL data loop — rejection-sampling rollouts with the current policy
#
# Pipeline:
#   1. expand_rollouts.py        duplicate each question N times
#   2. run_inference.py --local  sample N rollouts with the SFT checkpoint
#   3. filter_rollout_traces.py  keep questions solved in 1..K of N rollouts
#
# The kept questions (with their correct rollouts) form the next round of
# optimization data; retrain with sft/run_sft.sh and repeat.
#
# Usage:
#   MODEL=LOGIC-X-8B CKPT=train/saves/LOGIC-X-8B bash run_rl_rollouts.sh
#
# Environment:
#   MODEL       model family: LOGIC-X-8B | LOGIC-X-14B           (required)
#   CKPT        checkpoint directory to sample from              (required)
#   GPUS        GPUs for vLLM                                    (default: 0,1)
#   PORT        vLLM port                                        (default: 8010)
#   N_ROLLOUTS  rollouts per question                            (default: 5)
#   OUT_ROOT    output root                                      (default: outputs/rl)
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

MODEL="${MODEL:?Set MODEL=LOGIC-X-8B or MODEL=LOGIC-X-14B}"
CKPT="${CKPT:?Set CKPT to the checkpoint directory}"
GPUS="${GPUS:-0,1}"
PORT="${PORT:-8010}"
N_ROLLOUTS="${N_ROLLOUTS:-5}"
OUT_ROOT="${OUT_ROOT:-outputs/rl}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

OUT_DIR="$OUT_ROOT/$MODEL"

mkdir -p "$OUT_DIR"

echo "=============================================="
echo " T-MCTS RL rollouts"
echo " Model family: $MODEL"
echo " Checkpoint:   $CKPT"
echo " N rollouts:   $N_ROLLOUTS"
echo "=============================================="

# ---- 1/3 Expand ----
echo ">>> [1/3] Expanding prompts x$N_ROLLOUTS ..."
python3 rl/expand_rollouts.py \
    --input "train_data/$MODEL/rl_prompts.jsonl" \
    --output "$OUT_DIR/input_${N_ROLLOUTS}x.jsonl" \
    --n-rollouts "$N_ROLLOUTS"

# ---- 2/3 Sample rollouts ----
echo ">>> [2/3] Sampling rollouts with vLLM ..."
python3 ../LogicEval-X/run_inference.py \
    --local "$CKPT" \
    --input "$OUT_DIR/input_${N_ROLLOUTS}x.jsonl" \
    --output "$OUT_DIR/eval" \
    --gpus "$GPUS" --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --new

# ---- 3/3 Filter ----
echo ">>> [3/3] Filtering rollouts ..."
python3 rl/filter_rollout_traces.py \
    --eval-dir "$OUT_DIR/eval" \
    --source "train_data/$MODEL/rl_prompts.jsonl" \
    --model-name "$MODEL" \
    --n-rollouts "$N_ROLLOUTS" \
    --output-dir "$OUT_DIR"

echo "=============================================="
echo " Done. Kept questions: $OUT_DIR/rl_keep.jsonl"
echo "=============================================="
