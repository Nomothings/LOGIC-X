#!/usr/bin/env bash
# =============================================================================
# T-MCTS SFT — full fine-tune with verl
#
# Usage:
#   MODEL=LOGIC-X-8B bash run_sft.sh
#   MODEL=LOGIC-X-8B DATA=sft_atomic_tasks bash run_sft.sh   # 14B curriculum
#
# Environment:
#   MODEL        model family: LOGIC-X-8B | LOGIC-X-14B          (required)
#   MODEL_PATH   path to the base checkpoint (Qwen3-8B / Qwen3-14B)
#   DATA         training file stem under train_data/$MODEL/     (default: sft_trajectories)
#   GPUS         comma-separated GPU ids                          (default: 0,1,2,3,4,5,6,7)
#
# Prerequisites:
#   pip install verl torch transformers datasets pandas
#   python convert_to_verl_parquet.py --input  ../train_data/$MODEL/$DATA.jsonl \
#                                      --output ../train_data/$MODEL/$DATA.parquet
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${MODEL:?Set MODEL=LOGIC-X-8B or MODEL=LOGIC-X-14B}"
DATA="${DATA:-sft_trajectories}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the base checkpoint directory}"
CONFIG="$ROOT/sft/verl_sft_${MODEL}.yaml"
PARQUET="$ROOT/train_data/$MODEL/${DATA}.parquet"

echo "=============================================="
echo " T-MCTS SFT"
echo " Model family: $MODEL"
echo " Base checkpoint: $MODEL_PATH"
echo " Data: $PARQUET"
echo " GPUs: $GPUS ($NUM_GPUS cards)"
echo "=============================================="

# ---- Verify data ----
if [ ! -f "$PARQUET" ]; then
    echo "Parquet not found — converting from jsonl..."
    python3 "$ROOT/sft/convert_to_verl_parquet.py" \
        --input "$ROOT/train_data/$MODEL/${DATA}.jsonl" \
        --output "$PARQUET"
fi
echo "  $(ls -lh "$PARQUET" | awk '{print $5}') -> $PARQUET"

# ---- Verify model ----
[ ! -d "$MODEL_PATH" ] && echo "ERROR: $MODEL_PATH not found" && exit 1

# ---- Launch training ----
echo "Launching verl SFT..."
echo "  Config: $CONFIG"

mkdir -p "$ROOT/train/logs" "$ROOT/train/saves"

CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node "$NUM_GPUS" \
    -m verl.trainer.sft_trainer \
    --config-path "$ROOT/sft" \
    --config-name "verl_sft_${MODEL}" \
    data.train_files="$PARQUET" \
    model.path="$MODEL_PATH" \
    trainer.default_local_dir="$ROOT/train/saves/${MODEL}" \
    2>&1 | tee "$ROOT/train/logs/sft_${MODEL}.log"
