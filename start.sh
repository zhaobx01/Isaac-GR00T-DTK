#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${GROOT_DATA_ROOT:-/workspace/data}"
cd "$repo_dir"
mkdir -p "$data_root/logs" "$data_root/outputs"
unset TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR

export USE_WANDB=0 
export VLM_MODEL_PATH="$data_root/checkpoints/nvidia/Cosmos-Reason2-2B"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset VIDEO_DECODER_BACKEND
export NUM_GPUS=8 
export MAX_STEPS=100 
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-768}
export SAVE_STEPS=1000 
export MASTER_PORT=${MASTER_PORT:-29500}

machine=bw1000
date_tag=$(date +%Y%m%d)
time_tag=$(date +%H%M%S)
comm_tag=ov${GR00T_ZERO2_OVERLAP:-0}_rs${GR00T_ZERO2_REDUCE_SCATTER:-0}_ag${GR00T_ZERO2_ALLGATHER_PIPELINE:-0}_pt${GR00T_ZERO2_PACK_TIMING:-0}
label=${machine}_gbs${GLOBAL_BATCH_SIZE}_w8_h264_s100_${comm_tag}_${date_tag}_${time_tag}

# Measurement run by default: per-step metrics on, no final model save.
# Profiling: PROFILE=1 adds the profiler flags; extra launch_finetune args can be appended after start.sh.
launch_args=(--log-step-metrics --save-strategy no --no-save-final-model)
if [ "${PROFILE:-0}" = "1" ]; then
    launch_args+=(--enable-profiling --no-profiler-with-stack
        --profiler-skip-first "${PROFILER_SKIP_FIRST:-20}" --profiler-wait "${PROFILER_WAIT:-1}"
        --profiler-warmup "${PROFILER_WARMUP:-5}" --profiler-active "${PROFILER_ACTIVE:-3}"
        --profiler-repeat 1)
fi

bash examples/finetune.sh \
    --base-model-path "$data_root/checkpoints/GR00T-N1.7-3B-local-cosmos" \
    --dataset-path "$data_root/libero_10_no_noops_1.0.0_lerobot_h264" \
    --embodiment-tag LIBERO_PANDA \
    --output-dir "$data_root/outputs/$label" \
    --state-dropout-prob 0.2 \
    --save-only-model \
    -- "${launch_args[@]}" "$@" \
    2>&1 | tee "$data_root/logs/${label}.log"
