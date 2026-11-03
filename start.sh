#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${GROOT_DATA_ROOT:-/workspace/data}"
cd "$repo_dir"
mkdir -p "$data_root/logs" "$data_root/outputs"
unset TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR TORCH_LOGS TORCH_TRACE CUDA_LAUNCH_BLOCKING HIP_LAUNCH_BLOCKING
# 清掉终端里残留的 GR00T_* 开关（保留 GR00T_PROFILER_*），避免污染基线
while IFS= read -r name; do
    case "$name" in GR00T_PROFILER_*) ;; GR00T_*) unset "$name" ;; esac
done < <(compgen -e)

export USE_WANDB=0
export VLM_MODEL_PATH="$data_root/checkpoints/nvidia/Cosmos-Reason2-2B"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset VIDEO_DECODER_BACKEND
export NUM_GPUS=8
export MAX_STEPS=40
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-768}
export DATALOADER_NUM_WORKERS=4
export SAVE_STEPS=1000
export MASTER_PORT=${MASTER_PORT:-29500}

# 基线编译配置
export GR00T_COMPILE_VISION_BLOCKS=1
export GR00T_FLASH_ATTN_DYNAMO_DISABLE=1
export GR00T_VISION_PATCH_EMBED_CHANNELS_LAST=1
export GR00T_COMPILE_SEQUENCE_POLICY=pad-156

# 基线：优化项默认全部关闭
export GR00T_PIXEL_VALUES_BF16=0
export GR00T_APEX_FUSED_ADAM=0
export GR00T_ZERO2_OVERLAP=0
export GR00T_ZERO2_REDUCE_SCATTER=0
export GR00T_ZERO2_ALLGATHER_PIPELINE=0
export GR00T_ZERO2_PACK_TIMING=0
export GR00T_COMPILE_VARIANT=off

machine=bw1000
date_tag=$(date +%Y%m%d)
time_tag=$(date +%H%M%S)
comm_tag=ov${GR00T_ZERO2_OVERLAP}_rs${GR00T_ZERO2_REDUCE_SCATTER}_ag${GR00T_ZERO2_ALLGATHER_PIPELINE}_pt${GR00T_ZERO2_PACK_TIMING}
label=${machine}_gbs${GLOBAL_BATCH_SIZE}_w8_h264_s${MAX_STEPS}_${comm_tag}_${date_tag}_${time_tag}

# Measurement run by default: per-step metrics on, no final model save, no numerics check, no profiling.
# Profiling: PROFILE=1 adds the profiler flags; extra launch_finetune args can be appended after start.sh.
launch_args=(--gradient-accumulation-steps 1 --log-step-metrics --save-strategy no --no-save-final-model --no-check-numerics)
if [ "${PROFILE:-0}" = "1" ]; then
    launch_args+=(--enable-profiling
        --profiler-skip-first "${PROFILER_SKIP_FIRST:-20}" --profiler-wait "${PROFILER_WAIT:-1}"
        --profiler-warmup "${PROFILER_WARMUP:-5}" --profiler-active "${PROFILER_ACTIVE:-3}"
        --profiler-repeat 1)
else
    launch_args+=(--no-enable-profiling)
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
