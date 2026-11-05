#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${GROOT_DATA_ROOT:-/workspace/data}"
cd "$repo_dir"
mkdir -p "$data_root/logs" "$data_root/outputs"
# 诊断日志需显式通过 DIAG_TORCH_LOGS 传入（TORCH_LOGS 本身仍会被清理，避免污染测速）
diag_torch_logs="${DIAG_TORCH_LOGS:-}"
unset TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR TORCH_LOGS TORCH_TRACE CUDA_LAUNCH_BLOCKING HIP_LAUNCH_BLOCKING
if [ -n "$diag_torch_logs" ]; then export TORCH_LOGS="$diag_torch_logs"; fi

export USE_WANDB=0
export VLM_MODEL_PATH="$data_root/checkpoints/nvidia/Cosmos-Reason2-2B"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset VIDEO_DECODER_BACKEND
export NUM_GPUS=8
export MAX_STEPS=40
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-960}
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
# 编译范围可由调用方环境覆盖（如 GR00T_COMPILE_DIT=model bash start.sh），未设置则默认关闭
export GR00T_COMPILE_DIT=${GR00T_COMPILE_DIT:-off}   # off：关闭；model：整体；blocks：逐块
export GR00T_COMPILE_VL=${GR00T_COMPILE_VL:-0}       # 0：关闭；1：编译 VL 自注意力块
export GR00T_COMPILE_TEXT=${GR00T_COMPILE_TEXT:-off} # off：关闭；layers：文本层；forwards：层 forward
if [ -z "${GR00T_COMPILE_MODE:-}" ]; then unset GR00T_COMPILE_MODE; fi

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
