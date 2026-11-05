#!/usr/bin/env bash
# 后台运行：等待 GPU 空闲 → GEMM 采集（Layer3） → 等待 GPU 空闲 → 8 卡并行选优。
# 断开终端 / ssh 后继续运行。用法：bash run_gemm_capture_tune.sh（始终后台运行，启动后可直接关闭终端）
#
# 可用环境变量覆盖：
#   GEMM_GPUS="0 1 2 3 4 5 6 7"   需要空闲、并用于选优的卡
#   GEMM_CAPTURE_LOG              采集输出（默认 /tmp/train_layer3_combo.log）
#   GEMM_TUNE_INPUT               选优输入日志（默认等于 GEMM_CAPTURE_LOG；
#                                 想用已有的真实日志可设为 /tmp/train_layer3.log）
#   GEMM_IDLE_PCT=5               占用率不超过该值视为空闲
#   GEMM_IDLE_STABLE=3            连续多少次检查都空闲才放行
#   GEMM_IDLE_INTERVAL=60         检查间隔（秒）
#   GEMM_IDLE_TIMEOUT=0           最长等待秒数，0 为一直等
#   GEMM_LOG_ROOT=/tmp/gemm_tune  运行日志 / 锁 / 退出码目录
set -u

script="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
tool="$(dirname "$script")/dcu_gemm_autotune.sh"
gpus="${GEMM_GPUS:-0 1 2 3 4 5 6 7}"
capture_log="${GEMM_CAPTURE_LOG:-/tmp/train_layer3_combo.log}"
tune_input="${GEMM_TUNE_INPUT:-$capture_log}"
idle_pct="${GEMM_IDLE_PCT:-5}"
idle_stable="${GEMM_IDLE_STABLE:-3}"
idle_interval="${GEMM_IDLE_INTERVAL:-60}"
idle_timeout="${GEMM_IDLE_TIMEOUT:-0}"
root="${GEMM_LOG_ROOT:-/tmp/gemm_tune}"
lock="$root/.gemm-tune.lock"

log() { echo "[$(date '+%F %T')] $*"; }

# 打印不空闲的卡（占用率 > 阈值或无法解析）；全部空闲则无输出。
busy_gpus() {
  local use g pct out=
  use="$(rocm-smi --showuse 2>/dev/null || true)"
  for g in $gpus; do
    pct="$(echo "$use" | sed -nE "s/^(HCU|GPU)\[$g\][^:]*: *[A-Za-z]+ use \(%\): *([0-9.]+).*/\2/p" | head -1)"
    if [ -z "$pct" ]; then
      out="$out $g(未知)"
    elif awk "BEGIN{exit !($pct > $idle_pct)}"; then
      out="$out $g(${pct}%)"
    fi
  done
  echo "${out# }"
}

# 等待所有目标卡连续 idle_stable 次检查都空闲。
wait_idle() {   # wait_idle 阶段名
  local stage="$1" ok=0 waited=0 busy
  log "[$stage] 等待卡空闲：$gpus（阈值 ${idle_pct}%，连续 ${idle_stable} 次）"
  while :; do
    busy="$(busy_gpus)"
    if [ -z "$busy" ]; then
      ok=$((ok + 1))
      [ "$ok" -ge "$idle_stable" ] && { log "[$stage] 卡已空闲"; return 0; }
    else
      ok=0
      log "[$stage] 仍有卡忙：$busy"
    fi
    if [ "$idle_timeout" -gt 0 ] && [ "$waited" -ge "$idle_timeout" ]; then
      log "[$stage] 等待空闲超时（${idle_timeout}s）"; return 1
    fi
    sleep "$idle_interval"; waited=$((waited + idle_interval))
  done
}

# ---------------------------------------------------------------- worker ----
if [ "${1:-}" = "--worker" ]; then
  shift
  pidfile="$1"; exitfile="$2"
  child=

  cleanup() { rm -f "$lock"; }
  stop() {
    trap '' TERM INT
    log "收到终止信号，转发给子进程 ${child:-无}"
    if [ -n "$child" ]; then
      kill -TERM "$child" 2>/dev/null
      wait "$child" 2>/dev/null
    fi
    echo 143 > "$exitfile"
    exit 143
  }
  trap '' HUP                             # 终端关闭 / ssh 断开时忽略 SIGHUP
  trap cleanup EXIT
  trap stop TERM INT

  echo $$ > "$pidfile"
  echo $$ > "$lock"

  run_step() {   # run_step 名称 命令...
    local name="$1"; shift
    log "开始${name}"
    "$@" &
    child=$!
    wait "$child"
    local st=$?
    child=
    log "${name}结束，退出码：$st"
    return $st
  }

  # 采集训练脚本放在独立函数里，在子 shell 中执行，不污染 worker 的环境。
  train_cmd() {
    set -e
    cd /workspace/Isaac-GR00T

    # 与 benchmark.py 一样清理继承的 GR00T 开关和编译诊断变量。
    while IFS= read -r name; do unset "$name"; done < <(compgen -e | grep "^GR00T_" || true)
    unset TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR TORCH_LOGS TORCH_TRACE CUDA_LAUNCH_BLOCKING HIP_LAUNCH_BLOCKING

    export NUM_GPUS=8 GLOBAL_BATCH_SIZE=768 MAX_STEPS=5 DATALOADER_NUM_WORKERS=4
    export USE_WANDB=0 MASTER_PORT=29517
    export VLM_MODEL_PATH=/workspace/data/checkpoints/nvidia/Cosmos-Reason2-2B
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    local BASE_MODEL=/workspace/data/checkpoints/GR00T-N1.7-3B-local-cosmos
    local DATASET=/workspace/data/libero_10_no_noops_1.0.0_lerobot_h264

    export GR00T_COMPILE_VISION_BLOCKS=1 GR00T_FLASH_ATTN_DYNAMO_DISABLE=1
    export GR00T_VISION_PATCH_EMBED_CHANNELS_LAST=1 GR00T_COMPILE_SEQUENCE_POLICY=pad-156
    export GR00T_PIXEL_VALUES_BF16=1 GR00T_APEX_FUSED_ADAM=1
    export GR00T_ZERO2_OVERLAP=1 GR00T_ZERO2_REDUCE_SCATTER=1
    export GR00T_ZERO2_ALLGATHER_PIPELINE=1 GR00T_ZERO2_PACK_TIMING=0
    export GR00T_COMPILE_DIT=model GR00T_COMPILE_VL=1 GR00T_COMPILE_TEXT=forwards

    bash examples/finetune.sh \
        --base-model-path "$BASE_MODEL" \
        --dataset-path "$DATASET" \
        --embodiment-tag LIBERO_PANDA \
        --output-dir /tmp/capture_combo_out \
        --state-dropout-prob 0.2 \
        --save-only-model \
        -- \
        --gradient-accumulation-steps 1 \
        --log-step-metrics \
        --save-strategy no \
        --no-save-final-model \
        --no-check-numerics \
        --no-enable-profiling
  }

  cd "$(dirname "$tool")" || exit 1

  wait_idle "采集前" || { echo 1 > "$exitfile"; exit 1; }
  run_step "GEMM 采集" "$tool" capture --layer 3 --out "$capture_log" -- \
    bash -c "$(declare -f train_cmd); train_cmd"
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "$status" > "$exitfile"; exit "$status"
  fi
  [ -s "$capture_log" ] || { log "采集日志为空或不存在：$capture_log"; echo 1 > "$exitfile"; exit 1; }

  [ -s "$tune_input" ] || { log "选优输入不存在或为空：$tune_input"; echo 1 > "$exitfile"; exit 1; }
  # 训练进程刚退出，卡占用可能有尾巴，再确认一次空闲（autotune 自身也会检查）。
  wait_idle "选优前" || { echo 1 > "$exitfile"; exit 1; }
  run_step "GEMM 选优" "$tool" run --source layer3 --input "$tune_input" --gpus "$gpus"
  status=$?

  echo "$status" > "$exitfile"
  exit "$status"
fi

# -------------------------------------------------------------- launcher ----
mkdir -p "$root" || { echo "无法创建目录：$root" >&2; exit 1; }
[ -x "$tool" ] || { echo "找不到或不可执行：$tool" >&2; exit 1; }
command -v rocm-smi >/dev/null || { echo "找不到 rocm-smi" >&2; exit 1; }

if [ -f "$lock" ] && kill -0 "$(cat "$lock" 2>/dev/null)" 2>/dev/null; then
  echo "已有运行在进行中（PID $(cat "$lock")），拒绝重复启动；确认无误可删除 $lock" >&2
  exit 1
fi

stamp="$(date +%Y%m%d-%H%M%S)"
pidfile="$root/$stamp.pid"
exitfile="$root/$stamp.exit"
runlog="$root/$stamp.log"

# 新会话 + 忽略 SIGHUP + 脱离 stdin/stdout，关闭终端不会杀掉任务。
# 没有 setsid 命令（如 macOS）时用 python3 的 os.setsid 兜底。
if command -v setsid >/dev/null; then
  detach=(setsid nohup)
elif command -v python3 >/dev/null; then
  detach=(nohup python3 -c 'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])')
else
  detach=(nohup)
fi
"${detach[@]}" bash "$script" --worker "$pidfile" "$exitfile" \
  > "$runlog" 2>&1 < /dev/null &
disown 2>/dev/null || true

for _ in $(seq 1 50); do
  [ -s "$pidfile" ] && break
  sleep 0.1
done
[ -s "$pidfile" ] || { echo "启动失败，查看日志：$runlog" >&2; exit 1; }

pid="$(cat "$pidfile")"
echo "进程 PID：$pid"
echo "运行日志：$runlog"
echo "采集日志：$capture_log"
echo "退出码文件：$exitfile（运行结束后生成）"
echo "查看进度：tail -f $runlog"
echo "停止运行：kill -TERM $pid"
