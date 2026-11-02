#!/usr/bin/env bash
# 后台依次运行两组完整测速：当前默认序列策略，再加 auto 对照。
# 断开终端 / ssh 后继续运行；用法：bash run_background.sh [benchmark.py 额外参数...]
# 额外参数（如 --cases ... --rounds N）会同时传给两组运行；不要在其中指定 --output 或 --sequence-policy。
set -u

script="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
benchmark="$(dirname "$script")/benchmark.py"
out_root="${GR00T_BENCH_OUT_ROOT:-/workspace/data/outputs}"
lock="$out_root/.all-optimizations.lock"

# ---------------------------------------------------------------- worker ----
if [ "${1:-}" = "--worker" ]; then
  shift
  output="$1"; auto_output="$2"; pidfile="$3"; shift 3
  child=

  cleanup() { rm -f "$lock"; }
  stop() {
    trap '' TERM INT                      # 收尾期间不再被重复信号打断
    echo "收到终止信号，转发给子进程 ${child:-无}"
    if [ -n "$child" ]; then
      kill -TERM "$child" 2>/dev/null
      wait "$child" 2>/dev/null
    fi
    echo "已终止"
    exit 143
  }
  trap cleanup EXIT
  trap stop TERM INT

  echo $$ > "$pidfile"
  echo $$ > "$lock"

  run_one() {   # run_one 名称 输出目录 [额外参数...]
    local name="$1" dir="$2"; shift 2
    echo "[$(date '+%F %T')] 开始${name}：$dir"
    python3 -u "$benchmark" --output "$dir" "$@" &
    child=$!
    wait "$child"
    local status=$?
    child=
    echo "[$(date '+%F %T')] ${name}结束，退出码：$status"
    return $status
  }

  run_one "当前配置" "$output" "$@"
  first_status=$?
  run_one "auto 配置" "$auto_output" --sequence-policy auto "$@"
  auto_status=$?

  echo "$first_status $auto_status" > "${output}.exit"
  [ "$first_status" -eq 0 ] && [ "$auto_status" -eq 0 ]
  exit $?
fi

# -------------------------------------------------------------- launcher ----
mkdir -p "$out_root" || { echo "无法创建输出根目录：$out_root" >&2; exit 1; }
[ -f "$benchmark" ] || { echo "找不到 $benchmark" >&2; exit 1; }
command -v python3 >/dev/null || { echo "找不到 python3" >&2; exit 1; }

if [ -f "$lock" ] && kill -0 "$(cat "$lock" 2>/dev/null)" 2>/dev/null; then
  echo "已有运行在进行中（PID $(cat "$lock")），拒绝重复启动；确认无误可删除 $lock" >&2
  exit 1
fi

output="$out_root/all-optimizations-$(date +%Y%m%d-%H%M%S)"
auto_output="${output}-auto"
pidfile="${output}.pid"
log="${output}.log"

# setsid 让进程脱离当前会话，ssh 断开或会话被清理时不会随之被杀；nohup 额外忽略 SIGHUP。
detach=(nohup)
if command -v setsid >/dev/null; then detach=(setsid nohup); fi
"${detach[@]}" bash "$script" --worker "$output" "$auto_output" "$pidfile" "$@" \
  > "$log" 2>&1 < /dev/null &

# 等待 worker 写出自己的 PID，确认真正启动。
for _ in $(seq 1 50); do
  [ -s "$pidfile" ] && break
  sleep 0.1
done
if [ ! -s "$pidfile" ]; then
  echo "启动失败，查看日志：$log" >&2
  exit 1
fi

pid="$(cat "$pidfile")"
echo "进程 PID：$pid"
echo "当前配置结果：$output"
echo "auto 配置结果：$auto_output"
echo "运行日志：$log"
echo "退出码文件：${output}.exit（运行结束后生成）"
echo "查看进度：tail -f $log"
echo "停止运行：kill -TERM $pid"
