#!/usr/bin/env bash
# DCU GEMM 自动化选优一键脚本，流程见同目录《DCU_选优工具使用指南.md》。
#
# 子命令：
#   check                          仅做环境检查
#   capture [--layer 3|4] [--out F] -- <命令...>
#                                  以 ROCBLAS_LAYER 运行命令，抓取真实 GEMM 调用（stderr → 原始日志）
#   run --input F [选项]           全流程：检查 → 准备输入 → 改配置 → 选优 → 汇总
#
# run 选项：
#   --input F          size 输入文件（必填）：csv 文件，或 capture 得到的原始日志
#   --source S         csv | layer3 | layer4（默认 csv）
#   --hardware H       ZF|ZF2|KM|KME|ECO|BW|NMZ（默认 BW，对应 BW1000/gfx936）
#   --gpus "0 1 ..."   选优使用的卡（默认全部卡）
#   --cu-count N       默认 0（不加 cu 后缀）
#   --tf32             仅 layer3：每条命令末尾追加 --math_mode 1
#   --resume           中断续跑（should_clean=0，不重新准备输入与配置里的 source）
#   --set key=value    直接覆盖 tuning_args.txt 的其他参数，可重复（如 --set chunk_size=20）
#   --force            空闲卡/负载检查不通过时继续
#   --dry-run          只打印将执行的动作，不修改任何文件、不启动选优
#
# 不提供 update_target_config=1（自动覆盖原库配置）；结果生效方式在运行结束时打印。
set -eo pipefail

DTKROOT="${DTKROOT:-/opt/dtk}"
TOOL_DIR="$DTKROOT/lib/rocblas/auto_select_test/auto_select_tools"
BENCH="$DTKROOT/lib/rocblas/benchmark_tool/rocblas-bench"
ARGS_FILE="$TOOL_DIR/configs/tuning_args.txt"
TS="$(date +%Y%m%d-%H%M%S)"

declare -A LIB_GPU=([ZF]=1 [ZF2]=2 [KM]=3 [KME]=4 [ECO]=4 [BW]=5 [NMZ]=6)

DRY_RUN=0
FORCE=0

log()  { echo "[$(date '+%F %T')] $*"; }
warn() { echo "[$(date '+%F %T')] 警告：$*" >&2; }
die()  { echo "[$(date '+%F %T')] 错误：$*" >&2; exit 1; }
act()  { if [ "$DRY_RUN" -eq 1 ]; then echo "[dry-run] $*"; else "$@"; fi; }

usage() { sed -n '2,/^set -eo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

# ------------------------------------------------------------------ 检查 ----
gpu_count() { rocm-smi --showid 2>/dev/null | grep -cE '^(HCU|GPU)\[[0-9]+\]' || true; }

check_env() {   # check_env "<gpu ids>"
  local gpus="$1" bad=0
  log "== 环境检查 =="
  [ -d "$TOOL_DIR" ]            || die "找不到选优工具目录：$TOOL_DIR（检查 DTKROOT=$DTKROOT）"
  [ -f "$TOOL_DIR/Auto_tuning.py" ] || die "缺少 $TOOL_DIR/Auto_tuning.py"
  [ -f "$ARGS_FILE" ]           || die "缺少 $ARGS_FILE"
  [ -f "$DTKROOT/env.sh" ]      || die "缺少 $DTKROOT/env.sh"
  command -v python3 >/dev/null || die "找不到 python3"
  command -v rocm-smi >/dev/null || die "找不到 rocm-smi"
  [ -f "$BENCH" ]               || die "找不到 rocblas-bench：$BENCH"
  log "工具目录：$TOOL_DIR"

  if [ ! -x "$BENCH" ]; then
    log "rocblas-bench 无执行权限，执行 chmod +x"
    act chmod +x "$BENCH" || die "chmod +x $BENCH 失败（需要 root 或自行复制该二进制后修改配置）"
  fi

  if [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
    warn "当前 shell 设置了 HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES，会限制可见卡数；启动选优时将 unset"
  fi

  local use g pct
  use="$(rocm-smi --showuse 2>/dev/null || true)"
  for g in $gpus; do
    pct="$(echo "$use" | sed -nE "s/^(HCU|GPU)\[$g\][^:]*: *[A-Za-z]+ use \(%\): *([0-9.]+).*/\2/p" | head -1)"
    if [ -z "$pct" ]; then
      warn "无法解析卡 $g 的占用率，请用 rocm-smi 手动确认"
    elif awk "BEGIN{exit !($pct > 5)}"; then
      warn "卡 $g 占用 ${pct}%，不空闲"; bad=1
    else
      log "卡 $g 空闲（${pct}%）"
    fi
  done

  local cores load1 limit
  cores="$(nproc)"; load1="$(cut -d' ' -f1 /proc/loadavg)"; limit=$(( cores / 5 ))
  if awk "BEGIN{exit !($load1 > $limit)}"; then
    warn "load average ${load1} 高于参考阈值 ${limit}（${cores} 核 / 5），小 size 的计时会受影响"; bad=1
  else
    log "load average ${load1}（参考阈值 ${limit}）"
  fi

  if [ "$bad" -eq 1 ]; then
    [ "$FORCE" -eq 1 ] || die "检查未通过：换空闲的卡（--gpus）或等负载降下来；确认无误可加 --force"
    warn "--force：忽略以上警告继续"
  fi
}

# --------------------------------------------------------------- capture ----
cmd_capture() {
  local layer=3 out=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --layer) layer="$2"; shift 2;;
      --out)   out="$2"; shift 2;;
      --) shift; break;;
      *) die "capture：未知参数 $1";;
    esac
  done
  [ $# -gt 0 ] || die "capture：用 -- 分隔后给出要运行的命令"
  case "$layer" in 3|4) ;; *) die "--layer 只能是 3 或 4";; esac
  out="${out:-rocblas_layer${layer}_raw-$TS.log}"
  log "ROCBLAS_LAYER=$layer 运行：$*"
  log "stderr 原始日志 → $out（Layer 会打印每次调用，务必只跑很少的步数）"
  ROCBLAS_LAYER="$layer" "$@" 2> "$out" || warn "命令退出码非 0，日志仍已保存"
  log "完成。下一步：$0 run --source layer$layer --input $out"
}

# ------------------------------------------------------------ 输入准备 ----
prepare_input() {   # prepare_input source input tf32
  local source="$1" input="$2" tf32="$3" dir dest
  case "$source" in
    csv)    dir="$TOOL_DIR/csv_size/csv";;
    layer3) dir="$TOOL_DIR/rocblas_layer3_size";;
    layer4) dir="$TOOL_DIR/rocblas_layer4_size";;
  esac

  # 目录里已有的旧输入先备份移走，避免混入本次选优
  local old=() f
  while IFS= read -r f; do old+=("$f"); done < <(find "$dir" -maxdepth 1 -type f ! -name '.gitkeep')
  if [ "${#old[@]}" -gt 0 ]; then
    local bak="$TOOL_DIR/input_backup-$TS"
    log "$dir 中已有 ${#old[@]} 个旧输入文件，移到 $bak"
    act mkdir -p "$bak"
    act mv "${old[@]}" "$bak/"
  fi

  case "$source" in
    csv)
      head -1 "$input" | grep -qx 'gemmtype,TransA,TransB,M,N,K,B' \
        || die "CSV 表头必须是 gemmtype,TransA,TransB,M,N,K,B"
      dest="$dir/$(basename "$input")"
      log "CSV 输入：$(( $(wc -l < "$input") - 1 )) 条 size → $dest"
      act cp "$input" "$dest"
      ;;
    layer3)
      dest="$dir/layer3.log"
      local tmp; tmp="$(mktemp)"
      grep "rocblas-bench" "$input" | sed -E 's/^.*(\.\/rocblas-bench)/\1/' | sort -u > "$tmp" || true
      [ -s "$tmp" ] || { rm -f "$tmp"; die "原始日志里没有 rocblas-bench 行；确认抓取时 ROCBLAS_LAYER=3 且重定向的是 stderr"; }
      [ "$tf32" -eq 1 ] && sed -i 's/$/ --math_mode 1/' "$tmp"
      validate_layer3 "$tmp" || { rm -f "$tmp"; die "layer3 清理后仍有不合规的行"; }
      log "Layer3 输入：去重后 $(wc -l < "$tmp") 条 → $dest$([ "$tf32" -eq 1 ] && echo '（已追加 --math_mode 1）')"
      act cp "$tmp" "$dest"; rm -f "$tmp"
      ;;
    layer4)
      dest="$dir/layer4.log"
      local tmp; tmp="$(mktemp)"
      grep "rocblas_function" "$input" | sed -E 's/^.*(- \{ *rocblas_function)/\1/' | sort -u > "$tmp" || true
      [ -s "$tmp" ] || { rm -f "$tmp"; die "原始日志里没有 rocblas_function 行；确认抓取时 ROCBLAS_LAYER=4"; }
      log "Layer4 输入：去重后 $(wc -l < "$tmp") 条 → $dest"
      act cp "$tmp" "$dest"; rm -f "$tmp"
      ;;
  esac
}

validate_layer3() {
  local total starts multi
  total="$(wc -l < "$1")"
  starts="$(grep -c '^\./rocblas-bench' "$1" || true)"
  multi="$(grep -cE 'rocblas-bench.*rocblas-bench' "$1" || true)"
  [ "$total" -eq "$starts" ] && [ "$multi" -eq 0 ]
}

# -------------------------------------------------------------- 配置文件 ----
set_arg() {   # set_arg key value —— 保留行尾注释
  local key="$1" val="$2" tmp
  grep -qE "^${key}=" "$ARGS_FILE" || die "tuning_args.txt 中没有参数 $key"
  if [ "$DRY_RUN" -eq 1 ]; then echo "[dry-run] tuning_args.txt: $key=$val"; return; fi
  tmp="$(mktemp)"
  sed -E "s|^(${key}=)[^#]*|\1${val}      |" "$ARGS_FILE" > "$tmp"
  cat "$tmp" > "$ARGS_FILE"; rm -f "$tmp"
}

# ------------------------------------------------------------------- 汇总 ----
summarize() {   # summarize hardware
  local hw="$1" lib="library_gpu${LIB_GPU[$1]}"
  local res; res="$(ls -t "$TOOL_DIR"/optimization_results/merged_results_*.csv 2>/dev/null | head -1 || true)"
  local cfg="$TOOL_DIR/optimization_configs/new/config/$lib"
  log "== 结果 =="
  log "最优配置：$cfg"
  if [ -z "$res" ]; then warn "未找到 optimization_results/merged_results_*.csv，选优可能未成功"; return; fi
  log "性能对比：$res"
  python3 - "$res" <<'PY' || warn "汇总 CSV 失败，请直接查看该文件"
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8-sig")))
col = next((c for c in (rows[0].keys() if rows else []) if c.strip() == "gflops_ratio"), None)
if not col:
    print(f"共 {len(rows)} 行，但没有 gflops_ratio 列，请直接查看文件"); sys.exit(0)
r = []
for row in rows:
    try: r.append(float(row[col]))
    except ValueError: pass
if not r:
    print("gflops_ratio 无有效数值"); sys.exit(0)
up = sum(x > 1.0 for x in r); down = sum(x < 1.0 for x in r)
print(f"共 {len(r)} 个 size：提升 {up}，变慢 {down}，持平 {len(r) - up - down}")
print(f"gflops_ratio：最大 {max(r):.3f}  最小 {min(r):.3f}  平均 {sum(r)/len(r):.3f}")
PY
  cat <<EOF

生效方式（先用方式 1 验证，再考虑方式 2）：
  # 方式 1：临时环境变量（关终端失效）
  export ROCBLAS_TENSILE_LIBPATH=$cfg
  # 方式 2：永久替换（先备份）
  cp -r $DTKROOT/lib/rocblas/$lib $DTKROOT/lib/rocblas/$lib.bak
  cp $cfg/G* $DTKROOT/lib/rocblas/$lib/
EOF
}

# -------------------------------------------------------------------- run ----
cmd_run() {
  local source=csv input="" hw=BW gpus="" cu=0 tf32=0 resume=0 extra=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --input)    input="$2"; shift 2;;
      --source)   source="$2"; shift 2;;
      --hardware) hw="$2"; shift 2;;
      --gpus)     gpus="$2"; shift 2;;
      --cu-count) cu="$2"; shift 2;;
      --tf32)     tf32=1; shift;;
      --resume)   resume=1; shift;;
      --set)      extra+=("$2"); shift 2;;
      --force)    FORCE=1; shift;;
      --dry-run)  DRY_RUN=1; shift;;
      -h|--help)  usage; exit 0;;
      *) die "run：未知参数 $1";;
    esac
  done

  case "$source" in csv|layer3|layer4) ;; *) die "--source 只能是 csv|layer3|layer4";; esac
  [ -n "${LIB_GPU[$hw]:-}" ] || die "--hardware 只能是：${!LIB_GPU[*]}"
  [ "$tf32" -eq 0 ] || [ "$source" = layer3 ] || die "--tf32 仅适用于 layer3（csv 里直接写 tf32 行）"
  [ -n "$gpus" ] || gpus="$(seq -s' ' 0 $(( $(gpu_count) - 1 )))"
  [ -n "$gpus" ] || die "未检测到 GPU"
  if [ "$resume" -eq 0 ]; then
    [ -n "$input" ] || die "run：必须指定 --input"
    [ -f "$input" ] || die "输入文件不存在：$input"
    input="$(cd "$(dirname "$input")" && pwd)/$(basename "$input")"
  fi
  local kv; for kv in "${extra[@]}"; do [[ "$kv" == *=* ]] || die "--set 需要 key=value 形式：$kv"; done

  [ "$DRY_RUN" -eq 1 ] && log "dry-run 模式：不会修改任何文件，也不会启动选优"
  log "source=$source  hardware=$hw  gpus=[$gpus]  cu_count=$cu  resume=$resume"
  check_env "$gpus"

  if [ "$resume" -eq 0 ]; then
    log "== 准备输入 =="
    prepare_input "$source" "$input" "$tf32"
  else
    log "--resume：沿用目录里已有的输入"
  fi

  log "== 修改 tuning_args.txt =="
  if [ "$DRY_RUN" -eq 0 ]; then
    cp "$ARGS_FILE" "$ARGS_FILE.bak-$TS"; log "原配置已备份：$ARGS_FILE.bak-$TS"
  fi
  set_arg mode user
  set_arg source "$source"
  set_arg hardware_type "$hw"
  set_arg gpu_ids "$gpus"
  set_arg cu_count "$cu"
  set_arg should_clean "$([ "$resume" -eq 1 ] && echo 0 || echo 1)"
  set_arg update_target_config 0
  for kv in "${extra[@]}"; do set_arg "${kv%%=*}" "${kv#*=}"; done

  local tlog="$TOOL_DIR/tuning-$TS.log"
  log "== 启动选优 =="
  log "日志：$tlog"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] (cd $TOOL_DIR && source $DTKROOT/env.sh && unset HIP_VISIBLE_DEVICES && python3 -u Auto_tuning.py)"
    return
  fi
  (
    cd "$TOOL_DIR"
    set +u
    # shellcheck disable=SC1091
    source "$DTKROOT/env.sh"
    unset HIP_VISIBLE_DEVICES
    python3 -u Auto_tuning.py 2>&1 | tee "$tlog"
  ) || die "选优失败，查看日志 $tlog；修复后可用 --resume 续跑"

  summarize "$hw"
}

# ------------------------------------------------------------------- main ----
sub="${1:-}"; [ $# -gt 0 ] && shift
case "$sub" in
  check)   case "${1:-}" in --force) FORCE=1;; esac
           check_env "$(seq -s' ' 0 $(( $(gpu_count) - 1 )))";;
  capture) cmd_capture "$@";;
  run)     cmd_run "$@";;
  ""|-h|--help) usage;;
  *) usage; die "未知子命令：$sub";;
esac
