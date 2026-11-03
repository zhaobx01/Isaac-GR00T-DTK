# DCU GEMM 自动化选优工具使用指南

> 适用：DTK 26.04 自带的 `rocblas` 自动化选优工具（`auto_select_tools`）
> 版本备注：本指南基于实测验证（BW200 / gfx936 / 8 卡环境）

---

## 0. 工具是干什么的

输入一份 **GEMM size 清单**，工具自动完成：

```
拆分 size → kernel 选优 → 原库/选优库性能对比 → 生成最优配置 →（可选）更新配置
```

产出：比原库更优的 rocblas 配置文件 + 性能对比 CSV。

---

## 1. 工具目录

```
$DTKROOT/lib/rocblas/auto_select_test/auto_select_tools
```

> `$DTKROOT` 是 DTK 安装路径，例如 `/opt/dtk-26.04`。

目录结构要点：

| 目录/文件 | 用途 |
|---|---|
| `configs/tuning_args.txt` | ★ 选优参数配置（必改） |
| `csv_size/csv/` | CSV 格式输入目录 |
| `rocblas_layer3_size/` | Layer3 格式输入目录 |
| `rocblas_layer4_size/` | Layer4 格式输入目录 |
| `optimization_configs/new/config/` | 选优结果配置文件 |
| `optimization_results/` | 性能对比汇总 CSV |
| `Auto_tuning.py` | 主入口脚本 |

---

## 2. 准备输入文件（source 决定格式）

### 格式 1：CSV（手写 size 最方便，支持 tf32）

放 `csv_size/csv/`，表头固定 7 列：

```csv
gemmtype,TransA,TransB,M,N,K,B
sgemm,N,N,1024,1024,1024,1
hpa,N,N,4096,4096,4096,1
tf32,T,N,512,512,512,1
```

- `gemmtype`：`sgemm/dgemm/hgemm/cgemm/zgemm/hpa/bf16/bss/int8/tf32`（必须是硬件平台支持的）
- `TransA/TransB`：`N`/`T`（复数 cgemm/zgemm 才支持 `C`）
- 特殊：`N` 写 `-` 自动遍历，仅限 KME/ECO/BW/NMZ 平台的 `hpa/bf16/int8` + `NN/TN` + `B=1`

### 格式 2：Layer3（抓真实调用，支持 tf32）

1. 跑程序前设置环境变量，rocblas 会把每个 GEMM 打印成一行 `./rocblas-bench ...` 命令（**输出走 stderr**）：
   ```bash
   export ROCBLAS_LAYER=3
   python train.py ... 2> layer3.log        # 或用 2>&1 | tee layer3.log
   ```
2. **强烈建议只跑小步数**（Layer3 会打印每一次调用，日志巨大且拖慢运行）
3. 想选 TF32，在每条命令末尾手动加 `--math_mode 1`（layer3 里 TF32 与 sgemm 不区分）
4. **每行只能有一条 `rocblas-bench` 命令**，否则报错

### 格式 3：Layer4（抓真实调用，不支持 tf32）

```bash
export ROCBLAS_LAYER=4
python train.py ... 2> layer4.log
```

每行一条 YAML 配置（`- { rocblas_function: "rocblas_gemm_ex", ... }`），每行一条。

---

## 3. 日志清理去重（Layer3/4 必做）

训练时 tqdm 进度条用 `\r` 刷新会把 rocblas 输出串行成脏行，**必须先清理再放目录**：

```bash
grep "rocblas-bench" 原始日志.log \
  | sed -E 's/^.*(\.\/rocblas-bench)/\1/' \
  | sort -u > rocblas_layer3_size/layer3.log

# 验证：所有行应以 ./rocblas-bench 开头，且每行只有一条
grep -c "^\./rocblas-bench" rocblas_layer3_size/layer3.log
wc -l rocblas_layer3_size/layer3.log
```

---

## 4. 修改配置（必改 4 项）

编辑 `configs/tuning_args.txt`：

```ini
mode=user                  # 固定 user，别改
source=layer3              # 与输入目录对应：csv / layer3 / layer4
hardware_type=???          # ★目标机器平台：ZF/ZF2/KM/KME/ECO/BW/NMZ
gpu_ids=0 1 2 3            # ★目标机器空闲的卡号（空格分隔，用 rocm-smi 确认）
cu_count=0                 # 0=不加cu后缀；也可填实际cu数（如72）
```

安全项建议保持：

```ini
should_clean=1             # 全新选优=1；中断续跑=0
update_target_config=0     # 0=手动更新最安全；1=自动覆盖原库配置（先备份！）
```

可选调节项：`chunk_size`（每卡每轮处理 size 数）、`tune_warmup_times`、`tune_loop_times`、`bench_warmup_times`、`bench_loop_times`、`test_times`。

---

## 5. 启动命令

```bash
cd $DTKROOT/lib/rocblas/auto_select_test/auto_select_tools
source $DTKROOT/env.sh
python3 -u Auto_tuning.py 2>&1 | tee tuning.log
```

要点：

- **新开终端**，别加载多余环境变量（尤其不要设 `HIP_VISIBLE_DEVICES`，会限制可见卡数）
- 跑前用 `rocm-smi` 确认 `gpu_ids` 对应卡都空闲
- 中断续跑：`should_clean=0` 后重跑 `python3 -u Auto_tuning.py`

---

## 6. 结果使用

### 结果位置

| 产物 | 路径 |
|---|---|
| 最优配置 | `optimization_configs/new/config/library_gpu*/G*` |
| 性能对比 | `optimization_results/merged_results_*.csv` |

**`library_gpu*` 平台映射**：ZF→gpu1、ZF2→gpu2、KM→gpu3、KME/ECO→gpu4、BW→gpu5、NMZ→gpu6

### 生效方式（三选一）

```bash
# 方式1：临时环境变量（推荐先验证，关终端失效）
export ROCBLAS_TENSILE_LIBPATH=$(pwd)/optimization_configs/new/config/library_gpu*

# 方式2：永久替换（先备份！）
cp -r $DTKROOT/lib/rocblas/library_gpu* $DTKROOT/lib/rocblas/library_gpu*.bak
cp optimization_configs/new/config/library_gpu*/G* $DTKROOT/lib/rocblas/library_gpu*/

# 方式3：自动更新（config 里 update_target_config=1，会覆盖原配置，谨慎）
```

### 查看提升

看 `merged_results_*.csv`：

- `gflops_ratio = new_gflops / old_gflops`：**>1 = 提升**，<1 = 变慢
- `compare_kernel`：选优前后 kernel 是否一致

---

## 7. 常见问题

| 问题 | 解决 |
|---|---|
| 日志一行含多条命令报错 | 每行必须一条 `rocblas-bench`；用 `grep`/`sed` 清理 |
| 进度条串行成脏行 | `sed -E 's/^.*(\.\/rocblas-bench)/\1/'` 清前缀 + `sort -u` |
| 中断后继续 | `should_clean=0` → 重跑 `python3 -u Auto_tuning.py` |
| 选优被干扰 | 确认目标卡空闲（`rocm-smi` 看 HCU%） |
| 想手动 TF32 | layer3 每条命令末尾加 `--math_mode 1` |
| `rocblas-bench` 提示 Permission denied | DTK 自带二进制无执行位，`chmod +x` 或复制后执行 |

---

## 8. 测量准确性注意事项（重要）

1. **GPU 争抢**：同卡竞争 → 计时直接失真、可能选错 kernel；跨卡竞争 → 共享功耗/散热，boost 频率被压低 → 引入噪声。**选优时用的卡必须空闲**。
2. **CPU 负载**：工具用 GPU 事件计时，纯计算时间由 GPU 决定，但 **launch 开销依赖 CPU**。小 size 受 CPU 忙影响明显。**load average 高时别跑**（参考：96 核机器 load 应 < 20）。
3. **卡数 vs 结果**：每个 size 只在一张卡上选优，同型号卡相对排名一致 → **1 张和 8 张卡选出的 kernel 相同**，只是 8 卡并行快 ~8 倍。
4. **跨机器移植**：kernel 选择结果（同 DCU 型号）可跨机器用；但绝对性能数字不能跨机器直接比，只能本机自对比。
5. **工具自动抗噪**：`tune_loop_times` / `bench_loop_times` / `test_times` 多轮重复取平均，能扛轻微波动，扛不住持续争抢。

---

## 9. 换机器操作速查

```bash
# ① 找到工具目录
cd $DTKROOT/lib/rocblas/auto_select_test/auto_select_tools

# ② 确认机器平台和空闲卡
rocm-smi                    # 看 HCU 型号/占用
lscpu | grep "Model name"   # 确认 CPU

# ③ 放清理后的日志到 rocblas_layer3_size/

# ④ 改 configs/tuning_args.txt：hardware_type / gpu_ids / cu_count / source

# ⑤ 启动（新终端）
source $DTKROOT/env.sh
python3 -u Auto_tuning.py 2>&1 | tee tuning.log

# ⑥ 跑完拿配置
# optimization_configs/new/config/library_gpu*/G*
# optimization_results/merged_results_*.csv
```

**换机器最容易漏的 3 点**：① `hardware_type` 改成目标平台；② `gpu_ids` 填空闲卡；③ 日志每行只有一条命令且无进度条污染。
