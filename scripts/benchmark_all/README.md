# batch 768 测速

每个 case 依次运行 N 轮测速和 1 轮 Trace；`--rounds` 默认 2，因此当前启用 5 项，共启动 15 次训练，每次独立进程、40 步。

```bash
# 全部项目，使用新目录
python3 /workspace/Isaac-GR00T/scripts/benchmark_all/benchmark.py \
  --output "/workspace/data/outputs/benchmark-$(date +%Y%m%d-%H%M%S)"

# 基线和新增 OBR 组合
python3 /workspace/Isaac-GR00T/scripts/benchmark_all/benchmark.py \
  --output "/workspace/data/outputs/selected-$(date +%Y%m%d-%H%M%S)" \
  --rounds 2 \
  --cases baseline combo_obr combo_obr_dit_vl combo_obr_dit_vl_text combo_obr_dit_vl_text_allgather

# 只检查计划和环境，不启动训练、不创建目录
python3 /workspace/Isaac-GR00T/scripts/benchmark_all/benchmark.py \
  --output /tmp/benchmark-preview --dry-run

# 后台依次采集默认策略和 auto，分别写入两个目录
/workspace/Isaac-GR00T/scripts/benchmark_all/run_background.sh

# 在原目录续跑缺少的轮次，参数与首次一致
python3 /workspace/Isaac-GR00T/scripts/benchmark_all/benchmark.py \
  --output /workspace/data/outputs/benchmark-实际时间 \
  --rounds 2 --resume
```

`--cases` 必须包含 baseline，`--rounds` 至少为 2。`dit` 已改名为 `dit_model`。编译范围由 `GR00T_COMPILE_DIT`（`off/model/blocks`）、`GR00T_COMPILE_VL`（`0/1`）、`GR00T_COMPILE_TEXT`（`off/layers/forwards`）三个变量叠加控制，例如 `dit_model` 为 `GR00T_COMPILE_DIT=model`。组合 case 使用 `GR00T_COMPILE_DIT=model GR00T_COMPILE_VL=1 GR00T_COMPILE_TEXT=forwards`；`validate_run` 会逐项核对 `compiled_paths` 与 manifest 的边界。

后台脚本先采集当前默认策略，再采集 `auto`，分别使用输出目录及其 `-auto` 后缀目录，共用脚本打印的运行日志。两次各运行全部 case（当前分别 15 次训练）；第一轮有失败记录时仍继续第二轮，日志记录各轮退出码。恢复 auto 目录时需在原命令中加 `--sequence-policy auto --resume`；恢复默认策略目录时使用 `--resume`。

| 组合 | 优化项 |
|---|---|
| combo_runtime | 输入 BF16 + FusedAdam + GRO |
| combo_full | 输入 BF16 + FusedAdam + GRO + 整体 DiT/VL 编译 |
| combo_text_layers | 输入 BF16 + FusedAdam + GRO + 文本层模块编译 |
| combo_text_forwards | 输入 BF16 + FusedAdam + GRO + 文本层 forward 编译 |
| combo_allgather | 输入 BF16 + FusedAdam + GRO + 整体 DiT/VL 编译 + AllGather |
| combo_obr | 输入 BF16 + FusedAdam + GRO + OBR |
| combo_obr_dit_vl | combo_obr + 整体 DiT/VL 编译 |
| combo_obr_dit_vl_text | combo_obr_dit_vl + Text Forwards 编译 |
| combo_obr_dit_vl_text_allgather | combo_obr_dit_vl_text + AllGather |

默认序列策略为 pad-156，所有 case 的输入统一补齐至 156；输入长度超过 156 时会报错。

可用 combo_runtime 对照 combo_obr，再依次对照后续 combo_obr 组合，观察新增优化的影响。含 Text Forwards 的两个新组合使用旧 CLI 编译入口，同时启用 DiT、VL 和文本 forward；其他组合使用环境变量选择编译范围。报告的“相对基线”仍统一以 baseline 为参考，上述差异需按对应行自行比较。

新增 OBR 组合需要实际测试，不能仅凭配置认定数值正确或性能提升。动态分桶未加入组合。视觉布局、视觉 Block 编译及 FlashAttention Dynamo 边界隔离默认启用。进程间沿用框架默认磁盘编译缓存。

目录结构以 baseline 为例：

```text
output/
  metadata.json
  results.json
  report.md
  baseline/
    baseline-1/      # 测速，第 1 轮
      memory.jsonl  # 每 10 秒的八卡显存原始采样
      step_metrics/ # 各 rank 耗时及 PyTorch 显存峰值
    baseline-2/      # 测速，第 2 轮
    baseline-trace/  # 独立 Trace 轮
      profiling/
        trace_rank_0_iter_35.json  # rank 0–7 各一份
        rank0-key-averages.txt
```

测速按每步八卡最慢 rank 统计第 35 步及第 31–40 步均值，分别生成两张表。跨轮至少两轮有效才计算中位数；有效性仍采用第 11–40 步 CV ≤5%、前后均值漂移绝对值 ≤5%。每轮明细的吞吐量采用第 35 步口径。

测速轮每 10 秒记录设备显存和 GPU 利用率；报告显存表汇总单卡采样最大占用，以及各 rank PyTorch allocated/reserved 峰值的最大值。各卡、各 rank 明细保存在 results.json。设备采样可能漏过瞬时峰值，且包含 PyTorch 外占用；PyTorch 峰值为 train_begin 重置后到 train_end 的分配器统计。查询错误单独记录，缺失值显示为 `—`，不改变测速有效性。

Trace 轮启用 `with_stack`，配置 skip_first=28、wait=1、warmup=5、active=1、repeat=1，仅采第 35 步。导出后停止 profiler，训练继续至第 40 步；保存八卡 Trace 及 rank 0 的 `prof.key_averages().table(sort_by="self_cuda_time_total")` 输出。Trace 独立检查采集成功，不进入吞吐统计，也不使用测速稳定性判定。

`--resume` 仅支持新版嵌套目录，拒绝旧平铺目录且不会移动旧数据；如旧记录使用不同优化配置或无法核实，拒绝续跑，请使用新输出目录。Ctrl+C 或 SIGTERM 会清理训练进程并记录当前轮失败；已记录的失败或无效轮次也会跳过；存在未记录的运行目录时拒绝覆盖。测速和 Trace 任一记录失败，脚本最终退出码为 1；详情查看 report.md 和对应 train.log。

启动前沿用现有设备空闲检查：每 10 秒打印八卡均值，各卡显存低于脚本配置阈值才启动，GPU 利用率不参与放行判断。测速结果不代表数值一致性验收。
