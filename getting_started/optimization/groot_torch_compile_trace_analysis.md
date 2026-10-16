# GR00T N1.7 训练 trace 分析与 torch.compile 建议

> 归档说明：本副本随代码保存，原始记录保留在 `/workspace/scripts/Optimization/`。提交状态和实验结论描述记录时的状态；模型、数据、训练输出及 DTK 运行环境需单独准备。

分析日期：2026-10-15。对象：`/workspace/Isaac-GR00T`（GR00T N1.7、Cosmos Reason2、DTK/HIP 环境）。
本文记录现有 trace 的分析及 HTA 复核结果；候选窗口差值仅用于筛选 compile 位置，**编译收益尚未实测**，不是可回收时间或提速承诺。
姊妹文档：[groot_dtk_optimization_record.md](groot_dtk_optimization_record.md)（已有适配与优化的来源记录）。

## 0. 摘要

| 结论 | 证据 |
| --- | --- |
| 样本存在设备活动空档，比例依赖统计口径 | rank0 的 GPU annotation 区间为 369.134 ms，设备活动并集为 278.411 ms，未记录活动的区间约 90.723 ms（24.58%）；HTA 的独立口径给出各 rank idle 25.91%–29.70% |
| 通信耗时显著，该样本未观察到通信与计算重叠 | rank0 NCCL kernel 累计 104.609 ms；原始事件及 HTA 均未观察到重叠。DeepSpeed ZeRO-2 在本配置路径下让计算流等待刚发出的梯度归约，trace 中已排队的计算 kernel 也在 NCCL 结束后才开始，二者一致（见 2.2） |
| 值得试 compile 的是动作头 DiT 和冻结的文本层 | DiT 前向 Python 窗口 49.6 ms、窗口内 kernel 累计 34.4 ms；文本阶段窗口 37.9 ms、计算 kernel 累计 18.4 ms，均需核对异步归属 |
| 通信、优化器和同步路径需要分别诊断 | 当前先评估模型张量计算的编译，其他路径的优先级与限制见第 5 节 |

## 1. 环境、配置与数据口径

| 项目 | 内容 |
| --- | --- |
| 硬件 | 8 × BW1000 64G（80 SM），HIP 6.3.26113 |
| 软件 | torch 2.10.0、DeepSpeed 0.18.4（ZeRO-2、bf16）、transformers 4.57.3、flash_attn 2.8.4（均读自本机已安装包） |
| 运行 | `torchrun` 8 卡，`global_batch=32`（每卡 4 个样本），`dataloader_num_workers=4`，`max_steps=40`，`enable_profiling` |
| 可训练部分 | `tune_llm=false`、`tune_visual=false`、`tune_projector=true`、`tune_diffusion_model=true`；可训练参数 1,620,515,968（51.54%） |
| trace | `/workspace/data/outputs/libero-profile-20261015-161440-2221399/profiling/trace_rank_{0..7}_iter_23.json`，各约 83 MB，每个含一步 `ProfilerStep#22` |
| 分析范围 | 详细阶段分析以 rank0 的单个 `ProfilerStep#22` 为主；HTA 0.5.0 解析全部 8 个 rank，复核时间分类、kernel、idle 和通信计算重叠 |

**时间口径**：CPU `user_annotation` 的 step 长度为 382.079 ms；设备 stream 0 上同名 `gpu_user_annotation` 区间为 369.134 ms，二者起止不同。kernel 累计约 267.521 ms；在 GPU annotation 区间内，将 `kernel`、`gpu_memcpy`、`gpu_memset` 事件裁剪后取时间并集，设备活动为 278.411 ms，区间内未记录这些活动的时间为 90.723 ms（24.58%）。这描述事件覆盖率，不表示 SM 利用率，也不能从空档直接判定等待原因。

原来的 114.6 ms 来自 CPU step 减去 kernel-only 时长：382.079 − 267.521 ≈ 114.558 ms。该差值包含 memcpy/memset 活动及两种窗口的边界差异，不能直接称为 GPU idle。

**HTA 独立口径**：各 rank 的 `idle_time_pctg` 为 25.91%–29.70%，rank0 为 26.34%；rank0 的 `idle_time` 为 97.228 ms、`compute_time` 为 157.301 ms、`non_compute_time` 为 114.604 ms。HTA 按自身事件筛选与分类规则汇总，不与上面的原始事件并集结果直接互换。`non_compute_time` 不是 idle；其 114.604 ms 与原先的 kernel-only 差值接近，不代表二者来自同一统计。

**实际模型结构**（以 trace 与检查点为准）：

| 部件 | 数量 | 依据 |
| --- | --- | --- |
| 视觉块 | 24 | `Torch-Compiled Region: 0/0` 出现 24 次 |
| 文本层（`select_layer`） | 16 | `FlashAttnVarlenFunc` 共 40 次 = 24 视觉 + 16 文本；运行输出目录 `experiment_cfg/final_model_config.json` 中 `select_layer=16` |
| DiT 块 | 32（16 个交叉注意力 + 16 个自注意力，交错） | `aten::scaled_dot_product_attention` 共 36 次 = 32 DiT + 4 `vl_self_attention`；`final_model_config.json` 中 `num_layers=32`；日志 DiT 参数量 1,091,722,240 |
| `vl_self_attention` | 4 层 | 同上，前向 5.4 ms；训练日志每个 rank 打印 `SelfAttentionTransformer` 参数量 201,433,088；基础检查点 `config.json` 中该配置为 4 层 |

注意：`experiment_cfg/config.yaml` 中的 `num_layers=16`、`select_layer=12` 是配置类默认值（`gr00t/configs/model/gr00t_n1d7.py:47`、`:94`），不代表实际架构；实际架构来自检查点，记录在 `final_model_config.json`（32 层 DiT、`select_layer=16`）。该文件中未见 `vl_self_attention_cfg` 键，但日志与 trace 都显示 4 层存在，以后者为准。

**采集限制**：每个 rank 只有一个 iteration，HTA 明确警告结果可能不准确；解析时还因浮点时间戳事件重叠而进行了精度取整。rank0 有 219,479 条 `python_function` 栈事件，采集可能增加 CPU 开销及改变调度。因此主机等待、重叠率及 rank 差异均仅代表该样本，需用多个稳定 step、无栈采集及关闭 profiler 的测速复核。

## 2. 主要瓶颈及证据

本节分析了下列几类瓶颈，各小节给出详细证据；"如何解决"只写修补的大方向，具体方案见第 3–5 节。

| 类别 | 小节 | 主要证据 | 如何解决 |
| --- | --- | --- | --- |
| 通信开销候选 | 2.2 | NCCL kernel 累计 104.609 ms，该样本未观察到计算重叠；计算流在设备端等待每桶归约完成；各 rank 的 NCCL 时间 91–107 ms | 以 DeepSpeed 的流依赖为切入点，单变量评估 `contiguous_gradients`、分桶大小、DeepSpeed 版本行为或 DDP；再看 RCCL 拓扑 |
| 步边界主机工作与同步候选 | 2.1、2.3 | CPU step 开头约 18.3 ms 与结尾约 15.7 ms 未观察到 kernel；`model.train()` 6.3 ms、`floating_point_ops` 9.6 ms、`zero_grad` 5.4 ms；步内 3 次设备同步共 85.8 ms | 对模式切换、FLOPs 统计和吞吐计时器同步分别做单变量对照，确认实际收益 |
| 主机–设备同步拷贝 | 2.4 | 115 次同步拷贝共 18.2 ms 主机时间，其中 `sample_time` 6.55 ms | CPU 元数据一次性搬运、pinned + `non_blocking`；数据相关同步保持 eager |
| 内核启动开销与碎片化候选 | 2.5、2.6 | DiT 前向窗口内 kernel 时间占比约 69%，约 42 个内核/块；视觉塔与文本阶段占比约 43%、49% | 先验证 DiT 单块，再独立比较逐块与整体编译，随后测试冻结文本层 |
| GEMM 计算占比较高 | 2.5 | DiT 前向窗口内 GEMM 占 kernel 累计时间约 75% | 不能仅凭占比判定计算受限，暂不优先调整 GEMM |
| 优化器 | 2.1 | 阶段 H 共 64.6 ms，其中计算 37.3 ms、通信 24.8 ms | 评估 fused AdamW（HIP 支持待验证） |
| 已有 compile 的收益待对照 | 2.6 | 视觉块每块 4 张图加 1 次 FlashAttn；缺少 eager 对照，断点原因未证实 | 先抓 `graph_breaks/recompiles` 日志，再决定调整边界或 mode |
| 样本未显示明显数据等待 | 2.7 | DataLoader `__next__` 仅 0.135 ms | 暂不优先处理，需多步复核 |

### 2.1 一步的阶段分解（rank0）

| 阶段 | 起止 ms | 时长 ms | 窗口内 kernel 时间占比 | 计算 / NCCL ms | 内核数 | 类型（候选判断） |
| --- | --- | --- | --- | --- | --- | --- |
| 未归类：步起始 | 0.0–0.4 | 0.4 | 未单独统计 | — | — | 边界区间 |
| A 训练模式切换与输入准备 | 0.4–11.7 | 11.3 | 0% | 0 / 0 | 0 | 主机串行 |
| B backbone 前奏 | 11.7–18.3 | 6.6 | ~0% | 0 / 0 | 1 | 主机串行 |
| C 视觉塔（冻结） | 18.3–53.8 | 35.5 | 43% | 15.4 / 0 | 290 | 候选：主机/启动受限 |
| D 文本层（冻结） | 53.8–91.7 | 37.9 | 49% | 18.4 / 0 | 1287 | 候选：主机/启动受限 |
| E 动作头前向 | 91.7–155.6 | 63.9 | 72% | 46.1 / 0 | 1561 | 候选：小内核多 |
| 未归类：前向至反向过渡 | 155.6–157.3 | 1.7 | 未单独统计 | — | — | 阶段间区间 |
| F 反向（allreduce 尾部前） | 157.3–263.4 | 106.1 | 85% | 33.5 / **57.1** | 1805 | 含计算与通信；计算中 8.36 ms 属 DeepSpeed 归约路径 |
| G 梯度 allreduce 尾部 | 263.4–301.4 | 38.0 | 92% | 12.1 / 22.7 | 390 | 通信 |
| H 优化器步 | 301.4–366.0 | 64.6 | 96% | 37.3 / 24.8 | 296 | 优化器 + all-gather |
| I 尾部 | 366.0–382.1 | 16.1 | 0% | 0 / 0 | 4 | 主机串行 |

本表以 CPU step 起点为 0，补齐区间后时长按一位小数合计 382.1 ms。阶段边界取自 Python 栈帧（如 `trainer.py:3981 training_step`、`engine.py:2466 backward`、`engine.py:2642 step`）；kernel 按时间戳归入窗口。该比值为窗口内 kernel 累计时长 / Python 窗口时长，不是硬件利用率；多流重叠时累计值也不等于并集。GPU 异步执行可能跨越 Python 边界，因此阶段标签不构成算子归属证明，0% 仅表示该口径下没有或几乎没有 kernel，不能排除 memcpy/memset 等活动。阶段 A 中，HF `training_step` 调用的 `model.train()`（`trainer.py:4010`）递归耗时 6.29 ms，`_prepare_inputs` 耗时 4.84 ms，二者都含栈采集带来的放大。

### 2.2 通信：约 105 ms，样本未观察到计算重叠

- 17 次 bf16 `all_reduce`（另有 2 次单元素 fp32 `all_reduce`），单次 67,149,824–150,994,944 个元素（134–302 MB），17 次元素总数 1,620,515,968，恰好等于可训练参数量。首次 15.1 ms，稳态 2.7–5.7 ms；`all_reduce` 合计 80.5 ms。
- 优化器阶段 1 次大 `_all_gather_base`（每 rank 输入 202,482,192 个元素，24.1 ms）和 1 次 82,304 元素的小 all-gather。二者之和乘以 8 个 rank 恰好也等于 1,620,515,968，与 ZeRO-2 回收各分区更新后参数的行为一致。
- 原始事件中 NCCL kernel 累计为 104.609 ms，该样本未观察到与计算 kernel 的重叠；HTA 的 8 个 rank 均报告 `comp_comm_overlap_pctg=0%`。rank0 kernel-only 并集与累计时长均约 267.521 ms；该数值不含 memcpy/memset，不是全部设备活动时间。
- 流分布：stream 201 承载 17 次 bf16 AllReduce 及 DeepSpeed 归约路径的 cat/逐元素 kernel；stream 0 承载 AllGather 与 2 次 fp32 AllReduce。
- 各 rank 的 NCCL 时间 91.3–106.6 ms；差异可能来自等待较慢的 rank（推断，未逐 rank 对齐验证）。
- **看到 `all_reduce` 而不是 reduce-scatter 是 DeepSpeed 的预期行为**：`zero2_config.json` 设了 `reduce_scatter: true`，但 ZeRO-1/2 实现（`stage_1_and_2.py`）里没有 reduce-scatter 集合调用，该开关只选择分桶路径，最终由 `allreduce_bucket` 调 `dist.all_reduce`（`:1645`）。因此它不是配置失效，但 all_reduce 让每个 rank 都收到完整梯度，通信量比 reduce-scatter 大。
- **零重叠与 DeepSpeed 的流依赖一致**：
  - 源码：DeepSpeed 为梯度归约使用独立的 `reduction_stream`（`stage_1_and_2.py:1165`）。但在 `contiguous_gradients` 与 `overlap_comm` 同时开启时，`reduce_ipg_grads()` 把一桶 all_reduce 放到该流后，紧接着执行 `current_stream().wait_stream(self.reduction_stream)`（`:1035-1039`）。这让后续计算 kernel 必须等刚发出的归约完成；条件 `resolves_data_dependency()` 在 CUDA/HIP accelerator 下为假，该等待会执行。`average_tensor` 的 `:1167-1169` 也有同类等待。
  - trace：17 次 bf16 AllReduce 中有 16 次，在 NCCL kernel 结束前，主机已发射至少 83 个（多数达数百个）stream 0 kernel；NCCL 结束后，下一个 stream 0 kernel 多数在 30–60 µs 内开始。可见计算流是在设备端等待通信，而不是在等主机发射。
  - 结论：零重叠可由该依赖解释，HIP 采集序列化不是必要解释，但不能排除叠加影响。这也意味着用 compile 缩短反向计算，并不能把这部分通信隐藏起来。

### 2.3 CPU step 边界约 34 ms 未观察到 kernel

- 第一个内核在步开始 18.3 ms 后才出现，最后一个内核在步结束前 15.7 ms 就结束。
- 尾部有 HF `Trainer.floating_point_ops` 9.6 ms、DeepSpeed `zero_grad` 5.4 ms。`floating_point_ops` 源码为 `6 * estimate_tokens * num_parameters(...)`（transformers `modeling_utils.py:1787-1789`），耗时来自逐个遍历参数；9.6 ms 含栈采集带来的放大。
- 3 次 `hipDeviceSynchronize` 共 85.8 ms：
  - DeepSpeed `ThroughputTimer.start`（`timer.py:231` 起的函数，同步在 `:237`）发生在 12.0 ms，耗时 0.00 ms；
  - `ThroughputTimer.stop`（`timer.py:246` 起的函数，同步在 `:255`）发生在 315.5 ms，也就是 `_take_model_step` 刚结束之后，阻塞 50.4 ms，等待已排队的优化器计算与 24.06 ms 的参数 all-gather（341.8 ms 起）完成；
  - ZeRO `independent_gradient_partition_epilogue` 在 265.5 ms 同步 35.3 ms。
- 这些同步主要等待已排队的 GPU 工作，其主机阻塞时长不能直接算作 GPU idle。stop 同步可能限制主机提前准备后续工作的机会，但单步 trace 不能证明它造成当前尾部或下一步开头的全部空档。吞吐计时器同步由 `timers.throughput.synchronized` 控制，默认 `true`（`deepspeed/utils/config.py:35-41`）；需单变量对照确认关闭后的步时间、调度及计时变化。

### 2.4 同步拷贝：115 次、18.2 ms 主机时间

| 位置 | 次数 | 主机 ms |
| --- | --- | --- |
| `gr00t_n1d7.py:170 sample_time`（CPU Beta 采样 + 阻塞 `.to(device)`） | 1 | 6.55 |
| DeepSpeed `runtime/utils.py:870 mask_nan_or_inf_with_val_inplace` | 4 | 2.57 |
| `Qwen3VLTextAttention.forward`（`modeling_qwen3_vl.py:415`，16 层 × 2）；实际发生在 `modeling_flash_attention_utils.py:208 _get_unpad_data` | 32 | 2.49 |
| `qwen3_backbone.py:136 _grid_thw_as_int_rows` | 2 | 2.31 |
| `Qwen3VLVisionAttention.forward`（`modeling_qwen3_vl.py:182`，24 块 × 2）；实际发生在 `flash_attn/flash_attn_interface.py:1184` | 48 | 1.66 |
| `trainer.py:3795 _prepare_input` | 8 | 1.21 |

另有 `hipLaunchKernel` 4283 次共 16.2 ms、`hipExtModuleLaunchKernel` 1138 次共 5.5 ms；运行时 API 总主机时间 132.8 ms，其中同步 85.8 ms。

Qwen3 注意力下的 80 次同步拷贝都发生在其调用的 FlashAttention 封装内，位于 `torch._dynamo.disable` 区域。文本侧的 `_get_unpad_data` 用同一个 attention mask 在每层重复计算一次。这些同步不在候选编译图内，也不会因编译而消失。

这 18.2 ms **不能当作可全额回收的浪费**：同步拷贝的主机耗时可能包含等待已排队 GPU 工作的时间。例如 `sample_time` 的 6.79 ms 窗口内 kernel 累计约 6.36 ms，其中 6.22 ms 由 `vl_self_attention` 前向窗口（91.75–97.17 ms）内发射的 kernel 构成（按 correlation 关联）。也就是说，该次阻塞主要是在等 `vl_self_attention` 的 4 个块跑完，而不是拷贝或 CPU 采样本身慢。同步可能推迟后续发射，但这里不能据此确定可消除的停顿，也不能把主机耗时等同传输耗时。

### 2.5 DiT 前向：GEMM 占窗口内 kernel 时间约 75%，窗口差值约 15.2 ms

- `dit.py:348 AlternateVLDiT.forward` 的 Python 窗口为 49.6 ms，窗口内 kernel 累计约 34.4 ms（时间占比约 69%），候选窗口差值约 15.2 ms。该差值包含主机工作、其他设备活动和异步边界影响，不能直接认定为可由 compile 消除的空闲。
- 共 1344 个内核，约 42 个/块（1344 / 32）：GEMM 290 个、25.8 ms；逐元素/融合类 797 个、平均 7.5 µs，合计 5.9 ms。内核家族按名称关键字归类，为近似。
- 阶段 F 的反向计算统计（排除 NCCL）：1794 个计算内核、33.5 ms。其中 21 个（8.36 ms）在 stream 201 上，由 DeepSpeed 归约路径发射：10 个 `CatArrayBatchedCopy`（4.91 ms）来自 `allreduce_bucket` 的 `self.flatten`，即 `_flatten_dense_tensors`（`stage_1_and_2.py:1628`、`:209`）；11 个逐元素 kernel（3.45 ms）。这部分不在模型编译范围内。其余由模型/autograd 发射的部分中，逐元素 799 个（5.74 ms）、归约 280 个（5.0 ms）、cat 31 个（0.25 ms）；它们是阶段窗口汇总，可能包含其他动作头模块，尚未通过前反向关联单独归属到 DiT，不能全部计入 DiT 的编译机会。
- 在 HIP 上 SDPA 被强制为 math 后端（`dit.py:38 _should_force_math_sdpa`、`:52 _sdpa_context`，用于 `:196`），注释写明原因是 HIP 默认 SDPA 在 bf16 反向产生 NaN 梯度；trace 中 `aten::_scaled_dot_product_attention_math` 共 36 次，与 32 个 DiT 块加 4 层 `vl_self_attention` 吻合，说明 math 后端确实在用。

### 2.6 冻结的 backbone：占前向窗口约 55%

- backbone 前向 78.9 ms（`qwen3_backbone.py:587`），占 `compute_loss` 前向 144 ms 的 55%，无反向。
- **视觉块已有 compile**（`qwen3_backbone.py:287`，`mode="max-autotune-no-cudagraphs"`），样本中存在主机调度开销的线索，但缺少 eager 对照，不能判断现有 compile 收益是否不足：
  - 24 个块，每块 5 个嵌套的 `Torch-Compiled Region`（0/0 包含 1/0–4/0，**嵌套不可相加**）。
  - 最外层 0/0 平均 1.217 ms/块，合计 29.2 ms，占视觉前向 35.5 ms 的 82%。
  - GPU 侧 `CompiledFxGraph` 共 96 个（4 个/块），合计 10.9 ms（0.45 ms/块）。
  - 每块 4 张图 + 1 次 FlashAttn，推断存在图断点（与 `qwen3_backbone.py:292` 把 FlashAttention 设为 `torch._dynamo.disable` 吻合），**断点原因未证实**，也无 `graph_breaks` 日志。
- **文本层在训练前向中创建了未使用的 KV cache**：
  - Cosmos-Reason2-2B 的 `text_config.use_cache=true`；`Qwen3Backbone.forward`（`qwen3_backbone.py:587`）调用 `self.model(...)` 时未传 `use_cache`，`check_model_inputs` 回退到 config 值（transformers `utils/generic.py:963-973`），于是 `Qwen3VLTextModel` 创建 `DynamicCache`（`modeling_qwen3_vl.py:810-811`）。
  - trace 中 `DynamicCache.update` 共 16 次，每层一次，主机 1.03 ms。backbone 只取 `hidden_states[-1]`，缓存从未被读取。
  - 这部分直接开销不大，但每层都会写入缓存对象，会影响文本层编译的捕获与守卫（见第 3 节候选 3）。

### 2.7 本样本中暂不优先处理的部分

- DataLoader：该步 `enumerate(DataLoader).__next__` 仅 0.135 ms，说明此时 batch 已就绪；不能排除其他 step 的预取不足、读取抖动或 worker 等待。
- `AccumulateGrad` 537 次共 40.5 ms 主机时间（含在反向内），属于 autograd 路径。事件次数与累计时间只作诊断，不能直接当作独立耗时或 compile 正确性的验收条件。

## 3. 建议编译的位置

下表排序是建议的实验顺序，不代表已验证的收益排名。A 表示可直接尝试接入编译，B 表示当前保持 eager，C 表示已有证据支持先做小范围重构，D 表示证据不足待验证；A/C 均不意味着捕获、数值或性能已通过。

| 实验顺序 | 位置与符号 | 分类 | 依据 | 候选窗口差值 | 主要风险 |
| --- | --- | --- | --- | --- | --- |
| 1 | `dit.py:99 BasicTransformerBlock`（forward `:179`）；限定 `action_head.model.transformer_blocks` 的 32 个实例 | **A** 单块验证后试逐块编译 | DiT 前向每块约 42 个内核，逐元素内核较多；先分别验证自注意力块与交叉注意力块的前向、反向 | DiT 整段前向约 15.2 ms，仅供筛选，不能归给单块；反向收益未量化，阶段 F 中 DeepSpeed 归约路径的 8.36 ms 不计入 | HIP math 策略与数值；dropout=0.2；交叉注意力输入长度及 mask 变化；实际图变体数量待测 |
| 2 | `AlternateVLDiT.forward`（`dit.py:348`）；限定 `action_head.model` 实例 | **A** 与逐块方案独立对照 | 固定 32 层循环、配置分支和张量计算构成合理的子系统边界，可覆盖块间计算；静态循环及列表追加本身不排除编译 | 与逐块方案针对同一 DiT 窗口，不能重复计算收益 | 更大图的捕获、启动成本、显存及动态输入行为待测；不预设比逐块更慢或更脆弱 |
| 3 | `Qwen3VLTextDecoderLayer`（transformers `modeling_qwen3_vl.py:476`）；`backbone.model.language_model.layers` 保留的 16 层 | **A** 先直接尝试局部编译 | 阶段 D 窗口 37.9 ms、计算 kernel 累计 18.4 ms、1287 个内核平均 14.3 µs；冻结无反向，层内含 norm、attention、MLP 及 residual | 阶段 D 差值约 19.5 ms，包含外层准备与层间工作，不是 decoder layer 可回收时间 | FlashAttention 被 `dynamo.disable`，捕获可能分段；层内会写入训练中未使用的 `DynamicCache`（见 2.6），建议先单独对照 `use_cache=False`（行为保持需验证）再试编译；序列长度、mask 及 kwargs 特化需检查；仅在实际断点支持时再考虑重构 |
| 4 | `SelfAttentionTransformer`（`dit.py:416`，forward `:465`）；`action_head.vl_self_attention.transformer_blocks` 的 4 个实例 | **A** 单独试验 | 与候选 1 共享块类型，前向窗口 5.4 ms | 未单独估计 | 单独启用并验证，不与 32 块首次试验同时改动；不将此项耗时计入 32 层 DiT |
| 5 | 已有视觉块编译（`qwen3_backbone.py:276–289`） | **D** 调整依据待补充 | 4 张图/块、外层 Python 区间约 1.2 ms/块、CompiledFxGraph GPU 事件约 0.45 ms/块，缺少 eager 对照 | 待日志 | 属现有方案诊断，不新增重复编译；先查 `graph_breaks/recompiles`，再决定是否比较 eager、`default` 与现用 mode |

**实例与边界**：候选 1 的 32 块和候选 4 的 4 块共享 `BasicTransformerBlock` 类，需按上述实例路径启用，避免全局修改该类影响全部 36 块。逐块与整体 DiT 是两个替代实验，独立进程、相同输入和起始状态下比较，不同时叠加父子编译。

**文本层外部计算**：本机 Transformers 的 `modeling_qwen3_vl.py:846` 在 decoder 循环前准备共享 RoPE，`:863` 在单层返回后调用 `_deepstack_process`；多模态位置索引也在外层准备。逐层编译时这些工作自然位于所选层边界之外，无需为此预先重构。层内 FlashAttention 的禁用边界仍需以实际日志和图记录检查。

**SDPA 支持与待测风险**：`_sdpa_context()` 在 HIP 上使用 `sdpa_kernel(SDPBackend.MATH)`。本机 Torch Dynamo 在 `torch/_dynamo/variables/torch.py:474` 识别该上下文，并由 `variables/ctx_manager.py:1248` 的 `SDPAKernelVariable` 处理；源码已有捕获机制。仍需验证当前 DTK 编译路径是否保留所需后端策略、实际生成计算及反向数值是否安全。编译可能分解或融合原始算子，不以是否仍出现 `aten::_scaled_dot_product_attention_math` 名称作为唯一依据；检查诊断图/生成计算，并验证输出、梯度、优化器状态和参数更新。

另一项待测风险：`_sdpa_context()` 每次调用块时都会经 `_should_force_math_sdpa()` 执行 `os.environ.get`（`dit.py:39`），该查找会落在编译区域内。本机 Dynamo 源码中未找到专门处理 `os.environ` 的逻辑，是否引起图断或额外守卫，需以 `graph_breaks` 日志确认；如有影响，可在初始化时求值一次（假设训练期间环境变量不变）。

## 4. 推荐范围、初始配置与验证

**范围**（分阶段）：

1. 分别对 DiT 自注意力块、交叉注意力块做前向及反向兼容性验证。
2. 通过后，独立比较 eager、32 块逐块编译和整段 DiT 编译；不叠加父子编译，以正确性、稳态收益、启动成本和显存选择边界。
3. 再单独验证文本 16 层；如需评估组合收益，在单项结果明确后逐项启用。
4. 另外 4 个 `SelfAttentionTransformer` 块单独验证；视觉块先诊断已有编译路径，再决定是否调整。实验顺序不构成收益保证。

**初始配置**：

- `mode="default"`，`dynamic` 取默认（自动），`fullgraph=False`；暂不用 `max-autotune`（HIP 上 Triton 模板是否优于 rocBLAS 未验证）。
- 加可关闭的开关，沿用 `gr00t/configs` 的配置体系，默认关。
- 编译入口放在模型构造阶段、`deepspeed.initialize` 之前（HF `Trainer` 在 `train()` 内创建引擎）：文本层可并入 `Qwen3Backbone._apply_transformers_compilation_overrides`（`qwen3_backbone.py:404`，由 `:399` 调用）；DiT 可放在 `Gr00tN1d7ActionHead.__init__`（`gr00t_n1d7.py:43`）。
- 对模块用原地 `module.compile()`，避免 `OptimizedModule` 包装改变参数名或 `state_dict` 键；不要替换 `ModuleList` 里的实例。

**验证**：

1. **对照**：同一代码、同一数据序、相同模型及优化器起始状态和配置（8 卡、`global_batch=32`、workers 4），每次只改变一个编译范围，其他已有编译保持一致。文本层先直接试编译；若实际断点表明需要小重构，再在重构后的同一源码上比较 eager/compile，必要时另与原始路径核对。
2. **数值**：
   - 先对单块做微测试，关掉 dropout 后比较输出与梯度，用 bf16 容差。
   - 再跑约 200 步同数据序的 loss 曲线做统计比较。
   - dropout=0.2 使随机序列不同，不做逐位比对，也不禁用 dropout 来冒充训练正确。
   - 开启 `--check-numerics` 查 NaN/Inf，这是 HIP math SDPA 风险的主要检验。
3. **重编译**：`TORCH_LOGS=graph_breaks,recompiles`。DiT 具有自注意力和交叉注意力两种调用形式，不代表只会生成两张图；实例守卫、输入形状、dtype、mask 和训练状态等都可能影响特化。逐块与整段分别记录图断、图数量及重编译；文本层覆盖代表性长度和 cache 状态，并核对是否触及目标构建的重编译次数上限。
4. **性能**：稳态测速关闭 profiler、`--check-numerics` 及图断诊断日志；每个配置至少 60 步，单独排除首次编译及未稳定的预热步，报中位数、各 rank 最大值、冷启动时间与峰值显存。若预热后仍有重编译，单独报告。另以相同 profiler 设置采集短 trace，对比 kernel 数、时间并集与重叠；诊断耗时不混入吞吐结论。
5. **DeepSpeed 兼容**：检查有效非零梯度、优化器状态及更新后参数有限，可训练参数实际更新，并验证保存重载和关闭路径。`AccumulateGrad` 的基线 537 次仅作辅助观察，编译可能改变 autograd 执行结构，次数相同不能证明正确，次数变化也不直接判失败。
6. **采集偏差**：后续采集多个稳定 step，并做无栈对照，复核主机等待及通信重叠；本轮不执行采集。

## 5. 当前不优先通过 compile 处理的路径

| 位置 | 开销与证据 | 当前判断 | 方向（本次不实施） | 风险 |
| --- | --- | --- | --- | --- |
| NCCL `all_reduce` / `_all_gather_base` | kernel 累计 104.609 ms，样本未观察到计算重叠；计算流在设备端等待每桶归约 | 当前候选编译范围不包含通信；compile 不能直接消除集合通信量，也不能把被流依赖串行化的通信隐藏起来 | 以 `stage_1_and_2.py:1035-1039` 的流依赖为切入点，单变量评估 `contiguous_gradients`、分桶大小（`reduce_bucket_size=1e8`）、DeepSpeed 版本行为或 `use_ddp`；再看 RCCL 与拓扑；无 profiler 测速评估采集影响 | 改变并行与显存 |
| 优化器步 | 窗口 64.6 ms（计算 kernel 37.3 + 通信 kernel 24.8） | DeepSpeed ZeRO 更新与通信耦合，暂不作为首批编译边界；不是优化器普遍不可编译 | 评估 `adamw_torch_fused`（HIP 及 ZeRO 组合支持待验证） | 数值与可用性 |
| `Trainer.floating_point_ops` | 9.6 ms 纯主机 | 与计算无关 | 在 `Gr00tTrainer`（`trainer.py:176`）重写为返回 0 | 只影响 FLOPs 统计 |
| DeepSpeed 吞吐计时器同步 | start/stop 各一次 `hipDeviceSynchronize`，stop 阻塞 50.4 ms | 与计算无关 | DeepSpeed 配置里设 `timers.throughput.synchronized=false`（默认 `true`） | 吞吐统计变为主机计时，精度下降 |
| `sample_time`（`gr00t_n1d7.py:170`） | 主机约 6.55 ms，主要在等 `vl_self_attention` 已排队的 kernel（6.36 ms 中 6.22 ms） | CPU Beta 采样与设备搬运混合，捕获兼容性待测，暂留 eager | pinned 内存 + `non_blocking` 作为候选，可去掉主机阻塞但不减少 GPU 工作，需计入 pinning 成本、传输依赖与生命周期；改 GPU 采样会改变随机序列，而 `:101–111` 说明 CPU/FP32 初始化是刻意的 | 行为与收益需验证 |
| Qwen3-VL 的 `.item()`、`.tolist()`、`_deepstack_process` | 主机同步候选约 4–7 ms，不能等同设备传输耗时；注意力下的 80 次同步发生在 FlashAttention 封装内（`flash_attn_interface.py:1184`、`_get_unpad_data`） | FlashAttention 部分位于 `dynamo.disable` 区域，不在候选编译图内；其余可能涉及数据依赖控制流或动态形状，具体 graph break 需按版本及日志确认 | 当前边界暂留 eager；文本侧 `_get_unpad_data` 每层用同一 mask 重复计算，可另行评估在层外计算一次 | 保持索引与控制流语义 |
| 未使用的文本 KV cache（`qwen3_backbone.py:587` 调用未传 `use_cache`） | `DynamicCache.update` 16 次，主机 1.03 ms，另有缓存写入的设备工作与显存 | 与 compile 无关的冗余工作，但会影响候选 3 的捕获 | 调用时传 `use_cache=False`，单独对照输出与性能 | 需确认输出不变 |
| `model.train()` 与 `set_frozen_modules_to_eval_mode`（`trainer.py:4010`、`qwen3_backbone.py:503`） | 每步 6.29 ms + 2.58 ms 主机时间，均含栈采集放大 | 纯 Python 递归切换模块状态 | HF 每步调用 `model.train()`，冻结子模块不会"已处于 eval"，因此"已处于 eval 时跳过"无效；方向是减少每步对冻结子树的 train/eval 递归遍历 | 冻结模块的 dropout 等行为需保持 eval |
| DiT 的 GEMM | 25.8 ms，占前向窗口内 kernel 累计时间约 75% | 占比高不能证明算子已最优；编译后可能调用库 GEMM 或生成其他实现，需实际 kernel 记录判断 | 当前先测试周边融合，后续再评估 GEMM 后端或 autotune | 编译成本与数值 |

## 6. 证据等级与未验证项

| 等级 | 内容 |
| --- | --- |
| trace 实测 | CPU/GPU annotation 区间、设备事件并集、Python 窗口内 kernel 时间占比、NCCL 次数/大小/时长、样本中观察到的重叠、计算流在 NCCL 结束后才执行已排队 kernel、同步与拷贝次数、`sample_time` 等待的 kernel 来源、反向中 DeepSpeed 归约 kernel 的数量与时长、`DynamicCache.update` 次数、视觉块区域与图数量 |
| HTA 汇总 | 全部 8 个 rank 的时间分类、kernel 分类、idle 分解及通信计算重叠报告（本轮已用 HTA 0.5.0 复算，数字一致）；与原始事件并集按各自口径记录，不互换 |
| 源码确认 | 视觉块现有编译（`qwen3_backbone.py:287`）、FlashAttention `dynamo.disable`（`:292`）、HIP math SDPA（`dit.py:38–57, :196`）、本机 Dynamo 的 `SDPAKernelVariable` 捕获机制（`variables/ctx_manager.py:1248`）、文本层外部的共享 RoPE 与 deepstack 调用（`modeling_qwen3_vl.py:846, :863`）、`sample_time`（`gr00t_n1d7.py:170`）、ZeRO-2 配置与归约路径（`stage_1_and_2.py:1645`）、`contiguous_gradients`+`overlap_comm` 下计算流等待归约流（`:1035-1039`、`:1167-1169`）、吞吐计时器的同步开关及 `floating_point_ops` 计算式、训练前向因 `use_cache=true` 创建 `DynamicCache` |
| 推断 | 视觉块图断点及原因、窗口差值中可回收的部分、重编译数量、rank NCCL 差异的成因、计时器同步与步边界间隔的因果关系、HIP 采集对重叠结果是否有叠加影响 |
| 未验证 | 所有 compile 收益、冷启动成本及显存；逐块与整段 DiT 的比较；DTK 编译后的 SDPA 后端策略、生成计算和反向数值；`_sdpa_context()` 中 `os.environ.get` 的捕获行为；文本层捕获及是否需要重构；`use_cache=False` 的输出一致性与收益；改变 DeepSpeed 流依赖相关配置的收益；关闭计时器同步的收益；fused AdamW 的 HIP/ZeRO 兼容性；无栈、多 step 下的主机等待和重叠；单 iteration、取整及 HIP 采集的影响 |

"候选窗口差值"是 Python 阶段窗口时长减去按时间戳落入该窗口的 kernel 累计时长。它可能包含其他设备活动、不可消除的 CPU 工作、异步跨界和采集开销；不是 GPU idle 的统一定义，也不是 compile 收益的上下界。compile 还可能改变 kernel 本身的执行时间，实际收益只能由同配置的稳态对照确定。

## 7. 后续建议（本次不执行）

1. 在当前基线上用 `TORCH_LOGS=graph_breaks,recompiles` 做短程采集，先弄清视觉块的断点原因。
2. 多个稳定 step 的无栈短程复采，并用无 profiler 测速评估采集影响。
3. 依次进行 DiT 自注意力/交叉注意力单块验证、逐块与整体 DiT 独立对照、文本层验证，再处理另外 4 块及已有视觉编译；每项确认实际捕获、数值和性能后再考虑组合。所有编译收益尚未实测。
4. 对 `floating_point_ops`、计时器同步、`sample_time` 的 pinned 传输、`use_cache=False`、冻结子树的模式切换分别做单变量对照；主机耗时及窗口差值不直接当作可回收时间。
5. 单独排查通信：从 `stage_1_and_2.py:1035-1039` 让计算流等待归约流的依赖入手，单变量评估 `contiguous_gradients`、分桶大小、DeepSpeed 版本行为或 DDP，再看 RCCL 拓扑；用多步低开销时间线确认重叠变化，无 profiler 步时间本身不能证明重叠率。当前 ZeRO-2 路径出现 `all_reduce` 与已查阅实现一致。
