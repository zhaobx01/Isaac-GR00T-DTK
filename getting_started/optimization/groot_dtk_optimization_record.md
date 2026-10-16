# GR00T DTK 优化与适配记录

> 归档说明：本副本随代码保存，原始记录保留在 `/workspace/scripts/Optimization/`。提交状态和实验结论描述记录时的状态；模型、数据、训练输出及 DTK 运行环境需单独准备。

记录日期：2026-10-14。对象：`/workspace/Isaac-GR00T`，GR00T N1.7、Cosmos Reason2、DTK/HIP 环境。
本记录基于本地 Git 历史、当前差异和已有实验记录；不修改模型结构、权重或运行环境。

| 分类 | 主要内容 | 应如何理解 |
| --- | --- | --- |
| 性能优化 | CPU 位置元数据、索引缓存、视觉 Block 局部编译 | 具有减少同步或重复计算的动机，尚无逐项提速结论 |
| 兼容修复 | TF32 能力判断、FlashAttention 编译边界、DiT math SDPA | 优先保证可运行和数值稳定，可能增加开销 |
| 运行适配 | 本地模型、视频读取、启动路径、容器与依赖 | 解决环境和数据链路问题，不等于模型计算提速 |
| 可观察性与控制 | 吞吐显示、profiler、数值检查、保存策略 | 提供分析或保护能力；测量本身可能影响速度 |

## 1. 官方来源与适配时间线

| 层次 | 完整 commit / 状态 | 内容 |
| --- | --- | --- |
| 官方基线 | `51d4c89f72fda44cbf77285c6a8114b52676b8a1` | GR00T N1.7；提交说明为 Add support for Jetpack 7.2 on Orin and Thor |
| 第一轮 | `d9bdcf9686649761440496445dd60c1bbb225ea1` | HCU 基础适配，4 个文件，+61/-3 |
| 第二轮、当前 HEAD | `2838dd8f9319ac4cae0b36de413d25783456bfa7` | 位置计算、局部编译与训练观测，6 个文件，+269/-4 |
| 当前工作区 | 尚未提交 | HIP math、离线加载、启动与保存控制、数值检查和文档 |

[官方基线提交](https://github.com/NVIDIA/Isaac-GR00T/commit/51d4c89f72fda44cbf77285c6a8114b52676b8a1)。
演进关系为 `51d4c89 → d9bdcf9 → 2838dd8 → 未提交修改`。
可确认的是提交来源；当前检出状态不能证明最初下载自官方哪个分支，也不表示已同步官方后续更新。

### 1.1 不应计入本地新增优化的上游实现

以下内容在 `51d4c89` 中已经存在，后续继承，不属于上述两轮新增成果：

- Beta sampler 使用 CPU/FP32 构造，避免 meta/default-device 上下文影响初始化。
- 加载后重建非持久 RoPE buffer，避免频率缓冲区无效。
- 视觉 patch-embed Conv3d 的 `channels_last_3d` 处理及 tracing 保护。

核对入口：`git show 51d4c89:gr00t/model/gr00t_n1d7/gr00t_n1d7.py`，以及同提交的 `gr00t/model/modules/qwen3_backbone.py`。
这些实现可能具有性能或正确性价值，但不能重复归功于本地适配。

## 2. 第一轮 HCU 基础适配

来源：`d9bdcf9`。主要改动在训练入口、视频工具和新增的 `start.sh`，不是重写模型架构。

| 适配项 | 原问题与动机 | 实现及边界 |
| --- | --- | --- |
| TF32 能力判断 | NVIDIA 配置不能直接假定 HCU 支持 TF32 | 用 `is_torch_tf32_available()` 与配置共同决定开关；不支持时关闭，不会自动把所有计算改成 FP32 |
| W&B 延迟导入 | 未使用跟踪也可能被其依赖阻断 | 仅在 `use_wandb` 为真时导入；是依赖解耦，不是算子加速 |
| 本地 Cosmos 路径 | 固定远端 ID 不适合离线容器 | 新增 `VLM_MODEL_PATH` 覆盖；当时仍以远端 ID 为默认值，当前未提交修复才将默认值改为本地路径并限制本地加载 |
| OpenCV 读取备选 | 为视频读取提供可替代路径 | `VIDEO_DECODER_BACKEND=opencv` 时按索引读取、转 RGB、恢复请求顺序；可能增加解码成本，当前不强制启用 |
| 启动脚本 | 集中配置模型、数据与多卡参数 | 新增 `start.sh`；路径与离线变量的后续修复属于当前工作区，不混记为第一轮实现 |

当前 `start.sh` 的 batch 960 是已有实验设置；batch 64 的验证不能证明 batch 960 已验收。

## 3. 第二轮模型计算与编译优化

来源：`2838dd8`。核心文件为 [qwen3_backbone.py](../../gr00t/model/modules/qwen3_backbone.py)。
通过实例方法替换安装 CPU 位置处理，通过类级包装控制局部编译；没有改变骨干加动作头的架构。

| 类别 | 优化项 | 主要针对的开销 |
| --- | --- | --- |
| 3.1 减少主机与设备同步等待 | 批量网格读取、插值改写、视觉 RoPE 改写、mRoPE 索引处理 | 设备标量驱动 Python、形状参数和分支引发的同步等待 |
| 3.2 减少重复构造与主机设备传输 | 插值元数据缓存、旋转坐标缓存 | 相同网格下每步重建索引、权重和坐标，并重复搬到设备 |
| 3.3 局部编译与外部算子边界控制 | 视觉 Block 编译、FlashAttention Dynamo 隔离 | eager 调度成本及外部注意力函数的编译兼容性 |

“减少 GPU 空泡”是降低同步等待的动机，不表示已经测得空泡占比下降。各项存在交叉作用，不能将其收益简单相加；本章末尾的“优点”描述机制收益，不代表实测提速比例。

代码来源约定：修改前的位置函数取自本机已安装 Transformers 4.57.3 的 `modeling_qwen3_vl.py`，与配套 `modular_qwen3_vl.py` 的对应方法一致；未将其认定为未经修改的官方发行包。它们是依赖侧实现，不是直接写在 GR00T `51d4c89` 中的函数。修改后取自仓库 `qwen3_backbone.py`；编译安装对照使用 `d9bdcf9 → 2838dd8` 的真实差异。

片段只展示对应逻辑，`# ...` 明确标记省略部分；新增 helper 不伪造旧同名函数。代码需结合原函数上下文阅读，不作为独立脚本执行。

### 3.1 减少主机与设备同步等待

#### 3.1.1 网格元数据批量 CPU 化

**问题与优化动机**：`grid_thw` 保存时间、宽高等整数元数据。原路径直接遍历其设备标量，再把它们交给 Python 控制流或 `linspace` 的长度参数。设备需要先完成相关计算并向主机提供数值，才能继续组装后续工作，可能在计算流中形成等待。

来源：依赖侧 [fast_pos_embed_interpolate](/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:642)；仓库 [_grid_thw_as_int_rows](../../gr00t/model/modules/qwen3_backbone.py#L136) 及其插值调用点。下面是相同遍历位置的变化，不表示原先存在同名 helper。

```diff
-grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
-for t, h, w in zip(grid_ts, grid_hs, grid_ws):
-    h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
-    w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
+grid_rows = _grid_thw_as_int_rows(grid_thw)
+# ... 省略插值函数中的缓存判断与列表初始化
+for _t, h, w in grid_rows:
+    h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device="cpu")
+    w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device="cpu")
```

新增 helper 的关键逻辑：

```python
def _grid_thw_as_int_rows(grid_thw: torch.Tensor) -> list[tuple[int, int, int]]:
    grid_cpu = grid_thw.detach().to(device="cpu", dtype=torch.long)
    return [tuple(row) for row in grid_cpu.tolist()]
```

**实现说明与边界**：CPU 化的是小规模整数网格，不是图像特征或模型权重。如果输入在设备上，这次批量拷贝仍会同步；优化目标是避免随后反复读取设备标量，而不是消除所有 DtoH。该 helper 是位置处理的公共基础，不能再作为每个调用者的独立收益累计。

**优点**：集中处理主机需要的整数信息，让后续循环与形状计算直接使用 Python 整数，减少设备标量读取带来的同步机会。

#### 3.1.2 位置嵌入插值改写

**问题与优化动机**：依赖实现已经在默认设备通常为 CPU 的位置构造插值列表，但其长度、分段和重排参数仍来自 `grid_thw` 的张量标量。只看“列表在 CPU 上”并不能排除这些尺寸读取引发的设备等待。

来源：依赖侧 [fast_pos_embed_interpolate](/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:642)；仓库 [_cpu_fast_pos_embed_interpolate](../../gr00t/model/modules/qwen3_backbone.py#L142)。

**修改前：依赖侧原实现，摘录构造和后处理两个位置**

```python
grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
idx_list = [[] for _ in range(4)]
weight_list = [[] for _ in range(4)]
for t, h, w in zip(grid_ts, grid_hs, grid_ws):
    h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
    w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
    # ... 省略四邻域索引、系数和列表填充
# ... 省略设备查表与四邻域加权求和
patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])
patch_pos_embeds_permute = []
merge_size = self.config.spatial_merge_size
for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
    pos_embed = pos_embed.repeat(t, 1)
    # ... 省略按 spatial_merge_size 进行 view / permute / flatten
```

**修改后：缓存未命中时在 CPU 构造元数据，设备上完成可训练查表**

```python
grid_rows = _grid_thw_as_int_rows(grid_thw)
weight = self.pos_embed.weight
# ... 省略缓存判断；以下构造位于未命中分支
idx_list = [[] for _ in range(4)]
weight_list = [[] for _ in range(4)]
for _t, h, w in grid_rows:
    h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device="cpu")
    w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device="cpu")
    # ... 省略四邻域索引、系数和列表填充
idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=weight.device)
weight_tensor = torch.tensor(weight_list, dtype=weight.dtype, device=weight.device)
# ... 省略缓存写入；查表位于缓存分支之外，每步执行
pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
patch_pos_embeds = patch_pos_embeds.split([h * w for _t, h, w in grid_rows])
merge_size = self.config.spatial_merge_size
permuted = []
for pos_embed, (t, h, w) in zip(patch_pos_embeds, grid_rows):
    pos_embed = pos_embed.repeat(t, 1)
    # ... 省略按 spatial_merge_size 进行 view / permute / flatten
```

**实现说明与边界**：四邻域插值及 patch 排列仍被保留；改变的是整数尺寸的来源和元数据构造位置。可训练 `self.pos_embed` 仍在设备上执行，梯度不需要经过 CPU 整数元数据。CPU 计算、索引搬运与重排仍有成本；缓存的复用单独归入 3.2.1。

**优点**：从插值列表生成到分段、时间重复和布局重排，都改用已读取的整数尺寸，减少处理链中重复取得设备尺寸的需求，同时保留可训练查表路径。

#### 3.1.3 视觉旋转位置坐标改写

**问题与优化动机**：原方法用 `.item()` 取得网格最大值和 token 总数，在设备上分配坐标，再逐网格构造行列坐标。尺寸较小的整数操作与 Python 循环交织，可能产生同步和碎片化设备任务。

来源：依赖侧 [rot_pos_emb](/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:603)；仓库 [_cpu_rot_pos_emb](../../gr00t/model/modules/qwen3_backbone.py#L211)。

**修改前：设备上构造坐标，设备标量参与形状与分支**

```python
merge_size = self.spatial_merge_size
max_hw = int(grid_thw[:, 1:].max().item())
freq_table = self.rotary_pos_emb(max_hw)
device = freq_table.device
total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
offset = 0
for num_frames, height, width in grid_thw:
    merged_h, merged_w = height // merge_size, width // merge_size
    block_rows = torch.arange(merged_h, device=device)
    block_cols = torch.arange(merged_w, device=device)
    # ... 省略块内坐标生成及广播、stack，得到 coords
    if num_frames > 1:
        coords = coords.repeat(num_frames, 1)
    num_tokens = coords.shape[0]
    pos_ids[offset : offset + num_tokens] = coords
    offset += num_tokens
# ... 省略 freq_table[pos_ids] 与 flatten
```

**修改后：CPU 构造完整坐标，合并后搬运到频率表所在设备**

```python
grid_rows = _grid_thw_as_int_rows(grid_thw)
max_hw = max(max(height, width) for _t, height, width in grid_rows)
freq_table = self.rotary_pos_emb(max_hw)
merge_size = self.spatial_merge_size
# ... 省略缓存判断；以下构造位于未命中分支
pos_ids = []
for num_frames, height, width in grid_rows:
    merged_h, merged_w = height // merge_size, width // merge_size
    block_rows = torch.arange(merged_h, device="cpu")
    block_cols = torch.arange(merged_w, device="cpu")
    intra_row = torch.arange(merge_size, device="cpu")
    intra_col = torch.arange(merge_size, device="cpu")
    # ... 省略相同分块顺序的广播、reshape 与 stack，得到 coords
    if num_frames > 1:
        coords = coords.repeat(num_frames, 1)
    pos_ids.append(coords)
pos_ids_device = torch.cat(pos_ids).to(freq_table.device)
# ... 省略缓存写入；频率表索引位于分支之外
return freq_table[pos_ids_device].flatten(1)
```

**实现说明与边界**：位置坐标改在 CPU 生成，频率表及其索引仍在原设备。没有把 RoPE 特征整体搬到 CPU，也没有缓存频率表；频率表调用仍逐次执行。缓存命中优化归入 3.2.2。

**优点**：把网格最大值、分块坐标和时间重复统一放在主机整数流程中，减少零碎设备坐标构造和标量驱动的等待，最后集中传输坐标。

#### 3.1.4 多模态 mRoPE 索引处理

**问题与优化动机**：原方法扫描视觉 token、统计图像/视频数量，再根据网格尺寸组装位置索引；其中 `.tolist()`、`.item()`、循环次数和条件判断依赖张量数值。把它们混在设备输入路径中，会使主机反复等待数值可用。

来源：依赖侧 [get_rope_index](/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:916)；仓库 [_cpu_get_rope_index](../../gr00t/model/modules/qwen3_backbone.py#L248)。修改后是包裹原方法，不是重新发明 mRoPE 算法。

**修改前：依赖侧原方法内部的控制流程摘录**

```python
# ... 省略按 batch、attention_mask 取有效 token 及位置张量分配
vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
vision_tokens = input_ids[vision_start_indices + 1]
image_nums = (vision_tokens == image_token_id).sum()
video_nums = (vision_tokens == video_token_id).sum()
input_tokens = input_ids.tolist()
llm_pos_ids_list: list = []
st = 0
remain_images, remain_videos = image_nums, video_nums
for _ in range(image_nums + video_nums):
    # ... 省略 token 查找及 image/video 网格选择，得到 t、h、w
    llm_grid_t, llm_grid_h, llm_grid_w = (
        t.item(),
        h.item() // spatial_merge_size,
        w.item() // spatial_merge_size,
    )
    # ... 省略文本/视觉位置构造与拼接
```

**修改后：整数输入批量转 CPU，原算法在 CPU 上计算，输出返回原设备**

```python
if input_ids is None or (image_grid_thw is None and video_grid_thw is None):
    return self._gr00t_original_get_rope_index(
        input_ids, image_grid_thw, video_grid_thw, attention_mask=attention_mask
    )
output_device = input_ids.device
cpu_inputs = (
    input_ids.detach().to("cpu"),
    None if image_grid_thw is None else image_grid_thw.detach().to("cpu"),
    None if video_grid_thw is None else video_grid_thw.detach().to("cpu"),
)
attention_mask_cpu = (
    None if attention_mask is None else attention_mask.detach().to("cpu")
)
position_ids, rope_deltas = self._gr00t_original_get_rope_index(
    *cpu_inputs, attention_mask=attention_mask_cpu
)
return position_ids.to(output_device), rope_deltas.to(output_device)
```

**实现说明与边界**：移动的是 token ID、网格和 mask 等元数据，视觉/语言特征计算并未改到 CPU；每个输入张量分别搬运，不是“一次拷贝整个 batch”。纯文本等不含多模态网格的路径回退原方法，批量往返及 CPU 工作本身仍有成本。

**优点**：沿用原位置算法，把反复的设备标量控制流集中为 CPU 计算和边界上的批量搬运，减少主机与设备交替等待，同时保留原输入输出设备约定。

### 3.2 减少重复构造与主机设备传输

#### 3.2.1 插值索引与权重缓存

**问题与优化动机**：固定相机分辨率常产生相同网格，但原方法每步都生成四邻域索引和插值系数，并重新建立设备张量。这部分结果只依赖网格和设备/dtype 等元数据，不需要随训练权重一起重算。

来源：依赖侧 [fast_pos_embed_interpolate](/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:642)；仓库 [_cpu_fast_pos_embed_interpolate](../../gr00t/model/modules/qwen3_backbone.py#L142)。此处只对照缓存相关路径，CPU 化机制见 3.1.2。

**修改前：每次调用都构造列表及设备张量**

```python
idx_list = [[] for _ in range(4)]
weight_list = [[] for _ in range(4)]
# ... 省略逐网格生成和填充
idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
weight_tensor = torch.tensor(
    weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
)
pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
```

**修改后：匹配时复用索引和系数，仍然每步执行可训练查表**

```python
grid_rows = _grid_thw_as_int_rows(grid_thw)
weight = self.pos_embed.weight
cache_key = (tuple(grid_rows), weight.device, weight.dtype, self.num_grid_per_side)
cache = getattr(self, "_gr00t_fast_pos_cache", None)
if cache is not None and cache[0] == cache_key and not torch.jit.is_tracing():
    idx_tensor, weight_tensor = cache[1:]
else:
    # ... 省略索引与系数列表的构造
    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=weight.device)
    weight_tensor = torch.tensor(weight_list, dtype=weight.dtype, device=weight.device)
    if not torch.jit.is_tracing():
        self._gr00t_fast_pos_cache = (cache_key, idx_tensor, weight_tensor)
pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
```

**实现说明与边界**：键包含网格、权重设备、dtype 和基础网格边长，只有最近一条缓存；键改变就重建，tracing 时不复用或写入缓存。缓存的是静态插值元数据，不是可训练位置嵌入的输出。命中时仍需读取网格生成键，占用的设备缓存也不会凭空消失。

**优点**：在重复网格下跳过插值元数据构造及设备张量重建，减少对应 HtoD/分配开销；保留每步查表，使权重更新和梯度路径不被陈旧特征缓存替代。

#### 3.2.2 视觉旋转坐标缓存

**问题与优化动机**：视觉网格和 merge size 不变时，整数坐标顺序也不变。即使已把生成过程放到 CPU，每步重建、拼接和搬运仍是可避免的重复工作。

来源：依赖侧 [rot_pos_emb](/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:603)；仓库 [_cpu_rot_pos_emb](../../gr00t/model/modules/qwen3_backbone.py#L211)。

**修改前：每次分配并填充坐标，没有跨调用缓存**

```python
total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
offset = 0
for num_frames, height, width in grid_thw:
    # ... 省略分块坐标计算和时间维度重复，得到 coords
    num_tokens = coords.shape[0]
    pos_ids[offset : offset + num_tokens] = coords
    offset += num_tokens
embeddings = freq_table[pos_ids]
```

**修改后：复用设备上的整数坐标，频率表索引仍逐次执行**

```python
# ... 省略 grid_rows、freq_table 和 merge_size 的获取
cache_key = (tuple(grid_rows), freq_table.device, merge_size)
cache = getattr(self, "_gr00t_rot_pos_cache", None)
if cache is not None and cache[0] == cache_key and not torch.jit.is_tracing():
    pos_ids_device = cache[1]
else:
    pos_ids = []
    # ... 省略逐网格构造 CPU coords 并追加到 pos_ids
    pos_ids_device = torch.cat(pos_ids).to(freq_table.device)
    if not torch.jit.is_tracing():
        self._gr00t_rot_pos_cache = (cache_key, pos_ids_device)
return freq_table[pos_ids_device].flatten(1)
```

**实现说明与边界**：单条缓存的键为网格、频率表设备和 merge size；不匹配则替换，tracing 时不复用或写入。缓存不保存频率表或最终 RoPE 特征；`self.rotary_pos_emb(max_hw)` 仍每次调用。网格频繁变化时命中率会降低。

**优点**：重复网格下省去坐标生成、拼接及 HtoD 传输，复用的只是整数索引，避免将位置频率表计算与静态坐标生命周期混在一起。

### 3.3 局部编译与外部算子边界控制

#### 3.3.1 视觉 Block 局部编译

**问题与优化动机**：视觉 Block 反复执行相似张量运算，适合尝试局部编译；但把动态位置处理、整个模型和外部算子一起编译，会扩大图捕获及兼容性问题的影响范围。

来源：`d9bdcf9` 的骨干初始化与 `2838dd8` 的新增安装调用；当前 [安装入口](../../gr00t/model/modules/qwen3_backbone.py#L390) 和 [编译 helper](../../gr00t/model/modules/qwen3_backbone.py#L276)。基线没有这个仓库编译 helper，下面展示的是新增调用，不是假设原先存在一个 eager 包装。

```diff
 self._reset_rotary_inv_freq()
+# ... 省略新增的 CPU 位置方法安装细节
+self._apply_cpu_position_preprocessing()
+self._apply_transformers_compilation_overrides()
 self._apply_vision_patch_embed_channels_last()
```

新增编译 helper 的关键实现；同一安装方法还会调用下一项的 FlashAttention 隔离：

```python
compiled = 0
block_classes = {type(block) for block in getattr(visual, "blocks", ())}
for block_class in block_classes:
    block_call = getattr(block_class, "__call__", None)
    if block_call is None or hasattr(block_call, "_torchdynamo_orig_callable"):
        continue
    torch.compile(block_class, mode="max-autotune-no-cudagraphs")
    compiled += 1
return compiled
```

**实现说明与边界**：只针对视觉 Block 类，不是全模型编译，也不是仅包装某个实例；类级别修改可作用于已有及后续实例。`max-autotune-no-cudagraphs` 使用 autotune 但不启用 CUDA Graphs，首次编译、缓存和重编译均有成本。方法安装依赖 Transformers 布局，缺少必要位置入口时会明确报错。

**优点**：将重复视觉计算限制在较小的编译范围，为局部图优化和调度开销降低提供空间，避免直接扩大为全模型编译；重复安装时跳过已有编译包装。

#### 3.3.2 FlashAttention Dynamo 边界隔离

**问题与优化动机**：局部编译仍可能遇到外部注意力入口。需要明确哪些函数不由 Dynamo 跟踪；此外，Transformers 注册表保存的是函数对象，只改模块属性可能仍让调用者使用旧入口。

来源：`d9bdcf9` 中没有该仓库包装，`2838dd8` 新增 [隔离 helper](../../gr00t/model/modules/qwen3_backbone.py#L292) 并由 [安装方法](../../gr00t/model/modules/qwen3_backbone.py#L404) 调用。下面的 diff 是真实新增，不虚构“删除旧 disable 装饰器”的历史。

```diff
+def _apply_transformers_compilation_overrides(self) -> None:
+    """Install repository-owned Qwen3-VL compile/Dynamo overrides."""
+    compiled_blocks = _compile_qwen3_vision_block_classes(self.model.visual)
+    flash_disabled = _disable_transformers_flash_attention_for_dynamo()
+    # ... 省略 logger.debug 日志
```

新增 helper 同时更新模块属性与仍引用原函数的注册项：

```python
original = flash_attention_module.flash_attention_forward
already_disabled = hasattr(original, "_torchdynamo_disable")
disabled = original if already_disabled else torch._dynamo.disable(original)
flash_attention_module.flash_attention_forward = disabled
for implementation in ("flash_attention_2", "flash_attention_3"):
    try:
        registered = attention_functions[implementation]
    except KeyError:
        continue
    if registered is original:
        attention_functions.register(implementation, disabled)
return not already_disabled
```

**实现说明与边界**：`torch._dynamo.disable` 隔离的是编译跟踪，不是禁用 FlashAttention 算子。它引入明确的 eager 边界，可能限制跨边界融合；并非每次图中断都必然发生设备同步。对不存在的注册项跳过，对已被其他实现替换的项不强行覆盖。

**优点**：让外部注意力函数与可编译视觉 Block 各自处在合适的执行边界，并同步处理注册表的旧引用，避免“模块属性已替换、实际调用却仍走旧函数”的问题。

## 4. 当前兼容修复与数值安全

### 4.1 DiT HIP 自动 math SDPA

- **动机**：优先保证反向传播有限，避免生成已损坏但仍能保存的模型。
- **原实现问题**：当前 DTK 的 SDPA 自动调度路径曾在首步反向产生 NaN/Inf；独立算子也观察到异常梯度，而前向 loss 仍可能有限。这些现象不能单独确认具体哪个融合内核存在问题。
- **修改机制**：原来仅 NVIDIA Spark（SM 12.1）默认强制 math，当前 HIP 环境未设置覆盖时返回空上下文，由 PyTorch 自动选择后端；现在增加 HIP 判断，默认明确指定 math，并将旧上下文 API 换为 `sdpa_kernel(SDPBackend.MATH)`。其他非 HIP 设备保留 Spark 判断及原有覆盖优先级。
- **预期收益与代价**：规避当前已观察到的数值问题，不承诺提速；math 相比正常工作的融合后端可能更慢、占用更多显存。HIP 判断覆盖 ROCm，不仅是已验证的 DTK。
- **证据/待验证**：实现见 [dit.py](../../gr00t/model/modules/dit.py#L38)；显式和自动 math 均通过 100 步，尚不能视为厂商内核修复或长期稳定性证明。

**前后行为对照**（修改前以 `2838dd8` 为准）：

| 场景 | 修改前 | 修改后 |
| --- | --- | --- |
| 当前 HIP/DTK，未设置覆盖 | `_sdpa_context()` 返回 `nullcontext()`，由 PyTorch 自动选择可用 SDPA 后端 | 检测到 `torch.version.hip is not None`，仅允许 math 后端 |
| 显式 `GR00T_DIT_SDPA_MODE=math` | 强制 math，使用旧上下文 API | 强制 math，使用新上下文 API |
| 显式 `GR00T_DIT_SDPA_MODE=default` | 不额外限制后端，由 PyTorch 自动调度 | 同左；该覆盖优先于 HIP 自动 math |
| 非 HIP，未设置覆盖 | Spark SM 12.1 强制 math，其他设备自动调度 | 保留相同选择逻辑；强制 math 时使用新 API |

**关键代码差异**（片段需结合原函数理解；新 API 同时新增 `from torch.nn.attention import SDPBackend, sdpa_kernel`）：

```diff
 def _should_force_math_sdpa() -> bool:
     override = os.environ.get("GR00T_DIT_SDPA_MODE")
     if override == "math":
         return True
     if override == "default":
         return False
+    if torch.version.hip is not None:
+        return True
     return _is_spark_sm121()

 def _sdpa_context():
     if not _should_force_math_sdpa():
         return nullcontext()
-    return torch.backends.cuda.sdp_kernel(
-        enable_flash=False,
-        enable_math=True,
-        enable_mem_efficient=False,
-        enable_cudnn=False,
-    )
+    return sdpa_kernel(SDPBackend.MATH)
```

因此，这次切换准确地说是 **PyTorch SDPA 自动调度 → 明确指定 math 后端**，不能直接写成“FlashAttention → math”。修改前实际选中的实现取决于 DTK 构建、dtype、形状和 mask 等，需要对应运行的 trace 才能确认。math 仍在张量所在设备执行，不表示改用 CPU。

修改作用于 `dit.py` 的 `BasicTransformerBlock` 注意力调用，包含 32 层 DiT 及使用同一块类型的 4 层 `vl_self_attention`；没有将 Cosmos 骨干的 FlashAttention 2 改为 math，也没有修改注意力的数学定义、损失函数或训练目标。

正式环境建议取消覆盖，让 HIP 自动选择 math；不要无意设置 `default` 重新启用自动调度：

```bash
unset GR00T_DIT_SDPA_MODE
```

这是当前未提交的兼容修改，不属于 `2838dd8` 的位置计算或编译优化。

### 4.2 辅助能力与运行修复

以下列出新增或增强的观测、数值保护、运行控制与环境适配能力，不改变模型架构或训练目标。

| 新增或增强项 | 实现方式 | 作用与边界 |
| --- | --- | --- |
| 吞吐进度条 | `SamplesPerSecondProgressCallback` 按 rank0 主机时间窗口和全局 batch 估算 sample/s | 直观观察吞吐波动；不是严格稳态测速或相对基线的提速比 |
| Profiler 微调入口与采样调整 | 为已有 profiler 增加 `FinetuneConfig.enable_profiling` 配置；当前窗口为 `skip_first=1, wait=1, warmup=20, active=1, repeat=1` | 延长预热后记录一个 step；采集可能扰动调度和耗时，基础 profiler 不是本地新增能力 |
| 可选数值安全检查 | `--check-numerics` 接入配置、训练入口、Trainer 和 `NumericalMonitor` | 默认关闭；检查参数、buffer、输入、动作掩码、主要模块输出、原始 loss、梯度、优化器状态及更新后参数，同时关闭 loss 日志 NaN/Inf 过滤 |
| 异常退出与逐 rank 日志 | 更新前后 hook 记录本地结果，在一致阶段边界汇总异常；写入 `<output-dir>/numerics/rank-*.jsonl` | 检查负数分组梯度范数，异常时停止更新或阻止最终保存，退出时清理 hook；不删除此前已有 checkpoint |
| 灵活保存控制 | 新增 `save_strategy`、`save_final_model`，分别控制中途 checkpoint 和最终保存 | 支持不保存、按策略保存或仅结束保存一次；与数值检查独立，沿用的 `save_only_model` 不是新增开关 |
| CPU 完整权重扫描 | `scripts/verify_checkpoint_numerics.py` 调用 `numerics.scan_checkpoint`，通过 `--report` 输出报告 | 按分片/张量检查 NaN/Inf 和索引一致性，不构造完整模型；增加 CPU/I/O 成本，权重有限不能证明训练质量 |
| 本地模型离线加载 | 启动入口使用本地 Cosmos 路径及 `transformers_local_files_only=True`；同步修改 checkpoint 的模型、处理器配置路径并备份 | 形成离线加载链路；仅修改配置，原始模型权重未变，本地文件仍需预先准备 |
| 启动路径与视频后端适配 | `start.sh` 按自身位置定位仓库；`GROOT_DATA_ROOT` 统一模型、数据、日志及缓存路径；设置离线变量并取消强制 OpenCV | 避免依赖启动目录和继承的视频后端变量；保留的 8 卡、100 步、batch 960 是原有实验设置，不在 batch 64 验收范围内 |
| 容器加速卡访问恢复 | 补齐 `/dev/kfd`、`/dev/mkfd`、`/dev/dri` 等接口，验证 8 卡可访问 | 解决容器设备不可见问题；设备接口仍需由宿主机和容器正确提供 |
| Python/DTK 依赖适配 | 保留 Python 3.10 和厂商加速栈，补齐兼容依赖；版本清单见 runbook 3.1 | 避免覆盖厂商 PyTorch 等组件；默认上游 `uv sync` 不适合作为当前 DTK 的复装方案 |
| LIBERO 数据准备与校验 | AV1 转 H.264，校验视频属性/帧数，更新编码元数据并补齐 `meta/modality.json`；验证 Parquet 读取 | 解决数据格式及解码链路问题，不改变状态、动作语义或训练目标 |
| torchcodec CPU 解码修复 | `/workspace/scripts/torchcodec_cpu_hip_sync.patch` 将无条件 HIP 同步限制为 CUDA/HIP 设备，重新构建并替换三个扩展库 | CPU 解码不再触发无效设备同步；未替换厂商 PyTorch，换容器/重装后需重新应用并验证，H.264 硬件解码限制仍存在 |
| HTA 离线分析 | 在 `/workspace/.venvs/hta` 安装 HTA 0.5.0，解析现有 8 rank trace，生成时间、kernel、idle 和通信重叠统计 | 与 DTK 主环境隔离；单 iteration、时间戳及后端兼容性限制仍需考虑，解析成功不代表归因全部成立 |
| 回归测试补充 | `test_action_head.py`、`test_numerics.py` 覆盖后端选择、多次更新、异常拦截、更新前停止、负数范数、保存、索引扫描及双 rank 退出 | 为兼容与检查能力提供回归覆盖，具体结果和缺口见 5.2 |
| 文档与代码说明 | 补充独立贡献指南、DTK LIBERO runbook、模型 forward 中文说明和 trace 分析报告 | 便于运行、验收和性能排查；说明性修改不改变模型计算 |

数值检查支持普通 AdamW 单卡/DDP 与 DeepSpeed ZeRO-2；其他 ZeRO stage 在模型加载前拒绝。完整扫描、同步和日志可能明显降低吞吐，与 4.1 的 math 后端开销分开评估。非零梯度与参数更新是整轮累计证据，不保证每个参数每步都更新；仅保存模型而不保留优化器等状态时不能精确续训。

使用方式见 [DTK LIBERO Runbook](../../getting_started/dtk_libero_runbook.md)，性能统计与编译候选见 [Trace 分析报告](groot_torch_compile_trace_analysis.md)。

## 5. 验证证据、局限与后续工作

### 5.1 已有实验记录

报告与输出均位于 `/workspace/data/`；下表不同运行不能混用结论。

| 运行 | 已观察到的证据 | 尚不能证明 |
| --- | --- | --- |
| 历史失败 1,000 步，batch 64 | 运行结束且保存，但全量扫描发现 537 个张量含 NaN；输出 `outputs/libero-long-b64-s1000-20261014-104840` | 保存成功不等于有效训练 |
| 显式 math，8 卡 batch 64，100 步 | 125.18 秒，平均 loss 1.184575；各 rank 数值检查通过；`outputs/numerics-math-b64-s100-20261014-143034/report.json` | 未保存，未做保存重载；不能衡量每个优化项收益 |
| 自动 math，同配置 100 步 | 变量未设置；125.16 秒，平均 loss 1.184575；`outputs/numerics-auto-b64-s100-20261014-145906/report.json` | 证明当前自动选择短程有效，不代表长期验收 |
| 用户新贴的 1,000 步日志 | 日志显示完成并保存至 `outputs/libero10-finetune-20261014-163503-1653289`；335.77 秒，平均 loss 0.387624，95.304 sample/s，片段中的 loss/grad_norm 有限 | 未核实该运行全部配置、检查开关和全部权重；缺保存权重全量扫描及离线动作预测证据 |

新 1,000 步的可见末段 loss 约 0.14～0.17、梯度范数约 0.34～1.11，只能作为日志层面的正常迹象。
它与早期失败运行不是同一输出；不能用有限日志代替全量参数、优化器状态和预测检查。
100 步与新 1,000 步的 batch、worker、检查及热身条件并未建立可比性，不计算提速百分比。
三轮匹配短对照未完整完成，历史记录在 `outputs/numerics-factorial-20261014-next/report.json`，不能写成三轮均通过。

### 5.2 证据等级与测试缺口

| 等级 | 可支持的结论 | 不可替代的验证 |
| --- | --- | --- |
| 理论动机 | 某种实现可能减少同步、构造或调度 | 不能证明实际瓶颈或收益 |
| 代码实现 | 已存在 CPU 处理、缓存或编译调用 | 不能证明等价性、实际后端或提速 |
| 实际验证 | 特定配置下测试或训练的明确结果 | 不能外推其他硬件、batch、输入或所有优化项 |

当前未找到直接覆盖三个 CPU 位置改写函数、缓存命中/失效及编译边界等价性的测试。
`tests/gr00t/model/test_qwen3_rope.py` 主要覆盖上游 RoPE buffer 与解析公式，不是这些本地改写的等价性证据。
`tests/gr00t/model/test_action_head.py` 覆盖 HIP/Spark/覆盖变量分派和 CPU 多次更新，不能代替真实 HIP BF16 或多卡测试。
`tests/gr00t/experiment/test_numerics.py` 验证异常检测、更新拦截、保存及分布式一致退出，不验证位置算法性能。
清理后的重点 CPU 回归曾为 45 项通过；项目 CPU 测试仍有缺少可选依赖和 LFS 示例的失败，不宣称全项目通过。

### 5.3 后续建议（本次不执行）

1. 逐项比较原始与改写位置函数的输出及梯度，覆盖动态网格、视频、mask、merge size、设备/dtype 和 tracing。
2. 测试缓存命中、失效及权重更新后结果；对 eager/compiled 和注意力注册入口做隔离验证。
3. 固定模型、数据、种子、batch、worker、检查和保存策略，一次只切换一个因素；分开记录冷启动、稳定吞吐、峰值显存和同步次数。
4. 对新 1,000 步输出扫描全部权重，再离线重载执行动作预测，检查有限性；与历史失败输出分开管理。
5. 完成更长训练和任务成功率评估，再讨论收敛、策略质量或部署结论；不得以日志低 loss 或完成保存代替这些验收。
