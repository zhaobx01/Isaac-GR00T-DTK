# DTK 环境下 GR00T、Cosmos 与 LIBERO 运行手册

适用于当前 `/workspace/Isaac-GR00T` 容器。内容按数据与模型、DTK 兼容修复、依赖适配、训练验证与数据采集、可选模型编译、ZeRO-2 通信优化组织；容器外辅助脚本位于共享工作区 `/workspace/scripts`，不随单独克隆本仓库自动提供。

仓库同时保存了[优化适配记录](optimization/groot_dtk_optimization_record.md)、[Trace 分析报告](optimization/groot_torch_compile_trace_analysis.md)及 [torchcodec 补丁](../scripts/patches/torchcodec_cpu_hip_sync.patch)。模型、数据、训练输出、DTK 依赖与其余外部辅助脚本需单独准备；上游示例素材保留原有 Git LFS 指针，子模块保留引用，本次代码归档不镜像这些资源。

## 当前状态

- 自动 math 配置已通过 8 卡、batch 64、BF16 的 100 步数值验证：取消 `GR00T_DIT_SDPA_MODE`，8 个 worker 均确认变量未设置，耗时 125.16 秒，平均 loss 为 1.184575，与显式 math 对照一致。各 rank 的 loss、梯度、优化器状态、更新后参数和最终内存权重检查通过，存在非零梯度和参数更新。未保存模型，进程已退出。报告见 2.5；这不代表厂商默认后端、长期训练或部署验收通过。

- 环境：DTK-26.04-rc4、Python 3.10.12；容器已识别 8 张 BW1000 64G 加速卡。当前设备节点已恢复，无需重建容器。
- 资源：GR00T、Cosmos 和 LIBERO H.264 数据均已准备好，不要重复下载或把原始 AV1 snapshot 覆盖到转码目录。
- 历史流程测试：2026-10-13 的 8 卡、batch 8 两步测试完成并保存模型，但当时未检查梯度和输出权重，不能据此判定有效训练通过。日志：[libero-smoke-20261013-162451.log](../../data/logs/libero-smoke-20261013-162451.log)。
- 数值验收未通过：2026-10-14 的 8 卡、batch 64、1,000 步运行耗时 446.54 秒并保存一份模型，但最终权重中 537 个张量含 NaN。第 20 步起日志持续显示零 loss，不能解释为收敛。失败输出保留在 `/workspace/data/outputs/libero-long-b64-s1000-20261014-104840`，不可作为有效微调模型。
- CPU 全量复核：原始 GR00T 和 batch 32 两步输出的 1,031 个张量均无 NaN/Inf，报告在 `/workspace/data/logs/numerics-validation/`。后者仅证明保存权重有限，不代表长时间训练已通过。
- 保留默认关闭的逐阶段数值安全检查和 CPU 权重扫描。一次性对照调度、批次回放、细粒度诊断和 FP32/骨干后端对照已移除，历史实验记录仍保留在外部输出目录。

## 一、数据与模型

### 1.1 已准备文件

| 内容 | 路径 | 当前状态 |
| --- | --- | --- |
| GR00T N1.7 3B | `/workspace/data/checkpoints/GR00T-N1.7-3B-local-cosmos` | 完整，27 个文件，约 6.927 GB |
| Cosmos Reason2 2B | `/workspace/data/checkpoints/nvidia/Cosmos-Reason2-2B` | 完整，15 个文件，约 4.890 GB |
| LIBERO 10 H.264 | `/workspace/data/libero_10_no_noops_1.0.0_lerobot_h264` | v2.1，379 个 episode、758 个视频，约 2.434 GB；含 `meta/modality.json` |
| 原始 AV1 备份 | `/workspace/data/libero_10_no_noops_1.0.0_lerobot_av1_backup` | 保留原视频和 `meta/info.json`，约 0.626 GB |

以上数据规模使用十进制 GB。原始文件清单为 `/workspace/data/logs/model_dataset_file_inventory.tsv`。

```bash
cd /workspace/Isaac-GR00T
export DATASET_DIR=/workspace/data/libero_10_no_noops_1.0.0_lerobot_h264
export VLM_MODEL_PATH=/workspace/data/checkpoints/nvidia/Cosmos-Reason2-2B
mkdir -p /workspace/data/checkpoints/nvidia /workspace/data/logs /workspace/data/outputs
df -h /workspace
```

### 1.2 复用或下载模型

宿主机 `bw45` 已有 Cosmos 完整副本：`/public/home/zhangyf9/groot/checkpoints/nvidia/Cosmos-Reason2-2B`。容器内看不到宿主机 `/public`；当前 `/workspace` 挂载源为 `12.12.12.222:/public2/home/zhaobx/Groot`。需在宿主机执行复制：

```bash
test -d /public2/home/zhaobx/Groot || exit 1
mkdir -p /public2/home/zhaobx/Groot/data/checkpoints/nvidia/Cosmos-Reason2-2B
cp -a /public/home/zhangyf9/groot/checkpoints/nvidia/Cosmos-Reason2-2B/. \
  /public2/home/zhaobx/Groot/data/checkpoints/nvidia/Cosmos-Reason2-2B/
```

若文件缺失，GR00T 仓库公开；Cosmos 是 gated 仓库。先在 [Cosmos 模型页](https://huggingface.co/nvidia/Cosmos-Reason2-2B) 接受条款并取得账号访问资格，再创建具有读取权限的 [Hugging Face token](https://huggingface.co/settings/tokens)。不要将 token 写入仓库或分享。当前工作区下载脚本无需安装 Hugging Face CLI：

```bash
export HF_ENDPOINT=https://huggingface.co
python /workspace/scripts/download_hf_snapshot.py \
  nvidia/GR00T-N1.7-3B \
  /workspace/data/checkpoints/GR00T-N1.7-3B-local-cosmos \
  --workers 4 --direct

read -rsp 'Hugging Face read token: ' HF_TOKEN
printf '\n'
export HF_TOKEN
python /workspace/scripts/download_hf_snapshot.py \
  nvidia/Cosmos-Reason2-2B \
  /workspace/data/checkpoints/nvidia/Cosmos-Reason2-2B \
  --workers 4 --direct
unset HF_TOKEN
```

脚本也会读取 `/workspace/data/cache/huggingface/token`。`--direct` 实测下载较快；如网络必须经代理，去掉该参数。公开文件可使用 `HF_ENDPOINT=https://hf-mirror.com`；gated 仓库仍需官方账号授权。已有 CLI 时可改用 `hf download`；旧版 CLI 命令为 `huggingface-cli download`。完整下载配置、tokenizer 和预处理文件，不要只取权重分片。

### 1.3 下载与检查 LIBERO

使用仓库示例推荐的数据源 `IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot`。旧尝试 `lerobot/libero_10_no_noops` 返回 401，当前流程不使用该来源。

```bash
export HF_ENDPOINT=https://huggingface.co
python /workspace/scripts/download_hf_snapshot.py \
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \
  "$DATASET_DIR" --repo-type dataset --workers 24 --direct
```

数据集下载必须指定 `--repo-type dataset`。下载完成后至少检查 LeRobot v2 结构、episode 数、状态/动作维度和视频特征：

```bash
python - <<'PY'
import json, os
from pathlib import Path

info = json.loads((Path(os.environ["DATASET_DIR"]) / "meta/info.json").read_text())
assert info["codebase_version"].startswith("v2")
assert info["features"]["observation.state"]["shape"] == [8]
assert info["features"]["action"]["shape"] == [7]
for key in ("observation.images.image", "observation.images.wrist_image"):
    assert info["features"][key]["dtype"] == "video"
print(info["codebase_version"], info["total_episodes"], info["total_frames"])
PY
```

当前数据为 v2.1、379 episodes、101,469 frames。全部 379 个 Parquet 文件已用 PyArrow 20.0.0 读取和校验；报告：`/workspace/data/logs/pyarrow_parquet_verification.json`。v3 数据需先依照 `scripts/lerobot_conversion/README.md` 转成 v2；视频转码不能替代字段或动作语义转换。

### 1.4 转换 H.264 并补齐 modality

当前工作区下载器生成 `.download-result.json`，可用准备脚本验证下载状态、备份 AV1、转换并核对帧数/帧率/分辨率，完成后更新编码元数据和 modality：

```bash
python /workspace/scripts/prepare_libero_h264.py
```

本次 758 个视频全部成功，`examples/LIBERO/modality.json` 已复制到数据集 `meta/`。若改用 Hugging Face CLI 下载而没有下载报告，先备份完整数据，再转换并自行核对失败项；不要仅凭转换命令退出码判定完成：

```bash
cp -a "$DATASET_DIR" "${DATASET_DIR}_original_backup"
python examples/SimplerEnv/convert_av1_to_h264.py "$DATASET_DIR/videos" -j 4
cp examples/LIBERO/modality.json "$DATASET_DIR/meta/modality.json"
```

全部转码成功后还需用 `ffprobe` 检查视频，并确认 `meta/info.json` 编码字段为 H.264。不要把原始 snapshot 重新下载到 `_h264` 目录。

### 1.5 版本与权重校验

| 仓库 | revision |
| --- | --- |
| GR00T | `2fc962b973bccdd5d8ce4f67cc63b264d6886495` |
| Cosmos | `9ce19a195e423419c349abfc86fd07178b230561` |
| LIBERO | `e1a223d30b896c1613f270a2bfc63d382b3de7e1` |

关键权重 SHA-256：

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| GR00T 分片 1 | 4,990,519,232 | `8a1a1d8a33c99103c7c80c136073c5bb8bfe9ca8f7a970c93c033ea89742906d` |
| GR00T 分片 2 | 1,919,980,184 | `c3f61940deb2007ba1ad7743013b57f0f8462356151db9655175d7aca2d40661` |
| Cosmos `model.safetensors` | 4,877,470,304 | `fa5a6e6ef4fce40216b185cc48a3b24d31637ac3e2ba69c107ed1f389c1e6ede` |

```bash
sha256sum \
  /workspace/data/checkpoints/GR00T-N1.7-3B-local-cosmos/model-*.safetensors \
  /workspace/data/checkpoints/nvidia/Cosmos-Reason2-2B/model.safetensors
```

新 revision 可能产生不同哈希。下载辅助脚本会固定单次下载所解析到的 revision；需严格复现时显式指定 revision。原始下载校验记录分别在模型/数据目录的 `.download-result.json` 或 `.download-manifest.json`；转码后的视频不再匹配原始下载哈希。

## 二、DTK 兼容问题与修复

### 2.1 容器内加速卡不可见

容器必须挂载 `/dev/kfd`、`/dev/mkfd` 和 `/dev/dri`。曾出现宿主机内核已注册 8 个 DRM render 节点，但容器缺少对应设备节点、PyTorch 识别 0 卡的情况；单看 `hy-smi` 温度/功耗不能证明计算接口正常。当前通过 sysfs 主次设备号补齐缺失节点后，PyTorch 可见 8 卡，BF16 前向/反向通过，`hy-smi` 的 `Open mkfd failed` 消失。

如再次发生，先确认宿主机内核设备存在，再在具备创建设备节点权限的容器内执行：

```bash
python /workspace/scripts/restore_hcu_devices.py
hy-smi
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

恢复报告：`/workspace/data/logs/hcu_device_recovery.json`。以后重建容器可在宿主机运行 `/public/home/zhaobx/Groot/scripts/create_groot_container.sh`，然后进入新容器：

```bash
bash /public/home/zhaobx/Groot/scripts/create_groot_container.sh
docker exec -it zhaobx_groot_cosmos_128g_v2 bash
```

创建前确认设备节点存在；脚本默认创建 `_v2` 容器，不删除旧容器。当前容器无需重建。新容器不会继承旧容器 site-packages 的修改，torchcodec 补丁和兼容依赖需按本手册重新安装/验证。容器内没有 Docker CLI 或 socket。

### 2.2 本地 Cosmos 加载与离线运行

GR00T checkpoint 的 `config.json` 中 `model_name` 和 `processor_config.json` 的 `processor_kwargs.model_name` 已改为 `/workspace/data/checkpoints/nvidia/Cosmos-Reason2-2B`；原文件备份为 `config.json.before-local-loading` 和 `processor_config.json.before-local-loading`。权重未修改。`launch_finetune.py` 已设 `config.training.transformers_local_files_only = True`，`start.sh` 也会设置本地模型路径和离线变量。

手动启动或运行其他入口时：

```bash
export VLM_MODEL_PATH=/workspace/data/checkpoints/nvidia/Cosmos-Reason2-2B
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset VIDEO_DECODER_BACKEND
```

不要把 `VLM_MODEL_PATH` 改回远端 Cosmos ID；需下载模型时再临时取消离线变量。模型离线加载记录见 `/workspace/data/cache/dtk-finetune-dependencies/local-load-verification.json`。

### 2.3 启动脚本路径与视频后端

`start.sh` 已按脚本自身位置解析仓库根目录，并使用 `GROOT_DATA_ROOT`（默认 `/workspace/data`）构造模型、数据、日志和输出路径；从其他当前目录启动不再依赖相对路径。脚本会 `unset VIDEO_DECODER_BACKEND`，避免环境变量将 torchcodec 视频读取误导到 OpenCV 后端。当前 OpenCV headless 版本为 4.11.0.86。

注意：当前 `start.sh` 固定配置 8 卡、`MAX_STEPS=100` 和 `GLOBAL_BATCH_SIZE=960`，是已有实验脚本，不是本次冒烟命令，也不等同于下面的 10,000 步正式训练配置。

### 2.4 torchcodec CPU 解码的 `invalid device ordinal`

厂商 torchcodec 的 `SingleStreamDecoder::decodeAVFrame()` 原先在 CPU 解码时也无条件调用 `hipStreamSynchronize(0)`，导致没有可用设备时抛出 `invalid device ordinal`。已将同步限制为 CUDA/HIP 设备，并使用本地源码重新编译、备份和替换三个 native 扩展库；未替换 DTK PyTorch 或 torchcodec 版本。

- 补丁：`/workspace/scripts/torchcodec_cpu_hip_sync.patch`
- 构建与备份：`/workspace/data/cache/torchcodec-hip-sync-fix/`（含 MIOpen `librt.so` 和 RPP 缺失声明的局部构建适配）
- 安装清单：`/workspace/data/cache/torchcodec-hip-sync-fix/installation_manifest.json`
- 验证报告：`/workspace/data/logs/torchcodec_cpu_fix_verification.json`

验证覆盖双相机的 H.264/AV1 片段、乱序/重复/末尾索引，以及按索引、时间戳和整段读取；CPU 解码在 0 张可见卡时通过。当前无损 H.264 文件的 GPU 解码因硬件不支持其 profile 而报 `profile not support / Invalid sps`，因此只将 CPU 解码列为已验证。重装 torchcodec 或换容器后重新应用并验证补丁：

```bash
PYTHONPATH=/workspace/Isaac-GR00T PYTHONDONTWRITEBYTECODE=1 \
python /workspace/scripts/verify_torchcodec_cpu.py \
  --report /workspace/data/logs/torchcodec_cpu_fix_verification.json
```

### 2.5 训练数值异常：已知证据与未完成项

当前 DTK 默认 SDPA 路径曾在第一次反向、优化器更新之前产生 NaN/Inf；关闭视觉编译或仅替换 Cosmos 注意力未能规避。独立算子对照中，默认后端的 12 组反向有 2 组出现非有限值，math 均有限，报告为 `/workspace/data/logs/numerics-validation/sdpa-live-next.json`。这是后端兼容性证据，不代表已修复厂商内核的具体缺陷。

`gr00t/model/modules/dit.py` 在 HIP 构建的 PyTorch 上默认使用 `sdpa_kernel(SDPBackend.MATH)`。`GR00T_DIT_SDPA_MODE=math` 可显式启用，`default` 会覆盖兼容修复并重新使用默认后端；正常运行建议取消该变量。此判断覆盖 HIP/ROCm，不仅限于 DTK。

| 100 步验证 | 结果 | 报告路径（`/workspace/data/outputs/` 下） |
| --- | --- | --- |
| 显式 math | 8 卡、batch 64、125.18 秒、平均 loss 1.184575；8 rank 数值检查通过 | `numerics-math-b64-s100-20261014-143034/report.json` |
| 未设置覆盖变量，自动 math | 同配置、125.16 秒、平均 loss 1.184575；8 rank 数值检查通过 | `numerics-auto-b64-s100-20261014-145906/report.json` |

两轮均未保存模型；尚未验证保存后重载推理、修复后的 1,000 步或模型质量。历史 1,000 步输出有 537 个 NaN 张量，不可作为有效模型；完整扫描见 `/workspace/data/logs/numerics-validation/failed-long-checkpoint.json`。

历史短对照保留在 `/workspace/data/outputs/numerics-diagnosis-live-v2-20261014/`、`numerics-diagnosis-live-v3-20261014/`、`numerics-diagnosis-live-v4-20261014/`。三轮匹配对照提前中止，部分结果见 `/workspace/data/outputs/numerics-factorial-20261014-next/report.json`，不能当作完整三轮验收。首次诊断的浮点采样索引越界已改为整数索引，不属于原训练根因。

一次性实验脚本及对照参数已从仓库清理，历史日志保持不变。安全检查说明见 4.4，100 步验证命令见 4.2；不能只根据退出码、保存成功或日志低 loss 判断训练有效。

## 三、依赖项与版本适配

### 3.1 Python 3.10 / DTK 兼容版本

项目上游元数据要求 Python 3.12，且声明的 PyTorch 2.9、torchcodec 0.8 等版本与本容器 DTK-26.04 厂商栈不同。当前保留 Python 3.10.12 和厂商 PyTorch 2.10.0、torchvision 0.25.0、torchcodec 0.10.0、DeepSpeed 0.18.4、FlashAttention 2.8.4、Triton 3.5.0；不要运行默认 `uv sync` 或安装上游 PyTorch wheel。

| 用途 | 当前兼容版本 |
| --- | --- |
| 数值/数据 | NumPy `1.26.3`、Pandas `2.3.3`、PyArrow `20.0.0`、SciPy `1.15.3`、scikit-image `0.25.2` |
| 图像 | OpenCV headless `4.11.0.86`、Albumentations `1.4.18` |
| GR00T 入口/模型 | tyro `0.9.17`、OmegaConf `2.3.0`、Transformers `4.57.3`、HF Hub `0.35.3`、tokenizers `0.22.1`、safetensors `0.6.2` |
| 训练工具 | diffusers `0.35.1`、PEFT `0.17.1`、Accelerate `1.12.0`、dm-tree `0.1.9`、pytest `8.3.5` |

本次解析并离线安装了 43 个包，NumPy 从 1.24.3 调整为 1.26.3；同时解决 `tyro` 训练入口和 FlashAttention 导入所需 `pytest` 缺失。保留 DTK 加速栈、PyArrow 20.0.0 和 Pandas 2.3.3。GR00T 使用自带数据加载器，不需要为当前训练额外安装 Hugging Face `datasets`、ONNX 或 TensorRT。OpenCV 4.11 避免 4.12 对 NumPy 2.x 的要求。

已缓存 wheel 和约束文件可供同一基础环境复用。先 dry-run 确认不会替换厂商包，再离线安装：

```bash
deps=/workspace/data/cache/dtk-finetune-dependencies
python -m pip install --dry-run --no-index --find-links "$deps/wheels" \
  -r "$deps/install-lock.txt" -c "$deps/constraints.txt"
python -m pip install --no-index --find-links "$deps/wheels" \
  -r "$deps/install-lock.txt" -c "$deps/constraints.txt"
```

安装前后清单、锁定文件和 NumPy 回退 wheel 均在 `$deps`。若基础镜像版本不同，先重新核对约束。容器 site-packages 不由工作区挂载持久化；重建容器要使用包含这些修改的镜像或重新安装并验收。全环境 `pip check` 仍会报告预装 LeRobot 缺少的若干可选包、版本差异，以及 PyGObject 缺少 `pycairo`；这些不在当前 GR00T 数据/训练依赖范围内。

## 四、训练启动、验证与数据采集

### 4.1 已完成的验收

| 检查 | 结果 |
| --- | --- |
| 训练入口离线 `--help` | 通过 |
| 依赖安装后的回归测试 | 39 项通过；`/workspace/data/cache/dtk-finetune-dependencies/pytest.xml` |
| 后续轻量 CPU 测试 | 68 项通过，2.48 秒；覆盖安全 YAML、初始动作与统计缓存、动作格式；`/workspace/data/cache/dtk-finetune-dependencies/pytest-extra.xml` |
| 设备与 DTK 算子 | 8 卡可见，BF16 前向/反向及 NumPy/Torch 互操作通过 |
| 模型/数据 | 3.144B 参数权重离线加载无缺失或错配；379 个 Parquet、101,469 行通过；处理器和 collator 样本通过 |
| 视频 | 双相机 CPU 解码通过；torchcodec 修复报告见上文 |
| 历史 8 卡流程测试 | 两步完成与保存；当时未检查数值，不能作为有效训练验收 |
| 原始权重 / batch 32 两步输出 | CPU 全量扫描均无 NaN/Inf；报告见 `/workspace/data/logs/numerics-validation/` |
| batch 64、1,000 步 | 流程完成，但动作头权重含 NaN，数值验收失败 |
| 历史数值诊断 CPU 回归 | 清理前 69 项通过；含当时的临时诊断功能，不能作为当前测试计数；`/workspace/data/logs/numerics-validation/pytest.xml` |
| 精简后的安全检查与动作头 CPU 回归 | 45 项通过，含 HIP 自动选择、NaN/Inf 注入、更新前拦截、双 rank 一致退出、失败不保存及多步更新；本轮未启动设备训练 |
| 精简后的项目 CPU 测试 | 628 passed、41 skipped、34 deselected、5 failed、7 collection errors；不是全套通过 |

本轮项目失败/收集错误涉及缺少 `gymnasium`、`msgpack_numpy`、`tensorrt`；4 个 torchcodec 视频测试使用的示例 MP4 实为 Git LFS 指针。未安装可选依赖或下载示例，完整结果与逐文件清理清单位于 `/tmp/groot-cleanup.5k4iVN/`，该临时备份目录不保证长期保存。

训练日志中出现 Flash Attention / memory-efficient attention 不可用等提示，但未阻断本次训练；首次 Triton autotune 也会增加启动耗时。不能据此声称相关加速内核已启用。

### 4.2 100 步训练验证

当前仅通过 8 卡、batch 64 的 100 步数值验证，不代表长期训练或部署通过。启动前人工确认设备可访问、各目标卡显存不超过 15%、利用率不超过 5%，并确认端口空闲。普通训练入口没有自动资源调度保护；不要停止其他任务或自动修复设备节点。以下是清理后的复现命令，本轮仅校验命令，没有重新执行设备训练。

请使用 Bash：数组内可逐行注释，不要改成在反斜杠续行末尾插入注释。本例启用数值检查，不保存模型；取消环境变量后由 HIP 分支自动选择 math SDPA。

```bash
cd /workspace/Isaac-GR00T  # 从仓库根目录调用训练脚本
export VLM_MODEL_PATH=/workspace/data/checkpoints/nvidia/Cosmos-Reason2-2B  # 本地 Cosmos
export HF_HOME=/workspace/data/cache/huggingface  # 复用本地 Hugging Face 缓存
export HF_HUB_OFFLINE=1  # 禁止 Hugging Face Hub 联网
export TRANSFORMERS_OFFLINE=1  # Transformers 仅使用本地文件
export NUM_GPUS=8  # 启动 8 个训练进程，每卡 1 个
export MAX_STEPS=100  # 只做 100 步验证
export GLOBAL_BATCH_SIZE=64  # 梯度累积为 1 时，每卡 batch 为 8
export DATALOADER_NUM_WORKERS=0  # 验证时不启用数据加载子进程
export USE_WANDB=0  # 关闭在线实验跟踪
export MASTER_PORT=29685  # 分布式通信端口，启动前确认未被占用
unset GR00T_DIT_SDPA_MODE  # 不强制覆盖，让 HIP 自动选择 math SDPA
unset VIDEO_DECODER_BACKEND  # 不强制切换到 OpenCV 视频后端
output_dir="/workspace/data/outputs/check-$(date +%Y%m%d-%H%M%S)-$$"  # 独立输出目录
script_args=(  # 以下参数由 finetune.sh 解析
  --base-model-path /workspace/data/checkpoints/GR00T-N1.7-3B-local-cosmos  # 原始 GR00T 权重
  --dataset-path /workspace/data/libero_10_no_noops_1.0.0_lerobot_h264  # 已转码的数据集
  --embodiment-tag LIBERO_PANDA  # 选择 LIBERO Panda 的模态与动作配置
  --output-dir "$output_dir"  # 保存配置和检查日志，不保存权重
  --save-only-model  # 若启用保存，仅存模型，不存优化器状态
)
training_args=(  # 以下参数通过 -- 传给 launch_finetune.py
  --check-numerics  # 开启 NaN/Inf 检查，异常立即一致退出
  --gradient-accumulation-steps 1  # 每批反向后更新，不额外扩大有效 batch
  --save-strategy no  # 禁止中途 checkpoint
  --no-save-final-model  # 禁止训练结束时保存模型
)
bash examples/finetune.sh "${script_args[@]}" -- "${training_args[@]}"  # 启动验证
```

### 4.3 正式训练示例（尚未验收）

完整 10,000 步训练仍未运行，应先完成长期数值和保存重载验收。以下示例使用 8 卡、batch 32（每卡 4），只在结束时保存一次模型。数值检查可选，默认关闭以避免额外扫描和同步开销；首次长训练建议将 `CHECK_NUMERICS` 改为 `1`。

```bash
cd /workspace/Isaac-GR00T
export VLM_MODEL_PATH=/workspace/data/checkpoints/nvidia/Cosmos-Reason2-2B
export HF_HOME=/workspace/data/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NUM_GPUS=8
export MAX_STEPS=10000  # 正式训练目标步数，尚未验收
export GLOBAL_BATCH_SIZE=32  # 梯度累积为 1 时，每卡 batch 为 4
export DATALOADER_NUM_WORKERS=4  # 每个训练进程使用 4 个数据加载 worker
export USE_WANDB=0
export MASTER_PORT=29685
unset GR00T_DIT_SDPA_MODE
unset VIDEO_DECODER_BACKEND
CHECK_NUMERICS=0  # 改为 1 启用数值检查；会增加扫描与同步开销，降低训练吞吐
output_dir="/workspace/data/outputs/libero10-finetune-$(date +%Y%m%d-%H%M%S)-$$"
script_args=(
  --base-model-path /workspace/data/checkpoints/GR00T-N1.7-3B-local-cosmos
  --dataset-path /workspace/data/libero_10_no_noops_1.0.0_lerobot_h264
  --embodiment-tag LIBERO_PANDA
  --output-dir "$output_dir"
  --save-only-model  # 不保存优化器、调度器和 RNG 状态，不能精确续训
)
training_args=(
  --gradient-accumulation-steps 1  # 保持有效 batch 为 32
  --save-strategy no  # 不产生中途 checkpoint，SAVE_STEPS 不再决定保存频率
  --save-final-model  # 正常结束时仅保存一次最终模型
)
if [[ "$CHECK_NUMERICS" == 1 ]]; then  # 可选启用数值检查
  training_args+=(--check-numerics)
fi
bash examples/finetune.sh "${script_args[@]}" -- "${training_args[@]}"
```


### 4.4 可选数值检查

HIP 自动 math 是后端兼容修复，`--check-numerics` 是检测异常的安全检查，两者不能互相替代。不加该参数时检查默认关闭；`CHECK_NUMERICS` 仅是上面 Bash 示例的局部开关，单独 export 它不会让训练器启用检查。

启用后检查加载/准备后的权重和缓冲区、输入、动作掩码、主要模块输出、原始 loss、梯度、优化器状态和更新后参数，同时关闭 loss 日志的 NaN/Inf 过滤。结束时检查全部内存权重、非零梯度及实际参数变化；后两项为整轮累计证据，不保证每个参数或每一步都更新。检查会增加扫描、同步和日志开销。

**性能备注：** 当前是逐步完整检查，不仅检查 loss，还扫描大规模梯度、优化器状态和参数，并进行设备同步及跨卡异常汇总，可能明显降低训练吞吐。短程验证建议开启；确认稳定后的正式训练可关闭，并在保存后运行 CPU 全量权重扫描，但后者不能替代训练期间的异常检测。目前没有同配置下开启/关闭检查的测速对照，不能给出准确降速比例；此前约 125 秒的 100 步结果包含完整检查。关闭检查不会关闭 HIP 自动 math，math 后端自身的性能影响应与检查开销分开评估。

当前支持普通 AdamW/DDP 和 DeepSpeed ZeRO-2；其他 ZeRO stage 在模型加载前拒绝。DeepSpeed 更新发生在 Accelerate 的 backward 内，检查直接接入更新前后；hook 仅记录本地结果，各 rank 在一致边界汇总异常并退出，不继续最终保存。负数分组范数也会被拒绝，防止异常哨兵掩盖故障；已有的中途 checkpoint 不会被自动删除。

逐 rank 记录位于 `<output-dir>/numerics/rank-*.jsonl`，不捕获训练数据或 RNG。确认不再需要排查证据后可删除这些输出日志；不要单独删除 `gr00t/experiment/numerics.py`，它还被配置、训练器、扫描工具和测试引用，移除需要成组调整。

CPU 扫描可独立运行，不构建模型或初始化加速卡。保存训练模型后，将示例中的原始模型目录替换为实际输出目录；报告目录需预先存在：

```bash
cd /workspace/Isaac-GR00T  # 使相对脚本路径有效
export PYTHONPATH=.  # 允许扫描脚本导入仓库模块
export OMP_NUM_THREADS=1  # 限制 CPU 线程开销
scan_args=(  # 扫描全部 safetensors 分片，并检查索引一致性
  /workspace/data/checkpoints/GR00T-N1.7-3B-local-cosmos  # 改为待验收的训练输出目录
  --report /workspace/data/logs/numerics-validation/base-checkpoint.json  # 写入有限值报告
)
python scripts/verify_checkpoint_numerics.py "${scan_args[@]}"  # 不加载完整模型
```

### 4.5 保存策略与剩余验收

| 参数组合 | 行为 |
| --- | --- |
| `--save-strategy no --no-save-final-model` | 不保存中途或最终模型；100 步验证采用此组合 |
| `--save-strategy no --save-final-model` | 正常结束仅保存一次模型；正式示例采用此组合 |
| 脚本参数 `--save-only-model` | 仅保存模型，不保存优化器/调度器/RNG，不能精确恢复训练 |

数值检查与保存策略独立：启用检查不会自动禁用正常保存，关闭检查也不会关闭 HIP math。最终模型约 6.5 GB，配置与日志仍会写入输出目录。

后续验收按单卡 20 步、8 卡 batch 32 的 50 步、batch 64 的 100 步和 1,000 步推进；任一失败即停止。仍待验证厂商默认 SDPA 的具体内核缺陷、兼容配置的长期稳定性及保存后重载动作预测；完整 10,000 步训练、策略服务和机器人/仿真闭环均未验收。历史两步测试、含 NaN 的长训练输出以及本次 100 步结果都不能证明成功部署。

## 五、数据分析

按问题选择采集方式：Profiler 观察执行时间线，编译诊断日志解释图断与重新编译的原因。

| 数据 | 主要用途 | 开启方式 | 分析工具 |
| --- | --- | --- | --- |
| Profiler Trace | 查看算子、kernel、搬运、通信和设备活动空档 | `--enable-profiling` | profile-model、HTA |
| 图断与重编译日志 | 定位图断源码、guard 失败及重编译原因 | `TORCH_LOGS="graph_breaks,recompiles"`；可选 `TORCH_TRACE` | debug-graph-breaks；tlparse 可生成结构化报告 |

建议分两次独立采集，避免日志写入影响 Profiler 时间线。两类采集均有额外开销，耗时不作为正式测速结果；性能测试方法见 6.3。

### 5.1 抓取 Profiler Trace

#### 5.1.1 用途与实现

Trace 记录 CPU 与设备事件的时间和关联信息，用于分析前向、反向、搬运、通信及执行空档。rank 0 同时打印聚合算子统计表（`key_averages()`，按 `self_cuda_time_total` 排序）。

Profiler 默认关闭，`--enable-profiling` 经 `examples/finetune.sh` 透传到 `launch_finetune.py`。[experiment.py](../gr00t/experiment/experiment.py) 创建 `torch.profiler.profile`，[ProfCallback](../gr00t/experiment/trainer.py) 在每个优化器步结束后调用 `prof.step()`；窗口完成时，rank 0 打印算子统计表，所有 rank 导出 Trace JSON。窗口配置见 [profiling.py](../gr00t/experiment/profiling.py)。用 `tee` 保存训练输出时，算子统计表也会写入 `train.log`。

#### 5.1.2 采集窗口

默认值定义在 [finetune_config.py](../gr00t/configs/finetune_config.py#L207-L212) 和 [training_config.py](../gr00t/configs/training/training_config.py#L129-L134)，`launch_finetune.py` 负责拷贝到训练配置：

```python
schedule=torch.profiler.schedule(skip_first=1, wait=1, warmup=20, active=1, repeat=1)
```

即跳过 1 步、等待 1 步、预热 20 步，正式记录 1 步，只采集一轮，通常在约第 23 个优化器步导出，文件名以回调实际记录为准。只训练 20 步会在 warmup 阶段结束，正常退出也不会导出 Trace。默认仅 1 个 active step，结论应标明单步采样限制。

可在 `training_args` 中追加 `--profiler-skip-first`、`--profiler-wait`、`--profiler-warmup`、`--profiler-active`、`--profiler-repeat` 覆盖窗口。当前启用 CPU/CUDA activity 和 `with_stack=True`（可用 `--no-profiler-with-stack` 关闭）；`record_shapes` 与 `profile_memory` 尚未开启。

#### 5.1.3 完整采集命令

先人工确认 8 卡空闲及端口未占用，规则见 4.2。以下脚本可独立执行：8 卡、全局 batch 32、每卡 4、训练 40 步（覆盖 `warmup=20` 窗口），关闭数值检查、W&B 和全部模型保存，每个 rank 都采集；使用当前自动 math 配置，不切回存在数值风险的默认 SDPA。

```bash
cd /workspace/Isaac-GR00T
set -o pipefail  # 经过 tee 后仍保留训练失败的退出状态
export VLM_MODEL_PATH=/workspace/data/checkpoints/nvidia/Cosmos-Reason2-2B
export HF_HOME=/workspace/data/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NUM_GPUS=8 MAX_STEPS=40 GLOBAL_BATCH_SIZE=32 USE_WANDB=0
export DATALOADER_NUM_WORKERS=4  # 与正式训练的数据加载配置一致
export MASTER_PORT=29686  # 启动前确认该端口空闲
unset GR00T_DIT_SDPA_MODE VIDEO_DECODER_BACKEND
output_dir="/workspace/data/outputs/profile-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$output_dir"  # 提前创建 tee 的日志目录
script_args=(
  --base-model-path /workspace/data/checkpoints/GR00T-N1.7-3B-local-cosmos
  --dataset-path /workspace/data/libero_10_no_noops_1.0.0_lerobot_h264
  --embodiment-tag LIBERO_PANDA
  --output-dir "$output_dir"
  --save-only-model
)
training_args=(
  --enable-profiling  # 启用现有 trace 导出逻辑
  --no-check-numerics  # 避免数值扫描和同步污染性能采样
  --gradient-accumulation-steps 1  # profiler 每步对应一次优化器更新
  --save-strategy no  # 不产生中途 checkpoint
  --no-save-final-model  # 不保存最终权重
)
bash examples/finetune.sh "${script_args[@]}" -- "${training_args[@]}" 2>&1 | tee "$output_dir/train.log"
```

此命令会实际训练 40 步。当前 DTK 环境已有成功导出八 rank 设备 Trace 的记录（见 5.1.4 归档入口），但每次采集仍需核实 rank、active 步数及设备 kernel 事件，不能仅凭 JSON 存在判断成功。更换卡数或 batch 做比较时，保持每卡 batch、数据加载等条件可比；单卡与多卡 DeepSpeed 路径不同，不能直接等同。

#### 5.1.4 输出与检查

输出位于 `<output-dir>/profiling/trace_rank_<rank>_iter_<step>.json`，8 卡通常各生成 1 份（rank 0～7）。退出后在同一 Bash 会话中查看：

```bash
printf '本次输出目录：%s\n' "$output_dir"
find "$output_dir/profiling" -maxdepth 1 -type f -name 'trace_rank_*.json' -printf '%f  %s bytes\n'
grep -nE 'Starting training with profiling|Trace saved|Traceback|Error' "$output_dir/train.log" || true
```

rank 0 的 `key_averages()` 表位于终端和 `train.log`，表示算子自身的累计设备耗时（不含子调用）；Trace JSON 保留各 rank 的完整采样时间线。

已有归档和复算入口：

| 入口 | 内容 |
| --- | --- |
| [Baseline Trace 清单](../../data/benchmarks/baseline/reference/metadata/trace_sha256.json) | 同一次采集的八 rank 原始 JSON |
| [Baseline Trace 分析](../../data/benchmarks/baseline/reference/trace-analysis/README.md) | profile-model 方法的算子、kernel、搬运统计及 HTA 结果；附可指定输入、输出目录的复算命令 |
| [DiT model + VL Trace 与分析入口](../../data/benchmarks/optimizations/model-vl/trace-analysis/README.md) | 组合方案的 Trace 位置与分析脚本说明 |

Trace 与搬运统计不等于显存快照；窗口空档与 HTA idle 定义不同，需分别标明分母和计算口径。统计方法见 5.3。

### 5.2 图断与重编译诊断

#### 5.2.1 用途与配置

`torch.compile` 捕获计算图，并用 guard 检查已编译代码能否用于当前调用。以下日志用于解释编译范围及训练中的编译等待：

| 概念 | 含义 | 诊断作用 |
| --- | --- | --- |
| Graph break | Dynamo 遇到无法捕获的代码并结束当前图，`fullgraph=False` 下通常执行该段后继续捕获 | 定位图被切开的源码位置和原因 |
| Guard 失败 | 当前调用不满足已有图的适用条件，如 shape、dtype 或对象状态变化 | 判断变化是否使已有编译版本失效 |
| Recompile | 无可复用版本时重新捕获并编译 | 解释首次调用耗时尖峰；命中重编译上限后，相关路径回退到 eager |

启动训练前设置环境变量，`torchrun` 启动的各 rank 会继承：

| 环境变量 | 输出与用途 |
| --- | --- |
| `TORCH_LOGS="graph_breaks,recompiles"` | 将图断位置、重编译及相关 guard 失败原因写入训练输出；通过 `2>&1 \| tee` 一并保存到 `train.log` |
| `TORCH_TRACE=<目录>`（可选） | 保存逐 rank 的结构化编译诊断日志，供后续 tlparse 分析 |

只需文本诊断时设置 `TORCH_LOGS` 即可。下面的 `CAPTURE_TORCH_TRACE` 是示例脚本里的 Bash 变量，不是项目配置接口：默认 `0`，改为 `1` 后额外开启结构化日志。生成 tlparse 报告需要另行调用工具，仅设置 `TORCH_TRACE` 不会自动生成 HTML。

#### 5.2.2 完整采集命令

以 5.1.3 为基础，保留 8 卡、全局 batch 32、40 步和每 rank 4 个数据 worker，开启整体 DiT 与 4 个 VL 块编译；关闭 Profiler，独立采集编译诊断，沿用当前缓存配置。

```bash
cd /workspace/Isaac-GR00T
set -o pipefail
export VLM_MODEL_PATH=/workspace/data/checkpoints/nvidia/Cosmos-Reason2-2B
export HF_HOME=/workspace/data/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NUM_GPUS=8 MAX_STEPS=40 GLOBAL_BATCH_SIZE=32 USE_WANDB=0
export DATALOADER_NUM_WORKERS=4
export MASTER_PORT=29686
unset GR00T_DIT_SDPA_MODE VIDEO_DECODER_BACKEND TORCH_COMPILE_DEBUG
output_dir="/workspace/data/outputs/libero-compile-diagnostics-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$output_dir"
export TORCH_LOGS="graph_breaks,recompiles"  # 保存文本诊断日志
CAPTURE_TORCH_TRACE=0  # 改为 1，额外保存逐 rank 结构化日志
if [[ "$CAPTURE_TORCH_TRACE" == 1 ]]; then
  export TORCH_TRACE="$output_dir/torch_trace"
  mkdir -p "$TORCH_TRACE"
else
  unset TORCH_TRACE  # 默认关闭，避免继承之前的采集目录
fi
script_args=(
  --base-model-path /workspace/data/checkpoints/GR00T-N1.7-3B-local-cosmos
  --dataset-path /workspace/data/libero_10_no_noops_1.0.0_lerobot_h264
  --embodiment-tag LIBERO_PANDA
  --output-dir "$output_dir"
  --save-only-model
)
training_args=(
  --no-compile-dit-blocks
  --compile-dit-model  # 编译整个动作 DiT
  --no-compile-text-layers
  --no-compile-text-layer-forwards
  --compile-vl-self-attention-blocks  # 编译 4 个 VL 自注意力块
  --compile-mode default
  --no-enable-profiling
  --no-check-numerics
  --gradient-accumulation-steps 1
  --save-strategy no
  --no-save-final-model
)
bash examples/finetune.sh "${script_args[@]}" -- "${training_args[@]}" 2>&1 | tee "$output_dir/train.log"
```

#### 5.2.3 输出与分析

在同一 Bash 会话中查看结果；换终端后先将 `output_dir` 设为实际输出目录：

```bash
printf '诊断输出目录：%s\n' "$output_dir"
tail -n 60 "$output_dir/train.log"
grep -nE 'Graph break|graph break|Recompiling|guard failure|recompile_limit|cache_size_limit' "$output_dir/train.log" || true
if [[ -d "$output_dir/torch_trace" ]]; then
  find "$output_dir/torch_trace" -maxdepth 1 -type f -name '*.log' -printf '%f  %s bytes\n'
fi
```

文本日志用于查看图断源码位置与 guard 失配原因；开启结构化采集时，通常生成 `torch_trace/dedicated_log_torch_trace_rank_*.log`，以实际文件为准。日志也可能来自已有视觉编译路径，需结合源码判断归属，不能全部归因于新增 DiT 或 VL 编译；多 rank、同一位置的重复日志不能直接累加为独立图断或重编译次数。

将 `train.log`、可选的结构化日志或 tlparse 报告、实际编译开关及源码路径交给 debug-graph-breaks，分析图断、guard 失败和重编译原因。缺少对应日志或覆盖不足时，标记专项分析未完成，源码推测不能代替运行证据。

诊断会增加日志与执行开销，耗时不用于正式测速；40 步为目标步数，实际覆盖以完成步数和日志为准。没有匹配日志不能单独证明完整捕获或全程无重编译。正式测速或另行采集 Profiler 前，清除诊断变量：

```bash
unset TORCH_LOGS TORCH_TRACE
```

### 5.3 使用 profile-model 与 HTA 分析

#### 5.3.1 工具分工

两者读取同一批 Profiler Trace，可在 CPU 上离线分析，无需重新训练或加载模型权重。

| 工具 | 作用 | 本环境的入口 |
| --- | --- | --- |
| profile-model | 指导热点与时间线分析的 Skill，关注算子、设备活动间隙、搬运和 rank 差异 | `analyze_trace.py` 用 Pandas 按其方法生成统计；不是名为 `profile-model` 的 CLI |
| HTA | Python 分析库，计算时间分类、kernel 分类、按流等待和通信计算重叠 | `analyze_hta.py` 调用 HTA，并记录各项分析状态 |

#### 5.3.2 离线复算

默认分析最新的 DTK 26.04 baseline；分析 5.1 的新采样时，替换 `trace_dir`。输入应属于同一次采集，每个 rank 一份 `trace_rank_*.json`，并包含设备 kernel 和有效 active step。

主 Python 环境需有 NumPy、Pandas；HTA 使用现有独立环境。脚本位于仓库外的归档目录，只读原始 Trace，拒绝覆盖已有结果目录。

```bash
set -euo pipefail
archive=/workspace/data/benchmarks/baseline/dtk2604
trace_dir="$archive/profiler/traces"
analysis_dir="/workspace/data/outputs/trace-analysis-$(date +%Y%m%d-%H%M%S)-$$"
mkdir "$analysis_dir"
mkdir "$analysis_dir/logs"

OMP_NUM_THREADS=1 python "$archive/trace-analysis/scripts/analyze_trace.py" \
  --trace-dir "$trace_dir" --output-dir "$analysis_dir/profile-model" \
  2>&1 | tee "$analysis_dir/logs/profile-model.log"
OMP_NUM_THREADS=1 /workspace/.venvs/hta/bin/python "$archive/trace-analysis/scripts/analyze_hta.py" \
  --trace-dir "$trace_dir" --output-dir "$analysis_dir/hta" \
  2>&1 | tee "$analysis_dir/logs/hta.log"
printf '分析输出：%s\n' "$analysis_dir"
```

#### 5.3.3 结果阅读

先检查采样 rank、步数和 HTA 状态，再阅读主要结果：

| 输出（相对本次分析目录） | 内容 |
| --- | --- |
| `profile-model/rank-XX-*.csv` | 各 rank 的 kernel、CPU 算子、runtime 和搬运热点；时长列为 us，memory 表不是显存快照 |
| `profile-model/trace_summary.csv`、`trace_steps.csv` | 采样窗口、设备活动并集与间隙；汇总时长为 ms |
| `hta/temporal-0.csv`、`kernel-*.csv`、`idle-0.csv`、`overlap-0.csv` | 时间分类、kernel 分类、按流等待、通信计算重叠 |
| `hta/status.json`、`logs/` | 已解析 rank、各项成功/失败状态及执行日志；失败项应标记未完成 |

最新 [DTK 26.04 分析结果](../../data/benchmarks/baseline/dtk2604/trace-analysis/README.md)覆盖八 rank，每 rank 连续 3 个 active step：

| 指标 | 已有结果 |
| --- | --- |
| rank 0 设备活动间隙占比 | 24.289% |
| rank 0 HTA temporal idle | 27.1% |
| HTA kernel 累计分类占比 | 计算 55.8%、通信 40.0%、memory 4.2% |

设备活动间隙为采样设备窗口减去 kernel、memcpy、memset 的活动并集；HTA idle 使用自身的窗口与过滤规则，两者不能互换。kernel 累计时长可能包含并行活动，其占比不是训练墙钟占比。样本仅有 3 步，正式吞吐比较仍使用 6.3 中关闭 Profiler 的独立测速。

详细文件与复算说明见上述归档索引。其中 HTA 状态已归档到 `trace-analysis/logs/hta-status.json`；新复算仍写入本次输出的 `hta/status.json`。

#### 5.3.4 生成分析报告

上述脚本生成统计表与日志；结合源码解释瓶颈的专项报告 `analysis_report.md` 可用下面的提示词生成。它与汇总测速、环境及验收的综合实验报告 [baseline_report.md](../../data/benchmarks/baseline/dtk2604/reports/baseline_report.md)用途不同。

```text
使用 $profile-model，并结合 $torch-compile-integration，
基于已有 Trace/HTA 统计和项目源码生成瓶颈分析报告。

项目源码：/workspace/Isaac-GR00T
分析输出：<本次分析目录>
原始 Trace：<Trace 目录>
采集记录：<对应运行的 command.json、配置和 train.log>
对照结果（可选）：<对照方案的分析目录及采集记录>

保存为 <本次分析目录>/analysis_report.md，包含：
1. 采集配置、采样范围，以及主要发现的证据表。
2. 热点、设备间隙、搬运、通信和 rank 差异；结合源码解释，区分事实与假设。
3. 后续优化建议与验证方法；仅在配置和采样条件可比时做方案对照。

优先引用已有统计表和日志，必要时抽查原始 Trace，标明源码位置与证据缺口。
保持设备间隙、HTA idle、累计 kernel 时长的口径区别；不将 Trace 差异称为正式吞吐提升。
只做离线分析，不重新训练、安装依赖或修改模型代码。
```

## 六、可选模型编译优化

当前启动脚本和实验编排器统一清除 `TORCHINDUCTOR_CACHE_DIR`、`TRITON_CACHE_DIR`，使用框架默认路径并保留缓存以供复用，不再为每轮创建独立编译缓存。缓存命中可减少重复编译，但不能消除输入形状变化等原因触发的重编译。历史结果的原始采集条件以对应归档为准，本次修改不重算历史数据。

> 当前入口：已恢复 `gr00t/experiment/compilation.py`、6.1 的旧 CLI 字段及训练调用链，6.2／6.3 脚本可使用这些参数。环境变量入口 `GR00T_COMPILE_VARIANT=off|vl|dit-model|dit-model-vl` 仍保留，默认 `off`；启用旧 CLI 编译项时应将它设为 `off`，两种入口同时启用会报错。两者均保留原视觉 Block 编译。未恢复文本实验的 `use_cache=False` 改动；恢复入口不代表文本编译已重新通过 GPU 验收。批量采集编排脚本（原 `scripts/zero2_comm_experiment/`、`scripts/compile_sequence/`）已随优化结束移除，需要复算时使用各归档内的 `source-snapshot/scripts/`。


### 6.1 编译范围与选择

本节介绍可选的新增编译范围。它们默认关闭；关闭这些开关不会关闭已有的视觉 Block 编译。新增编译均使用 `fullgraph=False`，`dynamic` 保持 PyTorch 默认设置。

| VARIANT | 开关 | 编译目标 | 可用示例 |
| --- | --- | --- | --- |
| `baseline` | 新增编译开关均关闭 | 保留已有视觉 Block 编译，作为测速对照 | 仅 6.3 |
| `blocks` | `--compile-dit-blocks` | 动作 DiT 的 32 块，逐实例编译 | 6.2、6.3 |
| `model` | `--compile-dit-model` | 整个动作 DiT（`action_head.model`）作为一个区域；与 blocks 互斥 | 6.2、6.3 |
| `model_vl` | `--compile-dit-model` + `--compile-vl-self-attention-blocks` | 同时编译整个动作 DiT 与视觉语言自注意力的 4 块；组合方案 | 6.3 |
| `text_v2` | `--compile-text-layer-forwards` | 文本 Decoder 16 层的 `forward`；编译边界在隐藏状态收集包装之内 | 6.2、6.3 |
| `vl` | `--compile-vl-self-attention-blocks` | 视觉语言自注意力的 4 块 | 6.2、6.3 |
| 历史方案 | `--compile-text-layers` | 旧文本层编译路径；真实训练触发重编译上限，与 `text_v2` 互斥 | 不在当前 VARIANT 列表 |
| 通用参数 | `--compile-mode default` | 编译模式，不代表独立编译范围 | 6.2、6.3 |

本节脚本按 `VARIANT` 选择单项方案；`model_vl` 是整体 DiT 与 VL 自注意力的组合测速，不能单独归因任一项的收益。训练入口参数放在 `examples/finetune.sh` 的 `--` 之后。编译开关与 HIP 自动 math、数值检查互相独立。关闭某项时，在新进程中移除对应开关或传入 `--no-...`；编译安装或首次执行失败会直接报错，不会静默回退。

`--compile-text-layer-forwards` 通过 Transformers 调用期间替换并在正常返回后恢复各层实例的 `forward`。若执行中抛出异常，不要继续复用该模型；保存时使用 `state_dict`，并在新实例上重新安装编译。

**选择方案：** 在同一 Bash 会话中设置 `VARIANT`，再运行对应小节的完整脚本。例如：

```bash
VARIANT=text_v2  # 本次只启用文本 Decoder 16 层 forward 编译
```

切换方案时重新设置 `VARIANT` 并启动新进程。编译参数的传递与生效路径如下：

```text
命令行参数
→ finetune.sh 解析自身参数，并透传 -- 后的训练参数
→ launch_finetune.py 用 Tyro 解析为 FinetuneConfig
   （未传对应编译开关时，使用 finetune_config.py 中该字段的默认值）
→ launch_finetune.py 创建 base_config.Config
   └─ training_config.py 提供 Config.training 的初始默认值
→ launch_finetune.py 将 FinetuneConfig 的编译值写入 Config.training，覆盖初始值
→ experiment.py 构建模型、安装编译包装并启动训练
→ 首次执行对应模块时，torch.compile 才进行捕获和编译
```

在本节脚本中，`VARIANT` 由 Bash 用来选择要传入的编译开关；Python 训练入口接收的是该开关，不会直接读取 `VARIANT`。因此 `training_config.py` 的默认值在标准微调启动中会被 `FinetuneConfig` 的值覆盖。

### 6.2 功能与数值检查

**用途：验证短程训练功能和数值，不用于测速或采集 Trace。** 示例运行 8 卡、20 步训练，启用数值检查，并在正常结束时保存一次模型；同时检查是否触发重编译上限。命中上限时，未能编译的部分可能回到 eager，因此应结合日志确认实际状态。数值检查和诊断日志会增加开销，耗时不能用于性能比较。

```bash
set -euo pipefail
cd /workspace/Isaac-GR00T
VARIANT="${VARIANT:-blocks}"  # blocks | model | text_v2 | vl
case "$VARIANT" in
  blocks)     compile_args=(--compile-dit-blocks) ;;
  model)      compile_args=(--compile-dit-model) ;;
  text_v2)    compile_args=(--compile-text-layer-forwards) ;;
  vl)         compile_args=(--compile-vl-self-attention-blocks) ;;
  *) echo "Unknown VARIANT: $VARIANT" >&2; exit 1 ;;
esac
export VLM_MODEL_PATH=/workspace/data/checkpoints/nvidia/Cosmos-Reason2-2B
export HF_HOME=/workspace/data/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NUM_GPUS=8 MAX_STEPS=20 GLOBAL_BATCH_SIZE=32 USE_WANDB=0
export DATALOADER_NUM_WORKERS=4 MASTER_PORT=29687
export TORCH_LOGS=recompiles,graph_breaks  # 记录重编译原因与图断位置
unset GR00T_DIT_SDPA_MODE VIDEO_DECODER_BACKEND
output_dir="/workspace/data/outputs/compile-check-$VARIANT-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$output_dir"
unset TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR  # 复用框架默认编译缓存
script_args=(
  --base-model-path /workspace/data/checkpoints/GR00T-N1.7-3B-local-cosmos
  --dataset-path /workspace/data/libero_10_no_noops_1.0.0_lerobot_h264
  --embodiment-tag LIBERO_PANDA
  --output-dir "$output_dir"
  --save-only-model
)
training_args=(
  "${compile_args[@]}" --compile-mode default
  --check-numerics --gradient-accumulation-steps 1
  --save-strategy no --save-final-model
)
bash examples/finetune.sh "${script_args[@]}" -- "${training_args[@]}" 2>&1 | tee "$output_dir/train.log"
if grep -Eq 'hit config\.(accumulated_)?recompile_limit' "$output_dir/train.log"; then
  echo "触发重编译上限，部分模块已回退 eager：$output_dir/train.log" >&2
  exit 1
fi
```

### 6.3 编译测速对照

**用途：比较短程吞吐，不验证数值正确性。** 按 [Baseline 采集口径](optimization/groot_dtk_optimization_record.md#52-采集方法)运行 8 卡、全局 batch 32、每方案 80 步 × 3 轮；每轮独立进程，复用框架默认 Inductor/Triton 缓存，关闭 Profiler、数值检查和图断日志，不保存模型。建议先测 `VARIANT=baseline`，再测单项编译方案；也可用 `VARIANT=model_vl` 同时测试整体 DiT 与 VL 编译。组合结果不能替代两个单项结果，且不应据此拆分各自收益。尽量保持机器与时段一致。以下实测汇总仅作参考。

```bash
set -euo pipefail
cd /workspace/Isaac-GR00T
VARIANT="${VARIANT:-blocks}"  # baseline | blocks | model | model_vl | text_v2 | vl
export GR00T_COMPILE_VARIANT=off  # 使用旧 CLI 入口，避免与环境变量入口冲突
case "$VARIANT" in
  baseline) compile_args=(--no-compile-dit-blocks --no-compile-dit-model
                          --no-compile-text-layers --no-compile-text-layer-forwards
                          --no-compile-vl-self-attention-blocks) ;;
  blocks)   compile_args=(--compile-dit-blocks) ;;
  model)    compile_args=(--compile-dit-model) ;;
  model_vl) compile_args=(--compile-dit-model --compile-vl-self-attention-blocks) ;;
  text_v2)  compile_args=(--compile-text-layer-forwards) ;;
  vl)       compile_args=(--compile-vl-self-attention-blocks) ;;
  *) echo "Unknown VARIANT: $VARIANT" >&2; exit 1 ;;
esac
export VLM_MODEL_PATH=/workspace/data/checkpoints/nvidia/Cosmos-Reason2-2B
export HF_HOME=/workspace/data/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NUM_GPUS=8 MAX_STEPS=80 GLOBAL_BATCH_SIZE=32 USE_WANDB=0
export DATALOADER_NUM_WORKERS=4
unset GR00T_DIT_SDPA_MODE VIDEO_DECODER_BACKEND TORCH_LOGS TORCH_COMPILE_DEBUG
root="/workspace/data/outputs/compile-bench-$VARIANT-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$root"
for round in 1 2 3; do
  output_dir="$root/speed-r$round"
  mkdir "$output_dir"
  unset TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR  # 复用框架默认编译缓存
  export MASTER_PORT=$((29690 + round))
  script_args=(
    --base-model-path /workspace/data/checkpoints/GR00T-N1.7-3B-local-cosmos
    --dataset-path /workspace/data/libero_10_no_noops_1.0.0_lerobot_h264
    --embodiment-tag LIBERO_PANDA
    --output-dir "$output_dir"
  )
  training_args=(
    "${compile_args[@]}" --compile-mode default
    --no-check-numerics --log-step-metrics --gradient-accumulation-steps 1
    --save-strategy no --no-save-final-model
  )
  bash examples/finetune.sh "${script_args[@]}" -- "${training_args[@]}" 2>&1 | tee "$output_dir/train.log"
done
printf 'Results: %s\n' "$root"
```

同时启用整体 DiT 与 VL 编译时，在同一 Bash 会话中先设置 `VARIANT=model_vl`，再运行上述完整脚本；也可将脚本保存后用 `VARIANT=model_vl bash <脚本路径>` 启动。默认值仍为 `blocks`。

吞吐按 `32 × 60 / max_rank(sum(step_seconds[20:80]))` 计算；逐步耗时先取 8 rank 最大值，再统计第 21–80 步的中位数与 P90。P90 使用 `np.percentile(per_step, 90)`。以下命令复用同一终端的 `root`；换终端后需先将其设为实际输出目录。

```bash
python - "$root" <<'EOF'
import glob, json, statistics, sys
import numpy as np
for run in sorted(glob.glob(f"{sys.argv[1]}/speed-r*")):
    ranks = [json.load(open(f)) for f in glob.glob(f"{run}/step_metrics/rank-*.json")]
    assert len(ranks) == 8 and all(len(r["step_seconds"]) >= 80 for r in ranks), run
    window = [r["step_seconds"][20:80] for r in ranks]
    per_step = [max(s) for s in zip(*window)]
    throughput = 32 * 60 / max(sum(w) for w in window)
    p90 = float(np.percentile(per_step, 90))
    gib = lambda key: max(r[key] for r in ranks) / 2**30
    print(f"{run}: {throughput:.3f} samples/s, median {statistics.median(per_step) * 1e3:.3f} ms, "
          f"P90 {p90 * 1e3:.3f} ms, first step {max(r['step_seconds'][0] for r in ranks):.3f} s, "
          f"allocated {gib('max_memory_allocated_bytes'):.3f} GiB, "
          f"reserved {gib('max_memory_reserved_bytes'):.3f} GiB")
EOF
```

### 6.4 实测汇总

**Batch 768 重测（2026-10-27）。** 八卡、BF16、ZeRO-2、每卡96、累积1、worker4/rank；A/B/AllGather 全部关闭，保留原视觉编译和 HIP math SDPA。每组两轮测速和一次独立八 rank Trace，均运行100步。历史 E0 与第七章使用相同数据，仅用于跨批次参照。

| 配置 | 第一轮 samples/s | 第二轮 samples/s | 两轮中位数 samples/s | 相对历史 E0 | Trace |
|---|---:|---:|---:|---:|---|
| E0（历史基线） | 583.179 | 585.280 | 584.229 | — | 历史归档 |
| VL | 591.825 | 593.911 | **592.868** | **+1.48%** | valid |
| DiT model | 620.139 | 622.212 | **621.176** | **+6.32%** | valid |
| DiT model + VL | 632.454 | 633.892 | **633.173** | **+8.38%** | valid |

统一统计第 71–80 步，每轮吞吐为 `768 × 10 / max_rank(sum(step_seconds[70:80]))`，再取两轮中位数。以上是后段窗口的性能参考，不包含前期编译和重编译代价；原测速轮次按第 21–100 步验收时均无效。历史 E0 与编译结果来自不同批次，不作为受控提速结论。[逐 rank 步耗时](../../data/benchmarks/optimizations/torch-compile-b768-20261027-003126/reports/rank-step-metrics.csv)。

当前启停方式（新进程生效）：

```bash
export GR00T_COMPILE_VARIANT=vl           # VL
export GR00T_COMPILE_VARIANT=dit-model    # 整体DiT
export GR00T_COMPILE_VARIANT=dit-model-vl # 组合
export GR00T_COMPILE_VARIANT=off          # 关闭新增编译，保留原视觉Block编译
```

**以下为历史 batch 32 数据，与本批分开阅读。** 8 卡、全局 batch 32；统一统计第 71–80 步，每轮吞吐按 `32 × 10 / max_rank(sum(step_seconds[70:80]))` 计算，再取三轮中位数。基线使用 reference 的三轮数据，与编译实验来自不同批次。

| 方案 | 第 1 轮 | 第 2 轮 | 第 3 轮 | 中位吞吐 | 相对基线 | 说明 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 106.448 | 103.449 | 107.168 | 106.448 | — | reference 三轮 |
| DiT blocks | 116.231 | 115.528 | 115.987 | 115.987 | +8.96% | 三轮 |
| DiT model | 119.476 | 109.522 | 119.652 | 119.476 | +12.24% | 三轮；第 2 轮末段较慢 |
| VL | 108.679 | 108.232 | 109.718 | 108.679 | +2.10% | 三轮 |
| text_v2 | 107.416 | 108.965 | 104.339 | 107.416 | +0.91% | 原验收三轮均无效，仅作诊断 |

吞吐单位为 samples/s。末 10 步复算不改变原实验的有效性判定，也不包含窗口外的编译停顿；相对基线为跨批次参考。参见[基线原始数据](../../data/benchmarks/baseline/reference/speed/)、[编译实验逐 rank 逐步数据](../../data/benchmarks/comparisons/compile/speed/analysis/results/per_step.csv)与[原实验报告](../../data/benchmarks/comparisons/compile/reports/benchmark_report.md)。
### 6.5 重编译优化

#### 6.5.1 Cosmos3：对长度参数分桶

`torch.compile` 首次遇到输入时会捕获并编译计算图，同时记录该版本适用的条件，即 **guard**，例如张量的 shape、dtype 或某个 Python 整数的值。后续调用先检查 guard：条件满足时复用已有版本；不满足时可能生成新的编译版本。重编译会带来额外启动开销，但并非每次输入变化都一定触发重编译。

Cosmos3 的 FlashAttention 调用需要 Python 整数 `max_seqlen`，表示该批序列的最大长度。编译器可能按该整数的精确值建立 guard：不同 batch 的最大长度从 1500 变为 1600 时，即使调用的算法相同，也可能产生新的编译版本。

Cosmos3 把这个启动参数映射到有限的上限：`1024、2048、4096、8192、16384、32768、65536`。选择规则是**取不小于实际最大长度的最小桶**，例如：

```text
实际最大长度 1500 → 桶上限 2048
实际最大长度 1600 → 桶上限 2048
实际最大长度 2100 → 桶上限 4096
```

因此，1500 和 1600 对应的 `max_seqlen` 参数相同，有机会复用该参数相关的编译版本。桶不是按 batch size 直接选择的：先求本次所需的最大序列长度，再选择能容纳它的上限。实现对零长度返回 0，超过 65536 则报错。

这里**分桶的是 FlashAttention 的最大长度参数**：真实 token、累计长度和原始长度字段仍然保留，不把序列补齐到 2048。当前实现仅在 HIP 编译且存在对应桶字段时使用桶值；其他条件下读取原始长度。张量 shape 和其他 guard 仍可能变化，所以不保证完全消除重编译。参见 [Cosmos3 长度分桶](https://github.com/iMountTai/TurboPhysAI/blob/main/turbo_physai/optimizations/models/cosmos3/sequence_packing.py)和 [attention 调用](https://github.com/iMountTai/TurboPhysAI/blob/main/turbo_physai/optimizations/models/cosmos3/mot_attention.py)。

#### 6.5.2 GR00T：输入长度为什么变化

当前 LIBERO 数据中，每个样本包含两路图像和一条任务指令。GR00T 先把图像和指令组织成消息、套用对话模板，再由 processor 转换为模型输入。输入序列包含**图像 token、文本 token 和模板／特殊 token**。不同任务的语言指令不同，token 化后的长度也可能不同。以下是真实 LIBERO 指令在本次采样中的总有效长度，包含图像和模板，并非纯文本长度：

- `turn on the stove and put the moka pot on it`：149 token。
- `put the white mug on the left plate and put the yellow and white mug on the right plate`：156 token。

本次两路图像经处理后均为 `256×256`，对应网格各为 `1×16×16`；按当前空间合并配置，两路约占 128 个图像 token，其余来自文本和模板。图像数量、尺寸、文本或模板改变时，总长度也可能改变。数据整理器使用 `padding=True`，把**每个 rank 当前 batch** 的输入补到该 batch 最长的有效长度，例如：

```text
batch A：样本长度 [149, 150, 152, 156] → 全部补齐到 156
batch B：样本长度 [149, 150, 151, 152] → 全部补齐到 152
```

所以 batch size 不直接等于序列长度，它影响每批有哪些样本，进而影响该批最长长度。不同 rank 即使在同一步，也可能读到不同样本并使用不同补齐长度。编译后的 VL 特征和 DiT mask 随之改变形状，可能触发形状 guard。本次离线采样观察到的有效长度为 145–156，不能视为完整数据集的长度上限。

#### 6.5.3 GR00T：两种补齐策略

本次与 Cosmos3 的参数分桶不同：在 processor 按 batch 最长长度完成补齐后，进一步补齐输入张量，使模型看到较少的序列形状。

- **双桶 152／156**：batch 最长有效长度不超过 152 时补到 152，否则补到 156。短 batch 不必一律补到 156，尝试减少 padding 计算。
- **单桶 156**：所有 batch 补到 156，只有一个目标序列长度；短 batch 需要多处理一些 padding。

例如最长长度为 149 的 batch，双桶补到 152，单桶补到 156；最长长度为 155 的 batch，两者都补到 156。选择按**整个 batch 的最长有效长度**进行，不是为每个样本单独选择。长度超过 156 时直接报错，不截断。新增位置使用左 padding 和零 attention mask，额外 padding 增加计算和显存开销。例如 149 补到 156，序列长度增加约 4.7%，随长度平方增长的自注意力计算理论上增加约 9.6%。

```bash
export GR00T_COMPILE_SEQUENCE_POLICY=pad-152-156  # 双桶
export GR00T_COMPILE_SEQUENCE_POLICY=pad-156      # 单桶
export GR00T_COMPILE_SEQUENCE_POLICY=auto         # 回退原补齐，默认
```

#### 6.5.4 实验结果

四卡（物理 GPU 4–7）、全局 batch 768、每卡 192；各组均使用 DiT model＋VL 编译，运行 80 步。吞吐统计第 71–80 步，以最慢 rank 的累计耗时计算。

| 策略 | 末10步第1轮 | 末10步第2轮 | 末10步平均 | 相对基线 | 第20–80步第1轮 | 第20–80步第2轮 | 全程重编译次数 |
|---|---:|---:|---:|---:|---:|---:|---|
| 原补齐基线 | 355.945 | — | 355.945 | — | 202.558 | — | 2 |
| 双桶 152／156 | 328.341 | 350.707 | 339.524 | −4.61% | 319.792 | 345.569 | 0／0 |
| 单桶 156 | 347.224 | 352.598 | 349.911 | −1.70% | 345.404 | 347.682 | 0／0 |

表中吞吐单位均为 samples/s。末10步为第71–80步；第20–80步共61步，按最慢 rank 的累计计时计算，保留该窗口中的编译及重编译停顿。各轮复用框架默认编译缓存，运行顺序可能影响编译耗时。

重编译次数为四 rank 全程 1–80 步的合计，不包含首次编译；第 71–80 步各组均无重编译。基线在完成第 24 步后，rank 2 因长度 156→155 触发 VL 和 DiT 各一次重编译，第 25 个计时区间约 101.9 秒。表中末尾窗口不包含这段代价，因此它反映的是后段速度，不代表包含编译停顿的全程表现。

双桶的 152 桶使用次数为 0／640，实际均使用 156。每卡 batch 为 192 时，只要其中一个样本长度超过 152，整个 batch 就会选择 156；本次尚未验证短桶复用或计算收益。双桶两轮波动较大，不能将两种候选的吞吐差异直接归因于桶策略。

本轮固定补齐任务没有记录重编译，避免了基线中出现的长度失配；但观察范围只有 80 步和实际出现的形状，并不能保证其他输入也不会重编译。两种策略的后段吞吐都未超过本次基线，默认仍为 `auto`。

详见[实验报告](../../data/benchmarks/optimizations/torch-compile-padding-4gpu-20261028/report.md)、[基线 guard 日志](../../data/benchmarks/optimizations/torch-compile-padding-4gpu-20261028/baseline/recompiles-guards.txt)和[单桶四 rank Trace](../../data/benchmarks/optimizations/torch-compile-padding-4gpu-20261028/trace/profiling/)。

**八卡补齐策略实验（2026-10-28）。** 全局 batch 768、每卡96；各组80步。表中吞吐单位为 samples/s，统计口径与上表相同。

| 策略 | 末10步第1轮 | 末10步第2轮 | 末10步平均 | 相对基线 | 第20–80步第1轮 | 第20–80步第2轮 | 全程重编译次数 |
|---|---:|---:|---:|---:|---:|---:|---|
| 原补齐基线 | 631.245 | — | 631.245 | — | 220.269 | — | 10 |
| 双桶152／156 | 628.432 | 628.014 | 628.223 | -0.48% | 628.029 | 630.668 | 0／0 |
| 单桶156 | 631.754 | 634.454 | 633.104 | 0.29% | 630.892 | 632.813 | 0／0 |

双桶152使用 0/1280 个已记录 rank-batch，短桶未覆盖。基线仅一轮，且本次未验证 padding 数值等价。八卡每卡96，四卡每卡192，跨卡数结果不直接归因于桶策略。

详见[八卡实验报告](../../data/benchmarks/optimizations/torch-compile-padding-8gpu-20261028/report.md)和[原始归档](../../data/benchmarks/optimizations/torch-compile-padding-8gpu-20261028/)。

## 七、ZeRO-2 通信优化

本章比较 Gradient Reduction Overlap（GRO）、Owner-based Reduction（OBR）及 AllGather 流水。算法优化默认关闭，吞吐计时器使用 DeepSpeed 原始行为；适配只作用于当前实例，不修改厂商安装文件。

归档中的原始配置代号与本文名称对应：E0 为 Baseline（BASE），A 为 Gradient Reduction Overlap（GRO），B 为 Owner-based Reduction（OBR）。原始任务标识保留不变。

### 7.1 总览

基线配置虽开启 `overlap_comm`，本批 Trace 中 AllReduce 与 backward 的重叠仍为 0。GRO 用缓冲区 Event 依赖替代计算流逐桶等待，使梯度归约与后续计算重叠。

| 配置 | 优化内容 | 吞吐 samples/s | 相对 BASE | 轮数 | 结论 |
|---|---|---:|---:|---:|---|
| Baseline（BASE） | 原厂 ZeRO-2 梯度通信与参数更新路径 | 584.229 | 0.00% | 2 | 基线 |
| Gradient Reduction Overlap（GRO） | 用缓冲区 Event 依赖替代计算流逐桶等待 | 603.372 | **+3.28%** | 2 | **通过原实验验收，推荐** |
| Owner-based Reduction（OBR） | 按梯度 owner 选择定向 Reduce 或 ReduceScatter | 584.068 | −0.03% | 1 | 首轮未改善 |
| AllGather 流水 | 分组更新参数并异步聚合，重叠 AdamW 与 AllGather | 586.765 | +0.43% | 1 | 跨批次参考对比 |

本表统一统计每轮第 71–80 步。每轮吞吐按 `768 × 10 / max_rank(第71–80步累计耗时)` 计算；两轮取吞吐中位数。AllGather 为跨批次单轮比较，可能受外部任务影响。


### 7.2 实验口径与前提

本章吞吐对比统一按每轮第 71–80 步重新计算；原实验的稳态与验收仍按第 21–100 步执行。

| 项目 | 配置 |
|---|---|
| 硬件与软件 | 8 张 BW1000，BF16，ZeRO-2，`adamw_torch`；DeepSpeed `0.18.4+das185.dtk2604`，torch `2.10.0+das.opt1.dtk2604` |
| 训练 | 全局 batch 768，每卡 96，梯度累积 1，worker 4/rank，学习率 1e-4，seed 42，state dropout 0.2 |
| 测速 | 主实验运行 100 步，本章统计第 71–80 步；每任务独立进程，关闭 Profiler、数值扫描和模型保存 |
| 吞吐公式 | `全局 batch × 10 / max_rank(sum(step_seconds[70:80]))`；两轮取吞吐中位数 |
| 原实验稳态验收 | 第 21–100 步逐步取八 rank 最大耗时，步时间 CV ≤5%，后半窗口相对前半窗口的漂移绝对值 ≤5% |
| Trace | 八 rank，`skip_first=20, wait=1, warmup=5, active=3, repeat=1`，关闭调用栈；调用栈诊断另采 |


### 7.3 Gradient Reduction Overlap（GRO）：梯度归约与 backward 重叠

**问题。** 基线虽设置 `overlap_comm=true`，计算流仍在每个梯度桶提交后等待归约流。本批 BASE Trace 中，AllReduce 与 backward 的重叠为 0，位于 backward 之外的归约区间中位数为 126.32 ms。这说明现有流依赖没有让后续反向计算覆盖这段通信。

**优化思路。** 梯度桶使用双缓冲：半区 0 正在归约时，可以在半区 1 继续计算和收集梯度。计算流需要等待的是“即将复写的半区已被通信读完”，而非每个桶的归约都立即完成。

**关键实现。** 见 [zero2_comm.py](../gr00t/Replacement/zero2_comm.py) 与 [zero2_overlap.py](../gr00t/Replacement/zero2_overlap.py)：

1. 梯度桶满后，保留归约流等待梯度就绪的依赖，再提交 AllReduce。
2. 在归约流上为已提交的半区记录完成 Event，切换到另一半区继续接收梯度。
3. 计算流仅在复用某半区前等待其 Event，避免覆盖仍在通信的数据。
4. 用 `record_stream` 保护归约张量的生命周期。AllReduce 的切片和顺序保持不变。

时序示意（不按耗时比例）：

```text
原路径：
计算流：计算桶0 → 等桶0归约完成 → 计算桶1 → 等桶1归约完成
归约流：          归约桶0                    归约桶1

GRO：
计算流：写半区0 → 写半区1 → 复用半区0前等待 done0 → 继续写半区0
归约流：等半区0就绪 → 归约半区0 → 记录 done0 → 处理后续桶
```

**结果与结论。** 吞吐与同批 BASE 比较：

| 配置 | 吞吐 samples/s | 相对本批 BASE | 有效轮数 | 结论 |
|---|---:|---:|---:|---|
| BASE | 584.229 | 0.00% | 2 | 基线 |
| GRO | 603.372 | +3.28% | 2 | 通过原实验验收 |

八 rank、每 rank 三个采样步骤的 Trace 中，**反向传播之外的 AllReduce 区间中位数从 126.32 ms 降至 42.57 ms，缩短 83.75 ms（约 66.3%）**。这表示更多归约与反向计算同时进行；该区间不等同于训练步骤中实际等待的时间。

### 7.4 Owner-based Reduction（OBR）：按 owner 归约

**问题。** ZeRO-2 中，每个 rank 只负责更新自己的参数分片，因此只需要对应梯度。本机基线的梯度桶路径仍通过 AllReduce 汇总整桶，各 rank 随后使用自己负责的部分；配置中的 `reduce_scatter=true` 不代表这条路径已经调用原生 ReduceScatter。

**优化思路。** 按梯度 owner 组织归约，让结果到达负责更新的 rank；同时避免 owner 分布不均时为 ReduceScatter 添加过多 padding。

**关键实现。** [zero2_reduce_scatter.py](../gr00t/Replacement/zero2_reduce_scatter.py) 由 `GR00T_ZERO2_REDUCE_SCATTER=1` 启用：

1. 根据参数分区表，将桶映射为 owner、偏移和长度，合并连续且同 owner 的切片。
2. 统计各 owner 的元素数，计算补齐到最大分片长度后的总元素量。
3. 补齐后的总量不超过原桶两倍时，打包并调用原生 `reduce_scatter_tensor`；否则对切片调用定向 `dist.reduce`，避免高 padding。
4. 只使用 owner 对应的归约结果。

**结果与结论。** 本轮 OBR **实际走定向 Reduce，没有走原生 ReduceScatter**：

| 配置 | 吞吐 samples/s | 相对本批 BASE | 有效轮数 | 结论 |
|---|---:|---:|---:|---|
| BASE | 584.229 | 0.00% | 2 | 基线 |
| OBR | 584.068 | −0.03% | 1 | 首轮低于 BASE，未追加 |

### 7.5 AllGather 流水

**问题。** ZeRO-2 在各 rank 更新本地参数分片后，需要 AllGather 恢复完整参数。原路径先更新全部参数组，再聚合参数，最后一段通信缺少后续更新计算来覆盖。独立的 **batch 32** Trace 中，AllGather 与 AdamW 的重叠为 0，最后一次 AdamW 后仍有约 24.026 ms 的聚合尾部；这提供了流水原型的验证动机，不代表 batch 768 已有相同收益空间。

**优化思路。** 一组参数更新完成即可开始聚合，同时继续更新下一组，把参数更新与 AllGather 放入同一次 `optimizer.step()` 内的流水。返回前仍保证全部参数就绪。

**关键实现。** 见 [zero2_allgather.py](../gr00t/Replacement/zero2_allgather.py)：

1. 按约 100M 元素重组完整参数组，保留参数顺序、decay/no-decay 边界及 AdamW 超参，不拆单个参数。本模型得到 18 组，组数不是固定值。
2. 每组使用原有 AdamW 更新 FP32 分片，再复制到 BF16 参数分片。
3. 立即提交本组异步 AllGather，继续执行下一组更新，使用后端已有通信流。
4. 保存异步 Work 和输入/输出张量引用；聚合完成前不复写该组存储，保证依赖和生命周期正确。
5. 全部组提交后统一等待，完成权重映射，所有参数就绪才从 `step()` 返回；不跨迭代预取。

时序示意（不按耗时比例，未固定 Stream 编号）：

```text
原路径：更新全部参数组 → AllGather 各组 → 完整参数就绪

流水：
更新路径：更新组0 → 更新组1 → 更新组2 → … → 等待全部聚合 → 权重映射
通信路径：          聚合组0 → 聚合组1 → 聚合组2 → …
                   ↑ 与后续组的更新重叠
```

**结果与结论。**

| 全局 batch | 基线吞吐 samples/s | 流水吞吐 samples/s | 相对基线 | 测速口径 |
|---|---:|---:|---:|---|
| 32 | 108.699 | 112.598 | +3.59% | 同批各一轮，第 71–80 步 |
| 768 | 584.229 | 586.765 | +0.43% | 基线两轮中位数，流水跨批次一轮，第 71–80 步 |

batch 768 的流水测速是跨批次单轮参考，可能受外部任务影响；batch 32 结果来自独立历史对照。

### 7.6 启用、回退与运维

**BASE 环境／完整回退。** 启动新进程前明确重置，不能只关闭 GRO：

```bash
export GR00T_ZERO2_OVERLAP=0
export GR00T_ZERO2_REDUCE_SCATTER=0
export GR00T_ZERO2_ALLGATHER_PIPELINE=0
export GR00T_ZERO2_PACK_TIMING=0
export GR00T_COMPILE_VARIANT=off
unset GR00T_DEEPSPEED_CONFIG_PATH GPU_MAX_HW_QUEUES
```

以上环境下运行 `bash start.sh` 即为 BASE。**启用推荐的 GRO**：先按上面重置环境，再执行：

```bash
export GR00T_ZERO2_OVERLAP=1
bash start.sh
```

| 变量 | 默认 | 作用 |
|---|---|---|
| `GR00T_ZERO2_OVERLAP` | 0 | GRO：梯度归约重叠 |
| `GR00T_ZERO2_REDUCE_SCATTER` | 0 | OBR：按 owner 归约 |
| `GR00T_ZERO2_ALLGATHER_PIPELINE` | 0 | 优化器内 AllGather 流水 |
| `GR00T_ZERO2_PACK_TIMING` | 0 | 打包诊断计时，正式测速保持 0 |
| `GPU_MAX_HW_QUEUES` | 未设置 | HIP 硬件队列数 |

表中 `GR00T_*` 开关仅接受 0/1，须重启进程生效。训练结束或异常时恢复实例方法；通信异常时停止本次任务，不在运行中切回原路径。

**适用范围。** GRO/OBR 要求多卡，本章验证八卡；AllGather 原型强制八卡。使用已验证的 BF16、ZeRO-2、梯度累积 1、无 offload 配置，版本与关键源码哈希受守卫检查。流水仅支持 `adamw_torch`、`--save-strategy no`，不支持优化器 checkpoint 恢复。

`start.sh` 默认 batch 768、固定 100 步，记录逐步指标且不保存模型，与实验编排器均清除继承的编译缓存目录变量，使用框架默认缓存；它不会自动清零继承的优化开关。实验编排脚本（原 `scripts/zero2_comm_experiment/`）已随优化结束移除，复算使用各归档内的 `source-snapshot/scripts/`。原始逐步指标和生效配置随任务归档。

### 7.7 证据边界与归档

- 单轮只作筛选；Profiler 数据只解释机制，不代替正式测速。
- 固定梯度与 100 步数值检查不覆盖长训；机制验收通过不代表端到端性能有收益。
- 不同批次分别使用匹配基线，batch 32 与 batch 768 不混比。
- 原始记录与摘要冲突时，以逐轮指标、运行配置和验收结果为准；复算写入新目录，不覆盖归档。

| 内容 | 入口 |
|---|---|
| 主实验：BASE/GRO/OBR | [命名汇总](../../data/benchmarks/zero2-communication/gradient/base-gro-obr-b768-20261026/reports/named-summary.md)、[逐轮展示数据](../../data/benchmarks/zero2-communication/gradient/base-gro-obr-b768-20261026/reports/named-rounds.csv)、[原始报告](../../data/benchmarks/zero2-communication/gradient/base-gro-obr-b768-20261026/reports/report.md)、[验收](../../data/benchmarks/zero2-communication/gradient/base-gro-obr-b768-20261026/reports/acceptance.json)、[原始逐轮数据](../../data/benchmarks/zero2-communication/gradient/base-gro-obr-b768-20261026/reports/rounds.csv) |
| batch 768 仅 AllGather 流水后续测速 | [586.765 测速报告](../../data/benchmarks/zero2-communication/allgather/speed-b32-b768-20261027/report.md) |
| batch 768 仅 AllGather 流水历史实验 | [报告](../../data/benchmarks/zero2-communication/allgather/b768-20261027/reports/report.md)、[八 rank 区间分析](../../data/benchmarks/zero2-communication/allgather/b768-20261027/trace-analysis/pipeline/summary.json) |
| batch 32 AllGather 流水 | [报告](../../data/benchmarks/zero2-communication/allgather/b32-20261022/reports/retest_report.md)、[区间对照](../../data/benchmarks/zero2-communication/allgather/b32-20261022/trace-analysis/comparison/optimizer-comparison.csv) |

## 八、内存布局优化

### 8.1 优化内容与开关

`Qwen3Backbone` 初始化时，可将视觉模块中的 `Conv3d` 权重转换为 PyTorch 的 `channels_last_3d` 内存布局，并安装前置 hook，把五维输入转换为同一布局；其他维度的输入不转换，tracing 时跳过输入转换。实现见 [vision_patch_embed_layout.py](../gr00t/Replacement/vision_patch_embed_layout.py)。

该布局转换已在 [NVIDIA Isaac-GR00T N1.7 官方 main 分支](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/modules/qwen3_backbone.py)中实现。这里增加了环境变量开关和防重复安装保护，并在 DTK/MIOpen 环境下进一步测试不同布局及后端设置的训练吞吐。

权重处理的核心思路如下，实际实现还会防止重复安装 hook：

```python
# 每次前向：五维输入也使用相同内存格式；tracing 时跳过
def convert_input(module, inputs):
    if inputs and not torch.jit.is_tracing():
        x = inputs[0]
        if isinstance(x, torch.Tensor) and x.dim() == 5:
            return (x.contiguous(memory_format=torch.channels_last_3d), *inputs[1:])
    return inputs

# 模型初始化：转换视觉模块中 Conv3d 的权重布局
for module in visual.modules():
    if isinstance(module, torch.nn.Conv3d):
        module.to(memory_format=torch.channels_last_3d)
        module.register_forward_pre_hook(convert_input)
```

权重的逻辑形状仍是 `[输出通道, 输入通道, 深度, 高, 宽]`；转换改变的是内存中的排列与 stride，让输入通道在内存中相邻，并不改变卷积的维度定义。输入同样保留 `[批次, 通道, 深度, 高, 宽]` 的逻辑形状。

环境变量 `GR00T_VISION_PATCH_EMBED_CHANNELS_LAST` 默认是 `1`，设为 `0` 则在新建模型时跳过权重转换和 hook 安装。只接受 `0` 或 `1`，其他值会报错。开关在构建模型时读取，切换后需重启进程；已经转换的权重不会在同一进程内自动还原。

启动训练前启用或回退：

```bash
# 启用；未设置时也默认启用
export GR00T_VISION_PATCH_EMBED_CHANNELS_LAST=1

# 回退：在新的训练进程中使用默认布局
export GR00T_VISION_PATCH_EMBED_CHANNELS_LAST=0
```

`PYTORCH_MIOPEN_SUGGEST_NHWC` 和 `MIOPEN_FIND_MODE` 是独立的后端设置。未设置表示保留后端默认行为，不等于强制关闭布局建议；不需要设置它们才能使用上述布局开关。下面 C、D 组分别用于观察显式设置后的结果，不能由变量值直接推断后端实际选择的 kernel 或是否触发搜索。

### 8.2 测速口径与结果

两组 batch 的测试均使用 8 卡、100 步、BF16、ZeRO-2、`adamw_torch`、梯度累积 1，关闭 profiler、数值扫描和模型保存。吞吐取第 **21–30 步**，对每个 rank 累计 10 步耗时后取最大值：

```text
samples/s = global_batch_size × 10 / max_rank(sum(step_seconds[20:30]))
```

这里的切片是从零开始的索引，对应训练第 21–30 步。以下结果仅比较同一 batch 内的配置；小窗口结果不代表完整训练的平均吞吐。

**全局 batch 768**：A 组两轮加一次独立复测，B、C、D 各两轮。表中中位数使用列出的全部轮次。

| 组别 | 布局开关 | `PYTORCH_MIOPEN_SUGGEST_NHWC` | `MIOPEN_FIND_MODE` | 各轮 samples/s | 中位数 samples/s | 相对前一组 |
|---|---:|---|---|---|---:|---:|
| A | 0 | 未设置 | 未设置 | 393.82、393.10、395.6 | 393.82 | — |
| B | 1 | 未设置 | 未设置 | 581.09、586.31 | 583.70 | +48.22% |
| C | 1 | 1 | 未设置 | 588.67、585.49 | 587.08 | +0.58% |
| D | 1 | 1 | 1 | 591.43、590.65 | 591.04 | +0.68% |

**全局 batch 32**：A、B 各一轮，结果仅作单轮参考。

| 组别 | 布局开关 | MIOpen 两变量 | samples/s | B 相对 A |
|---|---:|---|---:|---:|
| A | 0 | 未设置 | 99.13 | — |
| B | 1 | 未设置 | 107.39 | +8.33% |

### 8.3 验证范围与归档

现有 [CPU 测试](../tests/gr00t/Replacement/test_vision_patch_embed_layout.py)覆盖默认和非法开关、权重及输入布局、前向输出和梯度一致性、tracing 跳过、重复安装和 `Qwen3Backbone` 方法。实际训练 checkpoint 的保存与加载、ZeRO-3 是否保持权重布局、视觉 ONNX 导出到 TRT，以及长期训练尚未专项验证；上述吞吐实验也不构成数值等价或长期稳定性证明。

| 内容 | 入口 |
|---|---|
| batch 768：21–30 步对照与逐轮数据 | [对照报告](../../data/outputs/vision-layout-miopen-b768-20261028-151751-1244744/comparison-steps-21-30-abcd.md)、[原始实验归档](../../data/outputs/vision-layout-miopen-b768-20261028-151751-1244744/report.md)、[A 复测归档](../../data/outputs/vision-layout-a-retest-20261028-160737-2690966/report.md) |
| batch 32：A、B 原始逐 rank 指标 | [A 归档](../../data/outputs/vision-layout-a-b32-100step-20261028-161855-2875875/attempt-A-b32/step_metrics/)、[B 归档](../../data/outputs/vision-layout-b-b32-100step-20261028-162246-2972435/attempt-A-b32/step_metrics/) |

batch 32 的 B 组归档子目录沿用了 `attempt-A-b32` 名称；其 [运行命令和环境](../../data/outputs/vision-layout-b-b32-100step-20261028-162246-2972435/attempt-A-b32/command.json)记录的布局开关为 `1`。
