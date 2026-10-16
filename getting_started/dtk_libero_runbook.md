# DTK 环境下 GR00T、Cosmos 与 LIBERO 运行手册

适用于当前 `/workspace/Isaac-GR00T` 容器。内容按数据与模型、DTK 兼容修复、依赖与验收组织；容器外辅助脚本位于共享工作区 `/workspace/scripts`，不随单独克隆本仓库自动提供。

仓库同时保存了[优化适配记录](optimization/groot_dtk_optimization_record.md)、[Trace 分析报告](optimization/groot_torch_compile_trace_analysis.md)及 [torchcodec 补丁](../scripts/patches/torchcodec_cpu_hip_sync.patch)。模型、数据、训练输出、DTK 依赖与其余外部辅助脚本需单独准备；上游示例素材保留原有 Git LFS 指针，子模块保留引用，本次代码归档不镜像这些资源。

## 当前状态

- 自动 math 配置已通过 8 卡、batch 64、BF16 的 100 步数值验证：取消 `GR00T_DIT_SDPA_MODE`，8 个 worker 均确认变量未设置，耗时 125.16 秒，平均 loss 为 1.184575，与显式 math 对照一致。各 rank 的 loss、梯度、优化器状态、更新后参数和最终内存权重检查通过，存在非零梯度和参数更新。未保存模型，进程已退出。报告见 2.5；这不代表厂商默认后端、长期训练或部署验收通过。

- 环境：DTK-26.04-rc4、Python 3.10.12；容器已识别 8 张 BW1000 64G 加速卡。当前设备节点已恢复，无需重建容器。
- 资源：GR00T、Cosmos 和 LIBERO H.264 数据均已准备好，不要重复下载或把原始 AV1 snapshot 覆盖到转码目录。
- 历史流程测试：2026-10-13 的 8 卡、batch 8 两步测试完成并保存模型，但当时未检查梯度和输出权重，不能据此判定有效训练通过。日志：[libero-smoke-20261013-162451.log](/workspace/data/logs/libero-smoke-20261013-162451.log)。
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

一次性实验脚本及对照参数已从仓库清理，历史日志保持不变。当前可用安全检查与命令见 3.3；不能只根据退出码、保存成功或日志低 loss 判断训练有效。

## 三、依赖项与验收

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

### 3.2 已完成的验收

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

### 3.3 100 步训练验证

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
output_dir="/workspace/data/outputs/libero-check-$(date +%Y%m%d-%H%M%S)-$$"  # 独立输出目录
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

### 3.4 正式训练示例（尚未验收）

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

### 3.5 可选数值检查

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

### 3.6 保存策略与剩余验收

| 参数组合 | 行为 |
| --- | --- |
| `--save-strategy no --no-save-final-model` | 不保存中途或最终模型；100 步验证采用此组合 |
| `--save-strategy no --save-final-model` | 正常结束仅保存一次模型；正式示例采用此组合 |
| 脚本参数 `--save-only-model` | 仅保存模型，不保存优化器/调度器/RNG，不能精确恢复训练 |

数值检查与保存策略独立：启用检查不会自动禁用正常保存，关闭检查也不会关闭 HIP math。最终模型约 6.5 GB，配置与日志仍会写入输出目录。

后续验收按单卡 20 步、8 卡 batch 32 的 50 步、batch 64 的 100 步和 1,000 步推进；任一失败即停止。仍待验证厂商默认 SDPA 的具体内核缺陷、兼容配置的长期稳定性及保存后重载动作预测；完整 10,000 步训练、策略服务和机器人/仿真闭环均未验收。历史两步测试、含 NaN 的长训练输出以及本次 100 步结果都不能证明成功部署。

### 3.7 抓取 Profiler Trace

现有实现位于 `gr00t/experiment/experiment.py` 的 `enable_profiling` 分支：用 `torch.profiler.profile` 包裹训练，`gr00t/experiment/trainer.py` 的 `ProfCallback.on_step_end()` 在每个优化器步结束调用 `prof.step()`，通过 `export_chrome_trace()` 输出 JSON。默认关闭；脚本分隔符 `--` 后加 `--enable-profiling` 即可启用，不需要新增 profiler 代码。

#### 3.7.1 具体采集命令

先人工确认 8 卡空闲及端口未占用，规则见 3.3。下面是可独立执行的 Bash 脚本：8 卡、全局 batch 32、每卡 4、训练 40 步，覆盖当前 `warmup=20` 的采集窗口，关闭数值检查、W&B 和全部模型保存；每个 rank 都会采集。使用当前自动 math 配置，不切回存在数值风险的默认 SDPA。

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
output_dir="/workspace/data/outputs/libero-profile-$(date +%Y%m%d-%H%M%S)-$$"
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

此命令会实际训练 40 步，并非纯静态采集；采集尝试结果见 3.7.4，目前尚未确认当前 DTK 的设备事件是否完整导出。更换卡数或 batch 做比较时，应保持每卡 batch、数据加载等条件可比；单卡会走不同于多卡 DeepSpeed 的路径，不能直接当作相同配置。

#### 3.7.2 采集窗口与输出

当前代码固定配置如下，这些值不是已有的 CLI 参数：

```python
schedule=torch.profiler.schedule(skip_first=1, wait=1, warmup=20, active=1, repeat=1)
```

即跳过 1 步、等待 1 步、预热 20 步，正式记录 1 步，只采集一轮；通常在约第 23 个优化器步导出，文件名以回调实际记录为准。若只训练 20 步，会在 warmup 阶段结束，正常退出也不会导出 trace。总步数应覆盖 `skip_first + wait + warmup + active`；将 `MAX_STEPS` 改为 100 不会自动扩大采集窗口。当前启用 CPU/CUDA activity 和 `with_stack=True`，`record_shapes` 与 `profile_memory` 尚未开启。

输出位于 `<output-dir>/profiling/trace_rank_<rank>_iter_<step>.json`；8 卡通常各生成 1 份，rank 编号为 0～7。退出后在同一 Bash 会话中查看：

```bash
printf '本次输出目录：%s\n' "$output_dir"
find "$output_dir/profiling" -maxdepth 1 -type f -name 'trace_rank_*.json' -printf '%f  %s bytes\n'
rg -n 'Starting training with profiling|Trace saved|Traceback|Error' "$output_dir/train.log"
```

若换了终端，先手动把 `output_dir` 赋为实际输出目录。没有文件时先核对开关、运行步数、退出错误和写入权限，不要只检查进度条。保留各 rank 文件，先打开 rank 0 分析，再比较其他 rank 的等待与负载差异。

#### 3.7.3 阅读与性能注意事项

Chrome trace JSON 可用本地 trace viewer 或 Perfetto UI 打开；日志可能包含源码路径、调用栈和算子信息，使用在线查看器前按内部数据管理要求确认是否允许导入，不上传未脱敏文件。

- DTK/HIP 仍使用 `ProfilerActivity.CUDA` 接口；JSON 存在只证明导出执行过，不证明已包含设备活动。打开后检查是否有设备 kernel、运行时调用、HtoD/DtoH 和通信相关轨道；只有 CPU 事件时需单独检查当前 PyTorch/profiler 后端支持，不能据此判断 GPU 空泡。
- 先看主机/设备时间线、kernel 间隙、Memcpy 与通信，再关联算子和调用栈。间隙可能来自数据等待、同步、通信或编译，不能只凭空白区域直接归因。
- 当前窗口较早，只有 1 个 active 步，可能仍包含首次编译、autotune 或分片预加载。分析稳定性能时，应在 `experiment.py` 调大 `skip_first`、按需增加 `active`，并确保总训练步数覆盖完整窗口；本次不修改这些代码。
- 所有 rank 采集及 `with_stack=True` 会增加开销与文件体积，采样窗口内吞吐不代表无 profiler 的稳定吞吐。`--no-check-numerics` 只关闭数值扫描，不会消除 profiler 或 math 后端自身的开销；本次采样也不作为数值稳定性验收。

#### 3.7.4 本次采集记录（2026-10-15）

- `libero-profile-20261015-155115-2122776`：20 步完成，退出码 0；实际源码 warmup 为 20，未进入 active 窗口，没有 trace。该尝试证明训练完成，不证明采集成功。
- `libero-profile-20261015-155941-2210704`：保持源码配置不变，改跑 40 步；rank 2 在模型迁移时 HIP OOM，其他 rank 随后退出，没有 trace。失败后设备显存占用约 94%～98%，本次 worker 已退出，不继续争用或自动停止其他任务。
- `libero-profile-20261015-161440-2221399`：资源释放后运行 40 步，8 个 rank 均成功导出 `trace_rank_<rank>_iter_23.json`，训练退出码 0，未保存模型。8 份 trace 共 665,626,025 bytes；rank 0 JSON 校验通过，含 300,420 个事件，包括 `kernel`、`cuda_runtime`、`gpu_memcpy`、`ac2g` 和 `cpu_op` 类别。文件位于该目录的 `profiling/` 下，可直接用于时间线分析。

前两次尝试目录及日志均保留在 `/workspace/data/outputs/`。成功目录同时保留 `train.log`、训练配置和 8 份 trace；本次仅完成短程采集，不代表数值稳定性或正式训练验收。
