# Apex FusedAdam 优化项

`GR00T_APEX_FUSED_ADAM` 默认关闭，只接受 `0` 和 `1`。通过 Transformers 现有的 `adamw_apex_fused` 工厂创建 `apex.optimizers.FusedAdam`，不重写参数分组或更新公式。

从仓库根目录启动新进程：

```bash
# 启用 Apex FusedAdam（AdamW 模式）
GR00T_APEX_FUSED_ADAM=1 bash start.sh

# 回退到微调入口原有的 torch.optim.AdamW
GR00T_APEX_FUSED_ADAM=0 bash start.sh
```

取消设置变量也会恢复原配置。开关在每个训练进程启动时读取，不支持在训练中切换。关闭时不额外导入 Apex；若直接加载的训练配置本身已指定 Apex，关闭开关会保留这个显式配置。标准微调入口原配置为 `adamw_torch`。

实现位于 `gr00t/Replacement/fused_adam.py`，由 `experiment.run()` 在选择 Trainer 和保存配置之前解析。学习率、betas、eps、weight decay 和不衰减参数分组继续由原 Trainer 路径管理；BF16 和 ZeRO 精度管理继续使用原训练栈，不启用 Apex AMP。Apex 的 `adam_w_mode=True` 保持 AdamW 的解耦权重衰减。

启用要求兼容当前 PyTorch/设备的 Apex 编译扩展。本机是 DTK 适配版 Apex；其他环境需要相应的构建。缺少 Apex 或多张量扩展会明确报错，不自动回退。

当前限制：

- 可与 `GR00T_ZERO2_ALLGATHER_PIPELINE=1` 组合；当前只验证 CPU 层参数组适配，八卡数值与性能尚未验证。仅支持 AdamW 模式，`capturable` 与 `master_weights` 必须关闭。
- DeepSpeed 配置不能包含独立的 `optimizer` 定义，以免绕过 Trainer 工厂；不支持 optimizer CPU/NVMe offload。
- 不会把 Adafactor 等非 AdamW 算法静默替换为 Apex。
- 旧 AdamW 优化器 checkpoint 到 Apex 的状态迁移未验证；做对照实验时从相同模型权重开始新训练。

训练开始后会核对包装器内的实际优化器类型及 AdamW 模式，并在 `<output_dir>/optimizer/rank-XX.json` 保存类名、Apex 版本和实际各参数组超参数；生效配置中的 `training.optim` 为 `adamw_apex_fused`。

CPU 单测：

```bash
NO_ALBUMENTATIONS_UPDATE=1 python -m pytest tests/gr00t/Replacement/test_fused_adam.py -q
```

单测覆盖开关回退、错误值、AllGather 兼容与 DeepSpeed 配置冲突、依赖缺失、训练入口选择、参数分组与运行记录。测试中用 CPU 优化器替代 GPU 依赖，不构成 Apex kernel、ZeRO 分布式运行、数值等价或性能验证。
